import numpy as np
from tqdm import tqdm
from math import exp
import os
import sys
sys.path.append("..")
import signal
import json
import argparse
import pickle as pkl
from dataset import dataset,CRSdataset
from model import CrossModel
import torch.nn as nn
from torch import optim
import torch
from .con_rl_search import batch_concept_beam_search
from collections import defaultdict
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TrainLoop_fusion_concept_rl():
    def __init__(self, opt, is_finetune):
        self.opt=opt
        self.train_dataset=dataset('data/train_data.jsonl',opt)

        self.dict=self.train_dataset.word2index
        self.index2word={self.dict[key]:key for key in self.dict}

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune)

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self,is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def torch_init_model(self):
        state_dict = torch.load(self.opt["model_path"])
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        # print('*'*10, type(state_dict), '*'*10)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self.model)

        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))
        
    def train(self):
        #self.model.load_model()
        losses=[]
        best_val=0.0 
        rl_stop=False
        with torch.no_grad():
            db_nodes_features = self.model.dbpedia_RGCN(None, self.model.db_edge_idx, self.model.db_edge_type)
            con_nodes_features = self.model.concept_GCN(self.model.concept_embeddings.weight,self.model.concept_edge_sets)
        db_node = db_nodes_features.cpu().detach().numpy()
        con_node = con_nodes_features.cpu().detach().numpy()

        total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
        step = 0
        self.model.train()
        for i in range(self.opt["epoch"]):
            train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,response_concept_mask in tqdm(train_dataset_loader):

                with torch.no_grad():
                    seed_sets = []
                    batch_size = context.shape[0]
                    for b in range(batch_size):
                        seed_set = entity[b].nonzero().view(-1).tolist() 
                        seed_sets.append(seed_set) 
                    user_representation_list = []
                    db_con_mask=[]
                    for i, seed_set in enumerate(seed_sets): 
                        if seed_set == []:
                            user_representation_list.append(torch.zeros(self.model.dim).cuda())
                            db_con_mask.append(torch.zeros([1]))
                            continue
                        user_representation = db_nodes_features[seed_set]  
                        user_representation = self.model.self_attn_db(user_representation) 
                        user_representation_list.append(user_representation)
                        db_con_mask.append(torch.ones([1])) 
                    user_representation_array = torch.stack(user_representation_list).cpu().detach().numpy()

                    graph_con_emb=con_nodes_features[concept_mask]
                    con_emb_mask=concept_mask==self.model.concept_padding
                    con_user_emb,attention = self.model.self_attn(graph_con_emb,con_emb_mask.cuda())
                    con_user_emb = con_user_emb.cpu().detach().numpy()

                    response_con_emb = con_nodes_features[response_concept_mask]
                    response_emb_mask = response_concept_mask==self.model.concept_padding
                    response_user_emb, attention = self.model.self_attn(response_con_emb.cuda(),response_emb_mask.cuda())
                    response_user_emb = response_user_emb.cpu().detach().numpy()
                    
                    con_node_sets = []
                    for concept in concept_mask:
                        node_set = []
                        for word in concept.numpy().tolist():
                            if word==0 or word not in self.model.concept_kg:
                                continue
                            node_set.append(word)
                        con_node_sets.append(node_set) 
                
                batch_state,batch_done = self.model.concept_env.reset(user_representation_array, con_user_emb,con_node_sets, db_node, con_node, response_user_emb) 
                for i in range(3):
                    batch_act_mask = self.model.concept_env.batch_action_mask(dropout=self.model.act_dropout) 
                    batch_act_idx = self.model.concept_ac.select_action(batch_state, batch_act_mask, self.model.device)
                    batch_state, batch_reward = self.model.concept_env.batch_step(batch_act_idx)
                    self.model.concept_ac.rewards.append(batch_reward)
                
                lr = self.model.lr * max(1e-4, 1.0 - float(step) / (self.opt["epoch"] / self.model.batch_size))
                for pg in self.model.ac_optimizer.param_groups:
                    pg['lr'] = self.model.lr
                
                total_rewards.append(np.sum(self.model.concept_ac.rewards))
                loss,ploss,vloss,eloss = self.model.concept_ac.update(self.model.ac_optimizer, self.model.device, self.model.ent_weight,1.0-np.array(batch_done,dtype=np.float).astype(int))
                total_losses.append(loss)
                total_plosses.append(ploss)
                total_vlosses.append(vloss)
                total_entropy.append(eloss)
                step += 1
                # if self.step%100==0:
                #     print('loss={:.5f}'.format(np.mean(loss)))
                #     print('ploss={:.5f}'.format(np.mean(ploss)))
                #     print('vloss={:.5f}'.format(np.mean(vloss)))
                #     print('entropy={:.5f}'.format(np.mean(eloss)))
                if step > 0 and step % 100 == 0:
                    avg_reward = np.mean(total_rewards) / self.model.batch_size
                    avg_loss = np.mean(total_losses)
                    avg_ploss = np.mean(total_plosses)
                    avg_vloss = np.mean(total_vlosses)
                    avg_entropy = np.mean(total_entropy)
                    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
                    print(
                            'epoch/step={:d}/{:d}'.format(i, step) +
                            ' | loss={:.5f}'.format(avg_loss) +
                            ' | ploss={:.5f}'.format(avg_ploss) +
                            ' | vloss={:.5f}'.format(avg_vloss) +
                            ' | entropy={:.5f}'.format(avg_entropy) +
                            ' | reward={:.5f}'.format(avg_reward))

        self.model.save_model()

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims
    
    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

