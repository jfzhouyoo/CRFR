import numpy as np
from tqdm import tqdm
from math import exp
import os
import signal
import json
import argparse
import pickle as pkl
from dataset import dataset,CRSdataset
from model import CrossModel
import torch.nn as nn
from torch import optim
import torch
from DBpediaRL.rl_search import batch_beam_search
from collections import defaultdict
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TrainLoop_fusion_rl():
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

    def train(self):
        #self.model.load_model()
        losses=[]
        best_val=0.0 
        rl_stop=False
        with torch.no_grad():
            db_nodes_features = self.model.dbpedia_RGCN(None, self.model.db_edge_idx, self.model.db_edge_type)
        db_node = db_nodes_features.cpu().detach().numpy()

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
                        # print("shape of user_representation: ",user_representation.shape)
                        # exit()
                        user_representation = self.model.self_attn_db(user_representation) 
                        user_representation_list.append(user_representation)
                        db_con_mask.append(torch.ones([1])) 
                    user_representation_array = torch.stack(user_representation_list).cpu().detach().numpy()
                
                
                batch_state,batch_done = self.model.kg_env.reset(user_representation_array, seed_sets, db_node, movie.cpu().detach().numpy()) 
                for i in range(3):
                    batch_act_mask = self.model.kg_env.batch_action_mask(dropout=self.model.act_dropout) 
                    batch_act_idx = self.model.ac.select_action(batch_state, batch_act_mask, self.model.device)
                    batch_state, batch_reward = self.model.kg_env.batch_step(batch_act_idx)
                    self.model.ac.rewards.append(batch_reward)
                
                lr = self.model.lr * max(1e-4, 1.0 - float(step) / (self.opt["epoch"] / self.model.batch_size))
                for pg in self.model.ac_optimizer.param_groups:
                    pg['lr'] = self.model.lr
                
                total_rewards.append(np.sum(self.model.ac.rewards))
                loss,ploss,vloss,eloss = self.model.ac.update(self.model.ac_optimizer, self.model.device, self.model.ent_weight,1.0-np.array(batch_done,dtype=np.float).astype(int))
                total_losses.append(loss)
                total_plosses.append(ploss)
                total_vlosses.append(vloss)
                total_entropy.append(eloss)
                step += 1
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


    def val(self,is_test=False):
        output_dict_rl = {"rate":0.0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(val_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        # recs=[]
        with torch.no_grad():
            db_nodes_features = self.model.dbpedia_RGCN(None, self.model.db_edge_idx, self.model.db_edge_type)
        db_node = db_nodes_features.cpu().detach().numpy()
        target_count = 0
        step = 0
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec,response_concept_mask in tqdm(val_dataset_loader):
            with torch.no_grad():
                user_representation_list = []
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                for i, seed_set in enumerate(seed_sets): 
                    if seed_set == []:
                        user_representation_list.append(torch.zeros(self.model.dim).cuda())
                        # db_con_mask.append(torch.zeros([1]))
                        continue
                    user_representation = db_nodes_features[seed_set]  
                    user_representation = self.model.self_attn_db(user_representation) 
                    user_representation_list.append(user_representation)
                user_representation_array = torch.stack(user_representation_list).cpu().detach().numpy()

                path_pool,probs_pool,user_id_reference = batch_beam_search(self.model.kg_env,self.model.ac,user_representation_array,seed_sets,db_node,self.model.device)
                userid_nodeset = defaultdict(list)
                for userid,path in zip(user_id_reference,path_pool):
                    if len(path)==0:
                        userid_nodeset[userid].extend(path)  
                    else:
                        userid_nodeset[userid].extend([path[-2],path[-1]]) 

                for key, label in zip(userid_nodeset, movie.cpu().numpy()):
                    if label in userid_nodeset[key]:
                        target_count += 1
                step += 1
        output_dict_rl["rate"] = target_count / (step * self.opt['batch_size'])
        
        print(output_dict_rl)

        return output_dict_rl