import numpy as np
from tqdm import tqdm
from math import exp
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import signal
import json
import argparse
import pickle as pkl
from dataset import dataset,CRSdataset
from model import CrossModel
import torch.nn as nn
from torch import optim
import torch
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu
from DBpediaRL.train_rl import TrainLoop_fusion_rl
from conceptRL.train_con_rl import TrainLoop_fusion_concept_rl

def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()

def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    train.add_argument("-batch_size","--batch_size",type=int,default=32)
    train.add_argument("-max_count","--max_count",type=int,default=5)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
    train.add_argument("-load_dict","--load_dict",type=str,default=None)
    train.add_argument("-learningrate","--learningrate",type=float,default=1e-3)
    train.add_argument("-optimizer","--optimizer",type=str,default='adam')
    train.add_argument("-momentum","--momentum",type=float,default=0)
    train.add_argument("-is_finetune","--is_finetune",type=bool,default=False)
    train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
    train.add_argument("-epoch","--epoch",type=int,default=30)
    train.add_argument("-gpu","--gpu",type=str,default='0,1')
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
    train.add_argument("-embedding_size","--embedding_size",type=int,default=300)

    train.add_argument("-n_heads","--n_heads",type=int,default=2)
    train.add_argument("-n_layers","--n_layers",type=int,default=2)
    train.add_argument("-ffn_size","--ffn_size",type=int,default=300)

    train.add_argument("-dropout","--dropout",type=float,default=0.1)
    train.add_argument("-attention_dropout","--attention_dropout",type=float,default=0.0)
    train.add_argument("-relu_dropout","--relu_dropout",type=float,default=0.1)

    train.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
    train.add_argument("-embeddings_scale","--embeddings_scale",type=bool,default=True)

    train.add_argument("-n_entity","--n_entity",type=int,default=64368)
    train.add_argument("-n_relation","--n_relation",type=int,default=214)
    train.add_argument("-n_concept","--n_concept",type=int,default=29308)
    train.add_argument("-n_con_relation","--n_con_relation",type=int,default=48)
    train.add_argument("-dim","--dim",type=int,default=128)
    train.add_argument("-n_hop","--n_hop",type=int,default=2)
    train.add_argument("-kge_weight","--kge_weight",type=float,default=1)
    train.add_argument("-l2_weight","--l2_weight",type=float,default=2.5e-6)
    train.add_argument("-n_memory","--n_memory",type=float,default=32)
    train.add_argument("-item_update_mode","--item_update_mode",type=str,default='0,1')
    train.add_argument("-using_all_hops","--using_all_hops",type=bool,default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)

    train.add_argument("-model_path","--model_path",type=str,default="./saved_model/net_parameter1.pkl")
    train.add_argument("-pre_train","--pre_train", type=bool,default=False)
    train.add_argument("-train_reasoning","--train_reasoning",type=bool,default=False)
    train.add_argument("-train_rec","--train_rec", type=bool,default=False)

    return train


class TrainLoop_fusion_mim():
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

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune)

        if opt['load_dict'] is not None:
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
        losses=[]
        best_val_rec=0
        rec_stop=False
        for i in range(3):
            train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,response_concept_mask in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist() 
                    seed_sets.append(seed_set) 
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(),
                                                                                                                            concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=False, pre_train=True)

                joint_loss=info_db_loss#+info_con_loss

                losses.append([info_db_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('info db loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    #print('info con loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    losses=[]
                num+=1

        self.model.save_model()

    def metrics_cal_rec(self,rec_loss,scores,labels):
        batch_size = len(labels.view(-1).tolist())
        self.metrics_rec["loss"] += rec_loss
        outputs = scores.cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        for b in range(batch_size):
            if labels[b].item()==0:
                continue
            target_idx = self.movie_ids.index(labels[b].item())
            self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics_rec["count"] += 1

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(val_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        recs=[]
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)

            recs.extend(rec.cpu())
            self.metrics_cal_rec(rec_loss, rec_scores, movie)

        output_dict_rec={key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        print(output_dict_rec)

        return output_dict_rec

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
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

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

class TrainLoop_fusion_rec():
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

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune)

        if opt['load_dict'] is not None:
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

    def torch_init_model(self):
        state_dict = torch.load(self.opt["model_path"])
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
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
    
    def build_model(self,is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        #self.model.load_model()
        losses=[]
        best_val_rec=0 
        # best_val_rec = [0, 0, 0] 
        rec_stop=False
        
        print("train recommendation module")
        losses=[]

        for i in range(self.epoch):
            train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,response_concept_mask in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec, test=False)

                joint_loss=rec_loss+0.0*info_db_loss

                losses.append([rec_loss,info_db_loss]) 
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('rec loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    print('info db loss is %f'%(sum([l[1] for l in losses])/len(losses))) 
                    losses=[]
                num+=1

            output_metrics_rec = self.val()

            if best_val_rec > output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]*10:
                rec_stop=True
            else:
                best_val_rec = output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]*10
                self.model.save_model()
                print("recommendation model saved once------------------------------------------------")
            if rec_stop==True:
                break

        _=self.val(is_test=True)

    def metrics_cal_rec(self,rec_loss,scores,labels):
        batch_size = len(labels.view(-1).tolist())
        self.metrics_rec["loss"] += rec_loss
        outputs = scores.cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        for b in range(batch_size):
            if labels[b].item()==0:
                continue
            target_idx = self.movie_ids.index(labels[b].item())
            self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics_rec["count"] += 1

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(val_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        recs=[]
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec, response_concept_mask in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)

            recs.extend(rec.cpu())
            self.metrics_cal_rec(rec_loss, rec_scores, movie)

        output_dict_rec={key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        print(output_dict_rec)

        return output_dict_rec

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
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

        lr = opt['learningrate']
        kwargs = {'lr': lr} 
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

class TrainLoop_fusion_gen():
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

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune=True)

        if opt['load_dict'] is not None:
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
        metadata = getattr(state_dict, '_metadata', None)
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
        losses=[]
        best_val_gen=1000
        gen_stop=False
        for i in range(self.epoch*3):
            train_set=CRSdataset(self.train_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,response_concept_mask in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=False)

                joint_loss=gen_loss

                losses.append([gen_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('gen loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    losses=[]
                num+=1

            output_metrics_gen = self.val(True)
            if best_val_gen < output_metrics_gen["dist4"]:
                pass
            else:
                best_val_gen = output_metrics_gen["dist4"]
                self.model.save_model()
                print("generator model saved once------------------------------------------------")

        self.model.save_model()
        _=self.val(is_test=True)

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(val_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        inference_sum=[]
        golden_sum=[]
        context_sum=[]
        losses=[]
        recs=[]
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec,response_concept_mask in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                _, _, _, _, gen_loss, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=False)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)

            golden_sum.extend(self.vector2sentence(response.cpu()))
            inference_sum.extend(self.vector2sentence(preds.cpu()))
            context_sum.extend(self.vector2sentence(context.cpu()))
            recs.extend(rec.cpu())
            losses.append(torch.mean(gen_loss))
            #print(losses)
            #exit()

        self.metrics_cal_gen(losses,inference_sum,golden_sum,recs)

        output_dict_gen={}
        for key in self.metrics_gen:
            if 'bleu' in key:
                output_dict_gen[key]=self.metrics_gen[key]/self.metrics_gen['count']
            else:
                output_dict_gen[key]=self.metrics_gen[key]
        print(output_dict_gen)

        f=open('context_test.txt','w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in context_sum])
        f.close()

        f=open('output_test.txt','w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in inference_sum])
        f.close()
        return output_dict_gen

    def metrics_cal_gen(self,rec_loss,preds,responses,recs):
        def bleu_cal(sen1, tar1):
            bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
            bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
            bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
            return bleu1, bleu2, bleu3, bleu4

        def distinct_metrics(outs):
            # outputs is a list which contains several sentences, each sentence contains several words
            unigram_count = 0
            bigram_count = 0
            trigram_count=0
            quagram_count=0
            unigram_set = set()
            bigram_set = set()
            trigram_set=set()
            quagram_set=set()
            for sen in outs:
                for word in sen:
                    unigram_count += 1
                    unigram_set.add(word)
                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + ' ' + str(sen[start + 1])
                    bigram_count += 1
                    bigram_set.add(bg)
                for start in range(len(sen)-2):
                    trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                    trigram_count+=1
                    trigram_set.add(trg)
                for start in range(len(sen)-3):
                    quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                    quagram_count+=1
                    quagram_set.add(quag)
            dis1 = len(unigram_set) / len(outs)#unigram_count
            dis2 = len(bigram_set) / len(outs)#bigram_count
            dis3 = len(trigram_set)/len(outs)#trigram_count
            dis4 = len(quagram_set)/len(outs)#quagram_count
            return dis1, dis2, dis3, dis4

        predict_s=preds
        golden_s=responses
        generated=[]

        for out, tar, rec in zip(predict_s, golden_s, recs):
            bleu1, bleu2, bleu3, bleu4=bleu_cal(out, tar)
            generated.append(out)
            self.metrics_gen['bleu1']+=bleu1
            self.metrics_gen['bleu2']+=bleu2
            self.metrics_gen['bleu3']+=bleu3
            self.metrics_gen['bleu4']+=bleu4
            self.metrics_gen['count']+=1

        dis1, dis2, dis3, dis4=distinct_metrics(generated)
        self.metrics_gen['dist1']=dis1
        self.metrics_gen['dist2']=dis2
        self.metrics_gen['dist3']=dis3
        self.metrics_gen['dist4']=dis4

    def vector2sentence(self,batch_sen):
        sentences=[]
        for sen in batch_sen.numpy().tolist():
            sentence=[]
            for word in sen:
                if word>3:
                    sentence.append(self.index2word[word])
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences

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

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

if __name__ == '__main__':
    args=setup_args().parse_args()
    print(vars(args))
    if args.is_finetune==False:

        if args.pre_train==True:
            print("pre_train...........")
            mim = TrainLoop_fusion_mim(vars(args),is_finetune=False)
            mim.train()
            print("pre_train Done...........")
        
        if args.train_reasoning==True:
            # epoch = 20, lr = 1e-4
            print("train DBpedia reasoning...........")
            loop=TrainLoop_fusion_rl(vars(args),is_finetune=False)
            loop.model.load_model()
            loop.train()
            print("train DBpedia reasoning Done...........")
            
            print("train ConceptNet reasoning...........")
            # print(vars(args))
            loop = TrainLoop_fusion_concept_rl(vars(args),is_finetune=False)
            # loop.model.load_model()
            loop.torch_init_model()
            loop.train()
            print("train ConceptNet reasoning Done...........")
            print("train reasoning Done...........")
        
        if args.train_rec==True:
            # lr = 4e-4
            print("train recommender module...........")
            loop=TrainLoop_fusion_rec(vars(args),is_finetune=False)
            # loop.model.load_model()
            loop.torch_init_model()
            loop.train()
            print("train recommender module Done...........")
    else:
        print("train dialog module...........")
        loop=TrainLoop_fusion_gen(vars(args),is_finetune=True)
        #loop.train()
        # loop.model.load_model()
        loop.torch_init_model()
        # met = loop.val(True)
        loop.train()
        print("train dialog module Done...........")
    
    if args.train_rec==True or args.is_finetune==True:
        met=loop.val(True)
    #print(met)
