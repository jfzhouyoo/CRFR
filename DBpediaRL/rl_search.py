from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from math import log
from datetime import datetime
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import threading
import numpy as np
from functools import reduce
from collections import defaultdict

def batch_beam_search(kg_env, ac_model, user_embed, batch_node_set, db_nodes_features, device, topk=[5,5,1]):
    def _batch_acts_to_mask(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(ac_model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)
    
    state_pool,batch_done = kg_env.reset(user_embed, batch_node_set, db_nodes_features) # state_pool shape: [batch_size, dim]
    path_pool = kg_env._batch_path
    probs_pool = [[] for _ in range(len(user_embed))]
    user_id_reference = [i for i in range(len(user_embed))] 
    ac_model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        if hop==0:
            acts_pool = kg_env._batch_curr_actions
        else:
            acts_pool = kg_env._batch_get_actions(path_pool)
        actmask_pool = _batch_acts_to_mask(acts_pool)
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = ac_model((state_tensor, actmask_tensor))
        probs = probs + actmask_tensor.float()
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()
        
        user_embed_id = []
        new_path_pool, new_probs_pool = [], []
        for row,user_id in zip(range(topk_idxs.shape[0]), user_id_reference):
            path = path_pool[row]
            probs = probs_pool[row]
            if hop==0:
                if len(batch_node_set[row])==0:
                    new_path_pool.append([])
                    new_probs_pool.append([])
                    user_embed_id.append(user_id)
                    continue
            else:
                if len(path)==0:
                    new_path_pool.append([])
                    new_probs_pool.append([])
                    user_embed_id.append(user_id)
                    continue
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):
                    continue
                next_node_id = acts_pool[row][idx]
                new_path = path + [next_node_id]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs+[p])
                user_embed_id.append(user_id)
        user_id_reference = user_embed_id
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            kg_env._user_emebd = user_embed[user_id_reference]
            state_pool = kg_env._batch_get_state(path_pool)
    return path_pool, probs_pool, user_id_reference

def db_eval_path(user_id_reference,path_pool,probs_pool,topk=10):
    userid_path = defaultdict(list)
    for userid,path,probs in zip(user_id_reference,path_pool,probs_pool):
        if probs==[]:
            probs=[0]
        path_prob = reduce(lambda x,y: x*y,probs)
        userid_path[userid].append((path,path_prob))
    sorted_userid_path = dict()
    for userid in userid_path:
        sorted_path = sorted(userid_path[userid], key=lambda x: x[1], reverse=True)
        sorted_userid_path[userid] = sorted_path[:topk] 
    userid_nodeset = defaultdict(list)
    original_userid_nodeset = defaultdict(list) 
    for userid,paths in list(sorted_userid_path.items()):
        for path in paths:
            if len(path[0])==0:
                userid_nodeset[userid].extend(path[0])  
                original_userid_nodeset[userid].extend(path[0])
            else:
                userid_nodeset[userid].extend([path[0][-2],path[0][-1]]) 
                original_userid_nodeset[userid].extend([path[0][0]]) 
    
    return original_userid_nodeset,userid_nodeset