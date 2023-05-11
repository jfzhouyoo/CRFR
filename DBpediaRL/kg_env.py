import os
import sys
from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime
import numpy as np

class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len  
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 3 * embed_size
        elif history_len == 2:
            self.dim = 4 * embed_size
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        else:
            raise Exception('mode should be one of {full, current}')


class KGEnvironment(object):
    def __init__(self, db_kg, max_acts, dim, max_hop=3,state_history=2):

        self.db_kg = db_kg
        self.max_acts = max_acts 
        self.act_dim = max_acts + 1
        self.max_hop = max_hop
        self.db_nodes_features = None 
        self.dim = dim
        # self.user_representation = None
        self.state_gen = KGState(self.dim, history_len=state_history)
        self.state_dim = self.state_gen.dim

        self._batch_node_set = None
        self._batch_target_set = None
        self._batch_curr_actions = None
        self._batch_curr_state_emb = None
        self._user_emebd = None
        self._batch_curr_reward = None
        self._batch_path = None

        self._done = False

    def _batch_get_actions(self, batch_path):
        return [self._get_actions(user_embed, path) for user_embed,path in zip(self._user_emebd, batch_path)]

    def _get_actions(self, user_embed, path):
        if len(path)==0:
            return path

        current_node_id = path[-1]
        actions = [current_node_id]
        
        visit_node = path
        candidate_acts = [neighbor[1] for neighbor in self.db_kg[current_node_id] if neighbor[1] not in visit_node]

        if len(candidate_acts)==0:
            return actions
        
        if len(candidate_acts)<=self.max_acts:
            actions.extend(candidate_acts)
            return actions[-self.max_acts:]
        
        scores = []
        for node in candidate_acts:
            score = np.matmul(user_embed, self.db_nodes_features[node])
            scores.append(score)
        candidate_idxs = np.argsort(scores)[-self.max_acts:]  # choose actions with larger scores
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs])
        actions.extend(candidate_acts)
        return actions
    
    def _batch_get_state(self, batch_path):
        bathc_state = [self._get_state(user_embed, path) for user_embed, path in zip(self._user_emebd, batch_path)]
        return np.vstack(bathc_state)

    def _get_state(self, user_embed, path):
        user_embed = user_embed
        zero_embed = np.zeros(self.dim)

        if len(path)==0: 
            state = self.state_gen(user_embed,zero_embed,zero_embed,zero_embed)
            return state
        
        node_id_0 = path[0]
        node_embed_0 = self.db_nodes_features[node_id_0]
        if len(path)==1:
            state = self.state_gen(user_embed, node_embed_0, zero_embed, zero_embed)
            return state
        
        node_id_1 = path[1]
        node_embed_1 = self.db_nodes_features[node_id_1]
        if len(path)==2:
            state = self.state_gen(user_embed, node_embed_0, node_embed_1, zero_embed)
            return state
        
        node_id_2 = path[2]
        node_embed_2 = self.db_nodes_features[node_id_2]
        if len(path)==3:
            state = self.state_gen(user_embed, node_embed_0, node_embed_1, node_embed_2)
            return state


    def _batch_get_reward(self, batch_path, batch_target_set=None):
        batch_reward = [self._get_reward(path,user_embed,done,target) for path,user_embed,done,target in zip(batch_path, self._user_emebd,self._batch_done,self._batch_target_set)]
        return np.array(batch_reward)

    def _get_reward(self, path, user_embed, done, target=None):
        if done:
            return 0.0

        target_score = 0.0
        act_idx = path[-1]
        if target == 0:
            return 0.0
        
        consim = np.dot(self.db_nodes_features[target], self.db_nodes_features[act_idx])/(np.linalg.norm(self.db_nodes_features[target])*(np.linalg.norm(self.db_nodes_features[act_idx])))

        if consim >= 0.8:
            target_score = 2.0
        elif consim>=0.5 and consim<0.8:
            target_score = 0.5 + consim
        else:
            target_score = min(consim, 0.5)
        return target_score
        
    
    def _is_done(self):

        return self._done or len(self._batch_path[0]) >= self.max_hop


    def reset(self, batch_state, batch_node_set, db_node_feature,batch_target_set=[]):
        
        self._user_emebd = batch_state
        self._batch_curr_actions = [node_set[-self.max_acts:] for node_set in batch_node_set]
        self.db_nodes_features = db_node_feature
        self._batch_path = [[] for i in range(len(batch_state))]
        self._batch_done = [False for i in range(len(batch_state))] 
        for i,node_set in enumerate(batch_node_set):
            if len(node_set)==0:
                self._batch_done[i] = True
        self._batch_curr_state_emb = self._batch_get_state(self._batch_path)

        self._batch_node_set = batch_node_set
        self._batch_target_set = batch_target_set
        return self._batch_curr_state_emb, self._batch_done
        

    def batch_step(self, batch_act_idx):
        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            if self._batch_done[i]:
                continue
            self._batch_path[i].append(self._batch_curr_actions[i][act_idx])
        
        self._batch_curr_state_emb = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)
        # self._done = self._is_done()
        return self._batch_curr_state_emb, self._batch_curr_reward

    def batch_action_mask(self, dropout=0.0):
        """Return action masks of size [bs, max_acts]."""
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)
    
    def print_path(self):
        for path in self._batch_path:
            msg = 'Path: {}({})'.format(path[0])
            for node in path[1:]:
                msg += ' =={}=> {}({})'.format(node)
            print(msg)