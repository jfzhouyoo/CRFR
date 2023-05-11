from __future__ import absolute_import, division, print_function

import sys
# sys.path.append("..")
import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
from .concept_env import ConceptEnvironment
# from concept_env import ConceptEnvironment
logger = None

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ConceptActorCritic(nn.Module):
    def __init__(self, dim, state_dim, max_acts, max_hop, batch_size, gamma=0.99):
        super(ConceptActorCritic, self).__init__()
        self.dim = dim
        self.act_dim = max_acts + 1
        self.max_hop = max_hop
        self.gamma = gamma
        self.batch_size = batch_size
        self.hidden_size_1 = self.max_hop*self.dim
        self.hidden_size_2 = (self.max_hop-1) * self.dim

        # self.state_encoder = nn.Linear(self.max_acts, 1)
        
        self.l1 = nn.Linear(state_dim, self.hidden_size_1)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.actor = nn.Linear(self.hidden_size_2, self.act_dim)
        self.critic = nn.Linear(self.hidden_size_2, 1)

        self.saved_actions = []
        self.saved_states = [] 
        self.rewards = []
        self.entropy = []
    
    def forward(self, inputs):
        batch_state, batch_act_mask = inputs 
        x = self.l1(batch_state)
        x = F.dropout(F.elu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.elu(out), p=0.5)

        actor_logits = self.actor(x)
        actor_logits[1 - batch_act_mask] = -999999.0
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]
        
        state_value = self.critic(x) # shape: [batch_size, 1]

        return act_probs, state_value


    def select_action(self, batch_state, batch_act_mask, device):
        batch_state = torch.FloatTensor(batch_state).to(device)
        batch_act_mask = torch.ByteTensor(batch_act_mask).to(device)

        probs, value = self((batch_state, batch_act_mask))

        m = Categorical(probs)
        acts = m.sample()
        valid_idx = batch_act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0
        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()
    
    def update(self, optimizer, device, ent_weight, done):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0
        done = torch.from_numpy(done).cuda()
        batch_rewards = np.vstack(self.rewards).T # shape: [batch_size, step]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1,num_steps):
            batch_rewards[:, num_steps -i - 1] += self.gamma * batch_rewards[:, num_steps - i]
        
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        
        for i in range(0,num_steps):
            log_prob, value = self.saved_actions[i]
            advantage = batch_rewards[:, i] - value.squeeze(1)
            
            actor_loss += -log_prob * advantage.detach()
            critic_loss += advantage.pow(2)
            entropy_loss += -self.entropy[i]
        
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()

    
    def get_state_memory(self, batch_state):
        if len(self.saved_states)==0:
            self.saved_states = batch_state
            state = torch.cat((batch_state, batch_state,batch_state),dim=1)
        
        else:
            if len(self.saved_states.shape) == 2:
                self.saved_states = torch.cat((self.saved_states.unsqueeze(1),batch_state.unsqueeze(1)), 1)
                state = torch.cat((self.saved_states.view(self.batch_size,-1),batch_state),dim=1)
            else:
                self.saved_states = torch.cat((self.saved_states, batch_state.unsqueeze(1)), 1)
                state = torch.cat((self.saved_states.view(self.batch_size,-1),torch.zeros(self.batch_size,(self.max_hop+1-self.saved_states.shape[1])*self.dim)))
        return state
                


