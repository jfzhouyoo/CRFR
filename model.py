from models.transformer import TorchGeneratorModel,_build_encoder,_build_decoder,_build_encoder_mask, _build_encoder4kg, _build_decoder4kg
from models.utils import _create_embeddings,_create_entity_embeddings
from models.graph import SelfAttentionLayer,SelfAttentionLayer_batch, DbOneHopAttention, DbTwoHopAttention, DbOneHopSelfAttention, mixSelfAttention, DbContextAttention, DbOneHopAttentionTopk, DbTwoHopAttentionTopk, DbTwoHopSelfAttention 
from models.graph import context_user_selfattention1,context_user_selfattention2 
from models.graph import conceptOneHopAttention,conceptTwoHopAttention,mixConceptSelfAttention 
from models.graph import DbRlSelfAttentionLayer 
from models.graph import conceptOneHopAttention2, dbToConceptAttention, calSimilarity,ConceptRlSelfAttentionLayer 
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
import json
from DBpediaRL.rl import  ActorCritic
from DBpediaRL.kg_env import KGEnvironment
from DBpediaRL.rl_search import batch_beam_search, db_eval_path
from conceptRL.con_rl import ConceptActorCritic
from conceptRL.concept_env import ConceptEnvironment
from conceptRL.con_rl_search import batch_concept_beam_search, eval_path
from torch import optim
import warnings
warnings.filterwarnings("ignore")

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

EDGE_TYPES = [58, 172]
def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1 
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx) 

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)

def direction_edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    # edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)
    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx) 

def concept_edge_list4GCN():
    node2index=json.load(open('key2index_3rd.json',encoding='utf-8'))
    f=open('conceptnet_edges2nd.txt',encoding='utf-8')
    edges=set()
    stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
    for line in f:
        lines=line.strip().split('\t')
        entity0=node2index[lines[1].split('/')[0]]
        entity1=node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0,entity1))
        edges.add((entity1,entity0))
    edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
    return torch.LongTensor(edge_set).cuda()

def get_conceptnet_kg():
    import pickle
    with open("./data/concept_kg.pkl","rb") as f:
        conceptnet_kg = pickle.load(f)
    return conceptnet_kg

class CrossModel(nn.Module):
    def __init__(self, opt, dictionary, is_finetune=False, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        self.batch_size = opt['batch_size'] # batch_size
        self.max_r_length = opt['max_r_length'] 

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx])) 
        self.longest_label = longest_label

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        ) 

        self.concept_embeddings=_create_entity_embeddings( 
            opt['n_concept']+1, opt['dim'], 0) 
        self.concept_padding=0

        self.kg = pkl.load( 
            open("data/subkg.pkl", "rb")
        ) 
        self.concept_kg = pkl.load(open("data/concept_kg2.pkl","rb")) 
        
        if opt.get('n_positions'):
            n_positions = opt['n_positions']
        else:
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        self.db_norm = nn.Linear(opt['dim'], opt['embedding_size']) 
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size']) 

        self.db_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.kg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])

        self.criterion = nn.CrossEntropyLoss(reduce=False)

        self.self_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        self.self_attn_db = SelfAttentionLayer(opt['dim'], opt['dim'])  
        self.db_context_attention = DbContextAttention(opt['dim'],opt['embedding_size'],opt['max_c_length']) 
        self.db_one_hop_attention = DbOneHopAttention(opt['dim'],opt['dim']) 
        self.db_one_hop_attention_topk = DbOneHopAttentionTopk(opt['dim'],opt['dim'])
        self.db_two_hop_attention = DbTwoHopAttention(opt['dim'],opt['dim']) 
        self.db_two_hop_attention_topk = DbTwoHopAttentionTopk(opt['dim'],opt['dim']) 
        self.mix_attention_db = nn.Linear(opt['dim']*3, opt['dim']) 
        self.db_one_hop_selfattention = DbOneHopSelfAttention(opt['dim'],opt['dim']) 
        self.db_two_hop_selfattention = DbTwoHopSelfAttention(opt['dim'],opt['dim']) 
        self.mix_selfattention_db = mixSelfAttention(opt['dim'],opt['dim']) 
        self.mix_user_representation_gate = nn.Linear(opt['dim'],1) 
        self.db_gru = nn.GRU(opt['dim'],opt['dim'])
        self.add_weight = nn.Parameter(torch.zeros(opt['dim'],opt['dim']))
        self.add_bias = nn.Parameter(torch.zeros(1,opt['dim']))
        self.mul_weight = nn.Parameter(torch.zeros(opt['dim'],opt['dim']))
        self.mul_bias = nn.Parameter(torch.zeros(1,opt['dim']))
        self.leakrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.context_norm = nn.Linear(opt['embedding_size']*opt['max_c_length'],opt['dim'])
        self.context_user_selfattention1 = context_user_selfattention1(opt['dim'],opt['dim'])
        self.context_user_selfattention2 = context_user_selfattention2(opt['dim'],opt['dim'])
        self.concept_one_hop_attention = conceptOneHopAttention(opt['dim'],opt['embedding_size'],opt['max_c_length'])
        self.concept_one_hop_attention2 = conceptOneHopAttention2(opt['dim'],opt['dim'])
        self.concept_two_hop_attention = conceptTwoHopAttention(opt['dim'],opt['dim'])
        self.db_to_concept_attention = dbToConceptAttention(opt['dim'], opt['dim']) 
        self.mix_selfattention_concept = mixConceptSelfAttention(opt['dim'],opt['dim'])
        self.con_add_weight = nn.Parameter(torch.zeros(opt['dim'],opt['dim']))
        self.con_add_bias = nn.Parameter(torch.zeros(1,opt['dim']))
        self.con_mul_weight = nn.Parameter(torch.zeros(opt['dim'],opt['dim']))
        self.con_mul_bias = nn.Parameter(torch.zeros(1,opt['dim']))
        self.con_leakrelu = nn.LeakyReLU()
        self.con_tanh = nn.Tanh()
        self.con_relu = nn.ReLU()
        self.mix_attention_con = nn.Linear(opt['dim']*3,opt['dim']) 
        self.con_sigmoid = nn.Sigmoid()
        self.max_c_length = opt['max_c_length']
        direction_edge,_ = direction_edge_list(self.kg, opt['n_entity'], hop=2)
        self.edge_kg = defaultdict(set) 
        for v in direction_edge:
            self.edge_kg[v[0]].add((v[2],v[1]))

        self.kg_env = KGEnvironment(self.edge_kg, 50, opt['dim'])
        self.ac = ActorCritic(opt['dim'], 4*128, 50, 3, opt['batch_size'])
        self.lr=opt['learningrate']
        self.ac_optimizer = optim.Adam(self.ac.parameters(), lr=opt['learningrate'])
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        self.ent_weight = 1e-3
        self.step = 0
        self.target_count = 0
        self.act_dropout = 0.5
        self.seed_none = 0
        self.db_rl_selfAttention = DbRlSelfAttentionLayer(opt['dim'],opt['dim'])
        self.avgPooling = nn.AdaptiveAvgPool2d((1,opt['dim']))
        self.maxPooling = nn.AdaptiveMaxPool2d((1,opt['dim']))
        self.db_rl_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.db_rl_gate_norm = nn.Linear(opt['dim'], 1) 
        self.concept_env = ConceptEnvironment(self.concept_kg,128,opt['dim'])
        self.concept_ac = ConceptActorCritic(opt['dim'],4*128,128,3,opt['batch_size'])
        self.con_rl_selfAttention = ConceptRlSelfAttentionLayer(opt['dim'],opt['dim'])
        self.user_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        self.copy_norm = nn.Linear(opt['embedding_size']*2+opt['embedding_size'], opt['embedding_size'])
        self.copy_user_emb_norm = nn.Linear(opt['embedding_size']*2, opt['embedding_size']) 
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)

        self.info_con_norm = nn.Linear(opt['dim'], opt['dim'])  
        self.info_db_norm = nn.Linear(opt['dim'], opt['dim'])   
        self.info_output_db = nn.Linear(opt['dim'], opt['n_entity']) 
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept']+1) 
        self.info_con_loss = nn.MSELoss(size_average=False,reduce=False) 
        self.info_db_loss = nn.MSELoss(size_average=False,reduce=False)  

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512) 
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4) 

        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.embedding_size=opt['embedding_size']
        self.dim=opt['dim']

        self.copy_gate = nn.Linear(opt['embedding_size'], 1)
        self.kg_path_norm = nn.Linear(opt['dim']*3, opt['embedding_size'])
        self.db_path_norm = nn.Linear(opt['dim']*3, opt['embedding_size'])
        edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2) 
        edge_list = list(set(edge_list))
        self.edge_kg = defaultdict(set) 
        for v in edge_list:
            self.edge_kg[v[0]].add((v[2],v[1]))
        print(len(edge_list), self.n_relation) 
        self.dbpedia_edge_sets=torch.LongTensor(edge_list).cuda() 
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t() 
        self.db_edge_type = self.dbpedia_edge_sets[:, 2] 

        self.dbpedia_RGCN=RGCNConv(opt['n_entity'], self.dim, self.n_relation, num_bases=opt['num_bases']) 
        self.concept_edge_sets=concept_edge_list4GCN()
        self.concept_GCN=GCNConv(self.dim, self.dim)

        w2i=json.load(open('word2index_redial.json',encoding='utf-8'))
        self.i2w={w2i[word]:word for word in w2i}

        self.mask4key=torch.Tensor(np.load('mask4key.npy')).cuda()
        self.mask4movie=torch.Tensor(np.load('mask4movie.npy')).cuda()
        self.mask4=self.mask4key+self.mask4movie
        if is_finetune:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),
                      self.concept_embeddings.parameters(),
                      self.self_attn.parameters(), self.self_attn_db.parameters(), self.user_norm.parameters(),
                      self.gate_norm.parameters(), self.output_en.parameters(),
                      self.db_rl_selfAttention.parameters(),self.db_rl_gate_norm.parameters(),self.db_rl_norm.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = False 
            
            for w in [self.add_weight,self.add_bias,self.mul_weight,self.mul_bias]:
                w.requires_grad = False

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def decode_greedy(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, bsz, maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS

            user_emb = self.db_attn_norm(attention_db) 

            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, user_emb, incr_state)
            scores = scores[:, -1:, :]
            voc_logits = F.linear(scores, self.embeddings.weight)

            sum_logits = voc_logits

            _, preds = sum_logits.max(dim=-1)

            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0) 
        seqlen = ys.size(1) 
        inputs = ys.narrow(1, 0, seqlen - 1) 
        inputs = torch.cat([self._starts(bsz), inputs], 1) 

        user_emb=self.db_attn_norm(attention_db) 

        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db, user_emb) 

        logits = F.linear(latent, self.embeddings.weight)
        sum_logits = logits
        _, preds = sum_logits.max(dim=2)
        return logits, preds

    def infomax_loss(self, con_nodes_features, db_nodes_features, con_user_emb, db_user_emb, con_label, db_label, mask):
        con_emb=self.info_con_norm(con_user_emb) 
        db_emb=self.info_db_norm(db_user_emb) 
        con_scores = F.linear(db_emb, con_nodes_features, self.info_output_con.bias)  
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias) 

        info_db_loss=torch.sum(self.info_db_loss(db_scores,db_label.cuda().float()),dim=-1)*mask.cuda() 
        info_con_loss=torch.sum(self.info_con_loss(con_scores,con_label.cuda().float()),dim=-1)*mask.cuda() 

        return torch.mean(info_db_loss), torch.mean(info_con_loss) 

    def forward(self, xs, ys, mask_ys, concept_mask, db_mask, seed_sets, labels, con_label, db_label, entity_vector, rec, test=True, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None,pre_train=False):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        if test == False:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        encoder_states = prev_enc if prev_enc is not None else self.encoder(xs) 
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type) 
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets) 

        user_representation_list = []
        db_con_mask=[]
        for i, seed_set in enumerate(seed_sets): 
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]  
            
            user_representation = self.self_attn_db(user_representation) 
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1])) 
        
        if pre_train==False:
            db_node = db_nodes_features.cpu().detach().numpy()
            user_representation_array = torch.stack(user_representation_list).cpu().detach().numpy()
            
            path_pool,probs_pool,user_id_reference = batch_beam_search(self.kg_env,self.ac,user_representation_array,seed_sets,db_node,self.device,[5,4,1])
            userid_nodeset = defaultdict(list)
            original_userid_nodeset = defaultdict(list) 
            for userid,path in zip(user_id_reference,path_pool):
                if len(path)==0:
                    userid_nodeset[userid].extend(path)  
                    original_userid_nodeset[userid].extend(path)
                else:
                    userid_nodeset[userid].append(path[-1]) 
                    original_userid_nodeset[userid].extend([path[0]]) 
            db_entity = []
            for i,seed_set in enumerate(seed_sets):
                temp_entity = seed_set + userid_nodeset[i]
                temp_entity_3 = []
                temp_entity_2 = [temp_entity_3.append(node) for node in temp_entity if node not in temp_entity_3]
                temp_entity = temp_entity_3[:50]
                while len(temp_entity)<50:
                    temp_entity.append(0)
                db_entity.append(temp_entity)
            
            db_selected_original_entity = []
            for userid in original_userid_nodeset:
                entity = original_userid_nodeset[userid]
                db_selected_original_entity.append(entity)
            
            db_path_info = defaultdict(list)
            for userid,path in zip(user_id_reference,path_pool):
                if len(path)==0:
                    db_path_info[userid].append([])
                    continue
                db_path_info[userid].append([path[0],path[1],path[2]])
            
            hop_user_representation_list = []
            db_con_mask = []
            for (uid,node_set),self_user in zip(list(userid_nodeset.items()),user_representation_list):
                if node_set == []:
                    hop_user_representation_list.append(torch.zeros(self.dim).cuda())
                    db_con_mask.append(torch.zeros([1]))
                    continue
                user_representation = db_nodes_features[list(set(node_set))]  
                user_representation = self.db_rl_selfAttention(user_representation) 
                original_user_pre = db_nodes_features[list(set(original_userid_nodeset[uid]))] 
                original_user_pre = self.db_rl_selfAttention(original_user_pre) 
                
                add_hop_user_representation = original_user_pre + user_representation
                add_hop_user_representation = self.con_relu(torch.matmul(add_hop_user_representation,self.add_weight)+self.add_bias)
                mul_hop_user_representation = torch.mul(original_user_pre,user_representation)
                mul_hop_user_representation = self.con_relu(torch.matmul(mul_hop_user_representation,self.mul_weight)+self.mul_bias)
                hop_user_representation = (add_hop_user_representation + mul_hop_user_representation).squeeze(0)
                hop_user_representation_list.append(hop_user_representation)
                db_con_mask.append(torch.ones([1])) 
            hop_db_user_emb = torch.stack(hop_user_representation_list)
        db_user_emb=torch.stack(user_representation_list) 
        db_con_mask=torch.stack(db_con_mask) 

        if pre_train==False:
            hop_user_emb=self.db_rl_norm(torch.cat([hop_db_user_emb,db_user_emb],dim=-1)) 
            hop_gate = F.sigmoid(self.db_rl_gate_norm(hop_user_emb)) 
            db_user_emb = hop_gate * db_user_emb + (1 - hop_gate) * hop_db_user_emb 
        graph_con_emb=con_nodes_features[concept_mask] 
        con_emb_mask=concept_mask==self.concept_padding 

        con_user_emb=graph_con_emb 
        con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.cuda()) 

        if pre_train==False:
            db_node = db_nodes_features.cpu().detach().numpy()
            con_node = con_nodes_features.cpu().detach().numpy()
            user_representation_array = torch.stack(user_representation_list).cpu().detach().numpy()
            con_user_emb_array = con_user_emb.cpu().detach().numpy()
            con_node_sets = []

            for concept in concept_mask:
                node_set = []
                for word in concept.cpu().numpy().tolist():
                    if word==0 or word not in self.concept_kg:
                        continue
                    node_set.append(word)
                con_node_sets.append(node_set)
            
            path_pool,probs_pool,user_id_reference = batch_concept_beam_search(self.concept_env,self.concept_ac,user_representation_array,con_user_emb_array,con_node_sets,db_node,con_node,self.device,[25,5,1])
            original_userid_nodeset,userid_nodeset,concept_path_nodeset = eval_path(user_id_reference,path_pool,probs_pool,20)
            
            concept_entity = []
            for i,concept in enumerate(con_node_sets):
                temp_entity = concept + userid_nodeset[i]
                temp_entity_3 = []
                temp_entity_2 = [temp_entity_3.append(node) for node in temp_entity if node not in temp_entity_3 and node != 0]
                temp_entity = temp_entity_3[:self.max_c_length]
                while len(temp_entity)<self.max_c_length:
                    temp_entity.append(0)
                concept_entity.append(temp_entity)
            
            concept_selected_original_word = []  
            for userid in original_userid_nodeset:
                word = list(set(original_userid_nodeset[userid]))
                while len(word)<self.max_c_length:
                    word.append(0)
                concept_selected_original_word.append(word)

        user_emb=self.user_norm(torch.cat([con_user_emb,db_user_emb],dim=-1)) 
        uc_gate = F.sigmoid(self.gate_norm(user_emb)) 
        user_emb = uc_gate * db_user_emb + (1 - uc_gate) * con_user_emb 
        entity_scores = F.linear(user_emb, db_nodes_features, self.output_en.bias) 
        mask_loss=0
        info_db_loss, info_con_loss=self.infomax_loss(con_nodes_features,db_nodes_features,con_user_emb,db_user_emb,con_label,db_label,db_con_mask) 

        rec_loss=self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.cuda()) 
        rec_loss = torch.sum(rec_loss*rec.float().cuda())  

        self.user_rep=user_emb

        #generation---------------------------------------------------------------------------------------------------
        gen_loss = None
        scores = None
        preds = None
        if pre_train==False:
            db_path_emb = []
            path_vector = []
            for userid in db_path_info:
                path_emb = []
                vector = []
                
                for path in db_path_info[userid]:
                    # print(path)
                    if len(path)==0:
                        path_emb.append(torch.zeros(self.dim*3).cuda())
                        vector.append(0)
                        continue
                    path_emb.append(db_nodes_features[path].reshape(-1))
                    vector.append(1)
                while len(vector)<20:
                    path_emb.append(torch.zeros(self.dim*3).cuda())
                    vector.append(0)
                db_path_emb.append(torch.stack(path_emb))
                path_vector.append(vector)
            
            db_path_emb = torch.stack(db_path_emb).cuda()
            path_vector = torch.tensor(path_vector)
            db_mask4gen = path_vector != 0
            db_encoding = (self.db_path_norm(db_path_emb),db_mask4gen.cuda())
            concept_mask = torch.LongTensor(concept_entity)
            entity_vector = torch.LongTensor(db_entity)
            con_nodes_features4gen=con_nodes_features#self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
            con_emb4gen = con_nodes_features4gen[concept_mask]
            con_mask4gen = concept_mask != self.concept_padding 
            kg_encoding=(self.kg_norm(con_emb4gen),con_mask4gen.cuda()) 
            if test == False:
                scores, preds = self.decode_forced(encoder_states, kg_encoding, db_encoding, None, user_emb, mask_ys) 
                gen_loss = torch.mean(self.compute_loss(scores, mask_ys))

            else:
                scores, preds = self.decode_greedy(
                    encoder_states, kg_encoding, db_encoding, None, user_emb,
                    bsz,
                    maxlen or self.longest_label
                )
                gen_loss = None

        return scores, preds, entity_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.cuda(), score_view.cuda())
        return loss

    def save_model(self):
        torch.save(self.state_dict(), 'saved_model/net_parameter1.pkl')

    def load_model(self):
        self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'))

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        up_bias = up_bias.unsqueeze(dim=1)
        output += up_bias
        return output
