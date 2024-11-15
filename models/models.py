import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from models.layers import DPPLayer
from models.long_tail_enhancement import LTLayer

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype)
            return graph.edges[etype].data['score']

class BaseGraphModel(nn.Module):
    def __init__(self, args, dataloader):
        super().__init__()
        self.args = args
        self.hid_dim = args.embed_size
        self.layer_num = args.layers
        self.graph = dataloader.train_graph
        self.user_number = dataloader.user_number
        self.item_number = dataloader.item_number

        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim))
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('item').shape[0], self.hid_dim))
        self.predictor = HeteroDotProductPredictor()

        self.build_model()

        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding}

    def build_layer(self, idx):
        pass
    def lt_layer(self,idx):
        pass

    def build_model(self):
        self.layers = nn.ModuleList()
        self.lts = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            hh2 = self.lt_layer(idx)
            self.layers.append(h2h)
            self.lts.append(hh2)

    def get_embedding(self):
        h = self.node_features

        graph_user2item = dgl.edge_type_subgraph(self.graph, ['rate'])
        graph_item2user = dgl.edge_type_subgraph(self.graph, ['rated by'])

        for layer in self.layers:
            user_feat = h['user']
            item_feat = h['item']

            h_item = layer(graph_user2item, (user_feat, item_feat))
            h_user = layer(graph_item2user, (item_feat, user_feat))

            h = {'user': h_user, 'item': h_item}
        return h

    def get_long_tail_embedding(self):
        h2_user_embed = [self.user_embedding]
        h2_item_embed = [self.item_embedding]
        h2 = self.node_features

        for layer in self.lts:
            h2_item = layer(self.graph, h2, ('user', 'rate', 'item'))
            h2_user = layer(self.graph, h2, ('item', 'rated by', 'user'))
            h2 = {'user': h2_user, 'item': h2_item}
            h2_user_embed.append(h2_user)
            h2_item_embed.append(h2_item)

        h2_user_embed = self.layer_attention(h2_user_embed, self.W, self.a)
        h2_item_embed = self.layer_attention(h2_item_embed, self.W, self.a)

        return {'user': h2_user_embed, 'item': h2_item_embed}

    def forward(self, graph_pos, graph_neg):
        h = self.get_embedding()

        score_pos = self.predictor(graph_pos, h, 'rate')
        score_neg = self.predictor(graph_neg, h, 'rate')

        h_lt = self.get_long_tail_embedding()

        score_pos_lt = self.predictor(graph_pos, h_lt, 'rate')
        score_neg_lt = self.predictor(graph_neg, h_lt, 'rate')

        return score_pos, score_neg, score_pos_lt, score_neg_lt, h, h_lt

    def get_score(self, h, users):
        user_embed = h['user'][users]
        item_embed = h['item']
        scores = torch.mm(user_embed, item_embed.t())
        return scores

class DAGCL(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(DAGCL, self).__init__(args, dataloader)
        self.W = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size))
        self.a = torch.nn.Parameter(torch.randn(self.args.embed_size))

    def build_layer(self, idx):
        return DPPLayer(self.args)
    def lt_layer(self,idx):
        return LTLayer(self.args)

    def layer_attention(self, ls, W, a):
        tensor_layers = torch.stack(ls, dim=0)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(torch.matmul(weight, a), dim=0).unsqueeze(-1)
        tensor_layers = torch.sum(tensor_layers * weight, dim=0)
        return tensor_layers

    def get_embedding(self):
        user_embed = [self.user_embedding]
        item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:
            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.graph, h, ('item', 'rated by', 'user'))
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)

        user_embed = self.layer_attention(user_embed, self.W, self.a)
        item_embed = self.layer_attention(item_embed, self.W, self.a)

        return {'user': user_embed, 'item': item_embed}

    def get_long_tail_embedding(self):
        h2_user_embed = [self.user_embedding]
        h2_item_embed = [self.item_embedding]
        h2 = self.node_features

        for layer in self.lts:
            h2_item = layer(self.graph, h2, ('user', 'rate', 'item'))
            h2_user = layer(self.graph, h2, ('item', 'rated by', 'user'))
            h2 = {'user': h2_user, 'item': h2_item}
            h2_user_embed.append(h2_user)
            h2_item_embed.append(h2_item)

        h2_user_embed = self.layer_attention(h2_user_embed, self.W, self.a)
        h2_item_embed = self.layer_attention(h2_item_embed, self.W, self.a)

        return {'user': h2_user_embed, 'item': h2_item_embed}


