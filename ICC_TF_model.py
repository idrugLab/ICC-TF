from torch import nn
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F

class IccTF(nn.Module):
    def __init__(self, vocab_size, d_model, seq_length, init_node_f, num_heads_gat, 
                 node_f, num_heads, d_ff, feat_dim, pred_additional_feat_dim, 
                 pred_n_layer, out_dim, pred_act="relu"):
        super(IccTF, self).__init__()

        self.node_f = node_f
        self.feat_dim = feat_dim
        self.has_additional_feat = pred_additional_feat_dim > 0
        self.pred_additional_feat_dim = pred_additional_feat_dim

        self.word_embed_layer = nn.Embedding(vocab_size, d_model)
        self.pos_embed_layer = nn.Embedding(seq_length, d_model)
        nn.init.xavier_uniform_(self.word_embed_layer.weight.data)
        nn.init.xavier_uniform_(self.pos_embed_layer.weight.data)

        # self.GAT1 = GAT(in_dims=init_node_f, out_dims=node_f, num_heads=num_heads_gat)

        self.Transformer1 = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, batch_first=True)
        
        self.aggregator1 = nn.Sequential(
            nn.Linear(d_model+node_f, feat_dim),
            nn.ReLU()
        )

        
        self.botteneck_layer = nn.Linear(seq_length*feat_dim, feat_dim)

        pred_input_dim = self.feat_dim * 2
        self.pred_n_layer = max(1, pred_n_layer)
        if self.has_additional_feat:
            feat_hidden_dim = min(pred_additional_feat_dim, 10)
            self.pred_feat = nn.Sequential(
                nn.Linear(pred_additional_feat_dim, feat_hidden_dim),
                nn.ELU(),
                nn.LayerNorm(feat_hidden_dim),
            )
            pred_input_dim += feat_hidden_dim
            self.additional_feat_hidden_dim = feat_hidden_dim

        if pred_act == "relu":
            pred_head = [
                nn.Linear(pred_input_dim, self.feat_dim // 2),
                nn.ReLU(inplace=True),
            ]
            for i in range(self.pred_n_layer - 1):
                pred_head.extend(
                    [
                        nn.Linear(self.feat_dim // 2, self.feat_dim // 2),
                        nn.ReLU(inplace=True)
                    ]
                )
        else:
            raise ValueError("Undefined activation function")
        
        pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        self.pred_head = nn.Sequential(*pred_head)


    

    
    def forward(self, data):
        size = int(data['batch'].max() + 1)
        seq = data.token.reshape(-1, 2048)
        bert = data.bert
        seq_pos = torch.arange(2048)
        seq_pos = seq_pos.reshape(1, 2048).to(seq.device)
        seq_word_embed = self.word_embed_layer(seq)
        seq_pos_embed = self.pos_embed_layer(seq_pos)
        embed = seq_word_embed + seq_pos_embed

        embed1 = self.Transformer1(embed)   # (batch_size, seq, d_model)
        x = embed1.view(embed1.size(0), -1)
        tf_vectors = self.botteneck_layer(x)
        # print(tf_vectors.shape)
        
        bert_vectors = bert
        # print(ecfp_vectors.shape)

        all_features = torch.cat([bert_vectors, tf_vectors], dim=1)
        # print(all_features.shape)


        if self.has_additional_feat:
            feat = data.feat.reshape(-1, self.pred_additional_feat_dim)
            feat = self.pred_feat(feat)
            h = torch.cat((all_features, feat), dim=1)
        
        return h, self.pred_head(h)

    