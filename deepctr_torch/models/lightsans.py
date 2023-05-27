# -*- coding:utf-8 -*-
"""
Author:
    Yuhao Xu
Reference:
    [1] Fan, X., Liu, Z., Lian, J., Zhao, W.X., Xie, X., & Wen, J. (2021). Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation.
    Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval.
"""
from deepctr_torch.layers.transformer import LightTransformerEncoder
from .basemodel import BaseModel
from ..inputs import *
from ..layers import *


class LightSANs(BaseModel):
    """Instantiates the LightSANs architecture.

    """

    def __init__(
            self, dnn_feature_columns, history_feature_list, seq_max_len, use_position=True,
            trm_num=1, trm_head_num=2, trm_hidden_size=64,
            use_bn=False, dnn_hidden_units=(256, 128, 64), dnn_activation='relu',
            l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0.0,
            k_interests=10, max_seq_len=10,
            init_std=0.0001, device='cpu', gpus=None, seed=1024, task='binary'
    ):
        super(LightSANs, self).__init__(
            [], dnn_feature_columns,
            l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
            init_std=init_std, seed=seed, task=task,
            device=device, gpus=gpus
        )
        self.seq_max_len = seq_max_len
        self.trm_hidden_size = trm_hidden_size

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)
        ) if dnn_feature_columns else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)
        ) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list
        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(
            map(lambda x: "hist_" + x, history_feature_list)
        )

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        self.position_embedding = nn.Embedding(seq_max_len, trm_hidden_size)

        self.transformer = LightTransformerEncoder(
            n_layers=trm_num,
            n_heads=trm_head_num,
            k_interests=k_interests,
            hidden_size=trm_hidden_size,
            inner_size=256,
            attn_dropout_prob=dnn_dropout,
            hidden_dropout_prob=dnn_dropout,
            seq_len=max_seq_len,
            hidden_act='gelu',
            layer_norm_eps=1e-12
        )

        self.dnn = DNN(
            inputs_dim=self.compute_input_dim(dnn_feature_columns),
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            l2_reg=l2_reg_dnn,
            use_bn=use_bn
        )

        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False
        ).to(device)
        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )

        query_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_feature_columns,
            return_feat_list=self.history_feature_list,
            to_list=True
        )
        keys_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.history_feature_columns,
            return_feat_list=self.history_fc_names,
            to_list=True
        )
        dnn_input_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_feature_columns,
            to_list=True
        )

        sequence_embed_dict = varlen_embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_varlen_feature_columns
        )

        sequence_embed_list = get_varlen_pooling_list(
            sequence_embed_dict,
            X,
            self.feature_index,
            self.sparse_varlen_feature_columns,
            self.device
        )

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)  # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [B, T, E]

        keys_length_feature_name = [
            feat.length_name
            for feat in self.varlen_sparse_feature_columns
            if feat.length_name is not None
        ]
        keys_length = torch.squeeze(
            maxlen_lookup(X, self.feature_index, keys_length_feature_name),
            1
        )  # [B, 1]

        keys_values = [
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]
            for feat in self.history_feature_columns
        ]  # feat_num * [B, T]

        # attention_mask = self.get_attention_mask(keys_values[-1], bidirectional=False)

        position_ids = torch.arange(self.seq_max_len, dtype=torch.long).cuda()
        position_embedding = self.position_embedding(position_ids)
        # keys_emb += position_embedding

        trm = self.transformer(
            keys_emb, position_embedding, output_all_encoded_layers=True
        )
        seq_output = trm[-1]

        keys_masks = torch.arange(self.seq_max_len).cuda().repeat(
            len(seq_output), 1)  # [B, T]

        keys_masks = keys_masks < keys_length.view(-1, 1)  # 0, 1 mask
        keys_masks = keys_masks.unsqueeze(2)
        keys_masks = keys_masks.repeat(
            1, 1, self.trm_hidden_size
        )
        # keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]
        paddings = torch.zeros_like(seq_output)
        seq_output = torch.where(keys_masks, seq_output, paddings)
        hist = torch.mean(seq_output, dim=1)

        # deep part
        deep_input_emb = torch.cat(
            (deep_input_emb.squeeze(), hist),
            dim=-1
        )
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim

    def get_attention_mask(self, sequence, bidirectional=True):
        """
        In the pre-training stage, we generate bidirectional attention mask for multi-head attention.

        In the fine-tuning stage, we generate left-to-right uni-directional attention mask for multi-head attention.
        """
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        if not bidirectional:
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(sequence.device)
            extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


if __name__ == '__main__':
    pass
