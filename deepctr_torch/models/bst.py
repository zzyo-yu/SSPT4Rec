# -*- coding:utf-8 -*-
"""
Author:
    Yuhao Xu
Reference:
    [1] Chen, Q., Zhao, H., Li, W., Huang, P., & Ou, W. (2019). Behavior sequence transformer for e-commerce recommendation in Alibaba.
    Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data.
"""
import torch

from deepctr_torch.layers.transformer import TransformerEncoder
from .basemodel import BaseModel
from ..inputs import *
from ..layers import *


class BST(BaseModel):
    """Instantiates the BST architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param seq_max_len: 序列最大长度
    :param trm_num: transformer的层数
    :param trm_head: 多头注意力头数
    :param trm_hidden_size: transformer隐层尺寸
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.
    """

    def __init__(
            self, dnn_feature_columns, history_feature_list, seq_max_len, use_position=True,
            trm_num=1, trm_head_num=2, trm_hidden_size=64, trm_overlay=True,
            dnn_use_bn=False, dnn_hidden_units=(256, 128, 64), dnn_activation='relu',
            l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0.0,
            init_std=0.0001, device='cpu', gpus=None, seed=1024, task='binary'
    ):
        super(BST, self).__init__(
            [], dnn_feature_columns,
            l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
            init_std=init_std, seed=seed, task=task,
            device=device, gpus=gpus
        )
        self.seq_max_len = seq_max_len
        self.use_position = use_position

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

        att_emb_dim = self._compute_interest_dim()

        if use_position:
            self.position_embedding = nn.Embedding(seq_max_len, trm_hidden_size)

        self.transformer = TransformerEncoder(
            n_layers=trm_num,
            n_heads=trm_head_num,
            hidden_size=trm_hidden_size,
            overlay=trm_overlay,
            inner_size=256,
            attn_dropout_prob=dnn_dropout,
            hidden_dropout_prob=dnn_dropout,
            hidden_act='gelu',
            layer_norm_eps=1e-12
        )

        self.dnn = DNN(
            inputs_dim=self.compute_input_dim(dnn_feature_columns),
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            l2_reg=l2_reg_dnn,
            use_bn=dnn_use_bn
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


    # 前向计算
    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(
            X,
            self.dnn_feature_columns,
            self.embedding_dict
        )

        keys_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.history_feature_columns,
            return_feat_list=self.history_fc_names,
            to_list=True
        )  # feature_num * [B, T, e]

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

        attention_mask = self.get_attention_mask(keys_values[-1], bidirectional=False)

        if self.use_position:
            position_ids = torch.arange(self.seq_max_len, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(len(X), self.seq_max_len).cuda()
            position_embedding = self.position_embedding(position_ids)

            keys_emb += position_embedding

        trm = self.transformer(
            keys_emb, attention_mask, output_all_encoded_layers=True
        )
        seq_output = trm[-1]

        seq_output = torch.unsqueeze(torch.mean(seq_output, dim=1), dim=1)

        # deep part
        deep_input_emb = torch.cat(
            (deep_input_emb, seq_output),
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
