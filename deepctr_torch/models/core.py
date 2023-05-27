# -*- coding:utf-8 -*-
"""
Author:
    Yuhao Xu, xyh0811@gmail.com
Reference:
    [1] Hou, Y., Hu, B., Zhang, Z., & Zhao, W.X. (2022). CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space.
        Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval.

"""
from deepctr_torch.layers.transformer import TransformerEncoder
from .basemodel import BaseModel
from ..inputs import *
from ..layers import *


class TransNet(nn.Module):
    def __init__(self, max_len, n_layers=2, n_heads=4, hidden_size=64,
                 inner_size=256, hidden_dropout_prob=0.2, attn_dropout_prob=0.2,
                 hidden_act='relu', layer_norm_eps=1e-12, initializer_range=0.02):
        super().__init__()

        self.initializer_range = initializer_range

        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fn = nn.Linear(hidden_size, 1)

        self.apply(self._init_weights)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CORE(BaseModel):
    """Instantiates the LightSANs architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param seq_max_len: 序列最大长度
    :param trm_num: transformer的层数
    :param trm_head_num: 多头注意力头数
    :param trm_hidden_size: transformer隐层尺寸
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :hidden_act: Activation function
    :param dnn_activation: Activation function to use in deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return:  A PyTorch model instance.
    """

    def __init__(
            self, dnn_feature_columns, history_feature_list, seq_max_len,
            trm_num=1, trm_head_num=2, trm_hidden_size=64,
            dnn_use_bn=False, dnn_hidden_units=(256, 128, 64), dnn_activation='relu',
            l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0.0,
            sess_dropout=0.2, item_dropout=0.2, attn_dropout_prob=0.5, hidden_dropout_prob=0.5,
            hidden_act='relu', init_std=0.0001, device='cpu', gpus=None, seed=1024, task='binary'
    ):
        super(CORE, self).__init__(
            [], dnn_feature_columns,
            l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
            init_std=init_std, seed=seed, task=task,
            device=device, gpus=gpus
        )
        self.seq_max_len = seq_max_len  # 序列最大长度
        self.trm_hidden_size = trm_hidden_size
        '''
            （1） 特征配置信息 分离 
        '''
        # 从dnn_feature_columns中过滤出 `sparse feature` 配置信息 (SparseFeat对象)
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)
        ) if dnn_feature_columns else []

        # 从dnn_feature_columns中过滤出 `VarLenSparse feature` 配置信息 (VarLenSparseFeat对象)
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)
        ) if dnn_feature_columns else []

        # 获取历史行为特征列 列表
        self.history_feature_list = history_feature_list

        # 空列表：用于放置 varlenFeat中的 用户行为列 配置信息
        self.history_feature_columns = []
        # 空列表：用于放置 varlenFeat中的 非用户行为列 配置信息
        self.sparse_varlen_feature_columns = []
        # 给历史行为序列进行重命名
        # ['item_id'] -> ['hist_item_id']
        self.history_fc_names = list(
            map(lambda x: "hist_" + x, history_feature_list)
        )
        # 分离varlenFeat特征
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        # 计算一个兴趣的embed维度
        att_emb_dim = self._compute_interest_dim()

        self.net = TransNet(
            seq_max_len, n_layers=trm_num,
            n_heads=trm_head_num, hidden_size=trm_hidden_size,
            inner_size=256, hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob, hidden_act=hidden_act,
            layer_norm_eps=1e-12, initializer_range=0.02
        )

        self.sess_dropout = nn.Dropout(sess_dropout)
        self.item_dropout = nn.Dropout(item_dropout)

        # DNN网络
        self.dnn = DNN(
            inputs_dim=self.compute_input_dim(dnn_feature_columns),
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            l2_reg=l2_reg_dnn,
            use_bn=dnn_use_bn
        )

        # 输出一个单元的线性层
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
        # 获取dense特征列表（_表示的是 sparse_embedding_list + varlen_sparse_embedding_list）
        _, dense_value_list = self.input_from_feature_columns(
            X,
            self.dnn_feature_columns,  # 特征Feat列表
            self.embedding_dict  # 这个参数就是embedding字典，在 base model 中创建， feat_name: Embedding()
        )

        item_seq = X[:, 2:2 + self.seq_max_len]

        '''序列池化部分'''
        # 候选项embedding表示
        query_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_feature_columns,
            return_feat_list=self.history_feature_list,
            to_list=True
        )  # feature_num * [B, 1, e]
        # 行为序列embedding表示
        keys_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.history_feature_columns,
            return_feat_list=self.history_fc_names,
            to_list=True
        )  # feature_num * [B, T, e]
        # 其他特征的embedding表示
        dnn_input_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_feature_columns,
            to_list=True
        )  #

        # 非用户行为的 变长序列 特征 embedding表示
        sequence_embed_dict = varlen_embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.sparse_varlen_feature_columns
        )

        # （4）给 非用户行为的 变长序列特征embedding 做池化（即把多个属性的embedding合并到一行）
        sequence_embed_list = get_varlen_pooling_list(
            sequence_embed_dict,
            X,
            self.feature_index,
            self.sparse_varlen_feature_columns,
            self.device
        )

        # 非用户行为的 变长序列特征embedding 加入 dnn部分的输入
        dnn_input_emb_list += sequence_embed_list
        for i in range(len(dnn_input_emb_list)):
            dnn_input_emb_list[i] = self.item_dropout(dnn_input_emb_list[i])
        # dnn_input_emb_list[-1] = F.normalize(dnn_input_emb_list[-1], dim=-1)
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        # query_emb = torch.cat(query_emb_list, dim=-1)  # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [B, T, E]

        # 获取表示行为序列长度的 特征名
        keys_length_feature_name = [
            feat.length_name
            for feat in self.varlen_sparse_feature_columns
            if feat.length_name is not None
        ]
        # 获取行为序列长度
        keys_length = torch.squeeze(
            maxlen_lookup(X, self.feature_index, keys_length_feature_name),
            1
        )  # [B, 1]

        # print('keys_values[-1].shape')
        # print(keys_values[-1].shape)

        '''在这加上transformer encoder操作'''
        # 取出keys_values，用来计算mask
        keys_values = [
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]
            for feat in self.history_feature_columns
        ]  # feat_num * [B, T]

        attention_mask = self.get_attention_mask(keys_values[-1], bidirectional=False)

        keys_emb = self.sess_dropout(keys_emb)
        alpha = self.net(item_seq, keys_emb)
        seq_output = torch.sum(alpha * keys_emb, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)

        # hist = torch.mean(seq_output, dim=1)

        # deep part
        deep_input_emb = torch.cat(
            (deep_input_emb.squeeze(), seq_output),
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
