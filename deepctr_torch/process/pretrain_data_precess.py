import json
import pickle
import random

import numpy as np
import pandas as pd
import six
import torch
import math
from tqdm import tqdm
import os

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names

# with open("deepctr_torch/process/param.json", 'r') as load_f:
#     param = json.load(load_f)
#     seed = param['seed']
#     random.seed(seed)
#     print('使用随机数种子：' + str(seed))


# 负采样
def neg_sample(item_set, n_items):  # [ , ]
    item = random.randint(1, n_items - 1)
    while item in item_set:
        item = random.randint(1, n_items - 1)
    return item


def padding_zero_at_left(sequence, max_len):
    # 根据最大长度在序列左侧添0
    pad_len = max_len - len(sequence)
    sequence = [0] * pad_len + sequence
    return sequence


# 对序列进行padding操作，用0填充至相同长度
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# 生成模型数据
def create_sequence_amazon_dataset(file, maxlen=20, behavior_embed=32, masked=True, pretrain=True):
    """读取amazon_electronic数据集，手动生成反例
    生成的数据包含用户id，用户行为序列，序列长度，
    x,   训练数据，包含用户特征，item特征序列，候选item特征
    y,   label
    feature_columns,    特征列信息
    behavior_feature_list,      行为特征列
    item_id_max,                最大item_id
    pretrain_fea_col            预训练特征列信息

    :param file: 文件路径
    :param trans_score:正例过滤的分数阈值
    :param embed_dim: embedding维度
    :param maxlen: 历史行为序列的最大长度
    :param masked: 是否使用mask，决定了embed词典的长度
    :param test_neg_num: 测试集反例个数
    :return:
    """
    print('------开始处理数据------')
    # 依次读取存到pkl文件的数据
    with open(file, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)
    reviews_df.columns = ['user_id', 'item_id', 'time']

    if masked:
        item_embed = reviews_df['item_id'].max() + 2
    else:
        item_embed = reviews_df['item_id'].max() + 1

    # # 指定读入数据的列名
    # reviews_df = reviews_df

    # 获取最大的item_id
    item_id_max = reviews_df['item_id'].max()
    # 获取最大的user_id
    user_id_max = reviews_df['user_id'].max()

    print('------生成用户行为序列数据------')
    uid = []  # uid
    hist_iid = []  # 历史item id
    seq_length = []  # 序列长度
    target = []  # 标签
    item_id = []  # 目标物品
    for user_id, df in tqdm(reviews_df[['user_id', 'item_id']].groupby('user_id')):
        # df是分组后的所有行构成的DataFrame
        pos_list = df['item_id'].tolist()

        # 生成反例的方法
        def get_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
            return neg

        # 生成和正例个数相同的反例
        neg_list = [get_neg() for i in range(len(pos_list))]

        pos_temp_hist = []
        # 从第二个正例（下标为1）开始生成历史记录（第一个正例没有历史记录）
        for i in range(1, len(pos_list)):
            # 把上一条浏览记录作为历史存入hist，格式为[item_id]
            pos_temp_hist.append(pos_list[i - 1])
            # 历史记录
            pos_hist_i = pos_temp_hist.copy()
            # 添加正例
            uid.append(user_id)
            hist_iid.append(pos_hist_i)
            seq_length.append(len(pos_hist_i) if len(pos_hist_i) < 20 else 20)
            item_id.append(pos_list[i])
            target.append(1)
            # 添加负例
            uid.append(user_id)
            hist_iid.append(pos_hist_i)
            seq_length.append(len(pos_hist_i) if len(pos_hist_i) < 20 else 20)
            item_id.append(neg_list[i])
            target.append(0)

    print('------padding处理------')
    # 行为序列 padding
    hist_iid = pad_sequences(hist_iid, maxlen=maxlen, padding='post')

    uid = np.array(uid)  # uid
    hist_iid = np.array(hist_iid)  # 历史item id
    seq_length = np.array(seq_length)  # 序列长度
    target = np.array(target)  # 标签
    item_id = np.array(item_id)  # 目标物品

    print('------shuffle处理------')
    # 打乱数据
    shuffle_ix = np.random.permutation(np.arange(len(uid)))
    uid = uid[shuffle_ix]
    hist_iid = hist_iid[shuffle_ix]
    seq_length = seq_length[shuffle_ix]
    target = target[shuffle_ix]
    item_id = item_id[shuffle_ix]

    print('------生成deepCTR训练数据集------')
    # 数据字典
    feature_dict = {
        'user': uid,  # uid
        'item_id': item_id,  # 候选商品id
        'hist_item_id': hist_iid,  # 历史行为序列
        'seq_length': seq_length  # 序列长度
    }

    # 特征配置信息
    feature_columns = [
        SparseFeat('user', user_id_max + 1, embedding_dim=10),
        # 暂时先这么做，如果是'bst'，正常embedding，如果是'bst_pre'，添加一个mask位
        SparseFeat('item_id', item_embed, embedding_dim=behavior_embed)
    ]

    feature_columns += [
        VarLenSparseFeat(
            SparseFeat(
                'hist_item_id', vocabulary_size=item_embed,
                embedding_dim=behavior_embed,
                embedding_name='item_id'
            ),
            maxlen=maxlen, length_name='seq_length'
        )
    ]

    # 用户行为特征名列表
    behavior_feature_list = ['item_id']

    # 生成 x
    x = {
        name: feature_dict[name]
        for name in get_feature_names(feature_columns)
    }

    y = target

    if pretrain is True:
        pretrain_fea_col = [
            VarLenSparseFeat(
                SparseFeat(
                    'hist_item_id', vocabulary_size=item_embed,
                    embedding_dim=behavior_embed,
                    embedding_name='item_id'
                ),
                maxlen=maxlen, length_name='seq_length'
            )
        ]
        return x, y, feature_columns, behavior_feature_list, item_id_max, pretrain_fea_col
    return x, y, feature_columns, behavior_feature_list, item_id_max


def create_sequence_amazon_books_dataset(file, maxlen=20, behavior_embed=32, masked=True, pretrain=True):
    """读取amazon_books数据集，手动生成反例
    生成的数据包含用户id，用户行为序列，序列长度，
    x,   训练数据，包含用户特征，item特征序列，候选item特征
    y,   label
    feature_columns,    特征列信息
    behavior_feature_list,      行为特征列
    item_id_max,                最大item_id
    pretrain_fea_col            预训练特征列信息

    :param file: 文件路径
    :param trans_score:正例过滤的分数阈值
    :param embed_dim: embedding维度
    :param maxlen: 历史行为序列的最大长度
    :param masked: 是否使用mask，决定了embed词典的长度
    :param test_neg_num: 测试集反例个数
    :return:
    """
    print('------开始处理数据------')
    # 依次读取存到pkl文件的数据
    rating_df = pd.read_pickle(file)

    rating_df.columns = ['user_id', 'item_id', 'time', 'category']
    
    print(rating_df['user_id'].max())
  
    # 获取最大的item_id
    item_id_max = rating_df['item_id'].max()
    # 获取最大的user_id
    user_id_max = rating_df['user_id'].max()

    if masked:
        item_embed = item_id_max + 2
        cate_embed = item_id_max + 2
    else:
        item_embed = item_id_max + 1
        cate_embed = item_id_max + 1
    
    print('------生成用户行为序列数据------')
    uid = []  # uid
    hist_iid = []  # 历史item id
    # hist_cid = []  # 历史cate id
    seq_length = []  # 序列长度
    target = []  # 标签
    item_id = []  # 目标物品id
    # cate_id = []  # 目标物品类别id
    for user_id, df in tqdm(rating_df[['user_id', 'item_id', 'category']].groupby('user_id')):
        # df是分组后的所有行构成的DataFrame
        pos_ilist = df['item_id'].tolist()
        # pos_clist = df['category'].tolist()

        # 生成反例的方法
        def get_neg():
            neg = pos_ilist[0]
            while neg in pos_ilist:
                neg = random.randint(1, item_id_max)
            return neg

        # 生成和正例个数相同的反例
        neg_list = [get_neg() for i in range(len(pos_ilist))]

        pos_temp_ihist = []
        pos_temp_chist = []
        # 从第二个正例（下标为1）开始生成历史记录（第一个正例没有历史记录）
        for i in range(1, len(pos_ilist)):
            # 把上一条浏览记录作为历史存入hist，格式为[item_id]
            pos_temp_ihist.append(pos_ilist[i - 1])
            # pos_temp_chist.append(pos_clist[i - 1])
            if i > 20:  # 历史行为长度大于20时，做截断
                pos_temp_ihist = pos_temp_ihist[-20:]
                # pos_temp_chist = pos_temp_chist[-20:]
            if i > 25:
                break
            # 历史记录
            pos_hist_i = pos_temp_ihist.copy()
            # pos_hist_c = pos_temp_chist.copy()
            # 添加通用项
            uid.extend([user_id] * 2)
            hist_iid.extend([pos_hist_i] * 2)
            # hist_cid.extend([pos_hist_c] * 2)
            seq_length.extend([len(pos_hist_i) if len(pos_hist_i) < 20 else 20] * 2)
            # 添加正例
            item_id.append(pos_ilist[i])
            # cate_id.append(pos_clist[i])
            target.append(1)
            # 添加负例
            item_id.append(neg_list[i])
            # cate_id.append(item_to_cate[neg_list[i]])
            target.append(0)

    print('------padding处理------')
    # 行为序列 padding
    hist_iid = pad_sequences(hist_iid, maxlen=maxlen, padding='post')
    # hist_cid = pad_sequences(hist_cid, maxlen=maxlen, padding='post')

    uid = np.array(uid)  # uid
    hist_iid = np.array(hist_iid)  # 历史item id
    # hist_cid = np.array(hist_cid)
    seq_length = np.array(seq_length)  # 序列长度
    target = np.array(target)  # 标签
    item_id = np.array(item_id)  # 目标物品
    # cate_id = np.array(cate_id)
    
#     print(uid.dtype)
#     print(hist_iid.dtype)
#     print(hist_cid.dtype)
#     print(seq_length.dtype)
#     print(target.dtype)
#     print(item_id.dtype)
#     print(cate_id.dtype)
    print('------shuffle处理------')
    # 打乱数据
    shuffle_ix = np.random.permutation(np.arange(len(uid)))
    uid = uid[shuffle_ix]
    hist_iid = hist_iid[shuffle_ix]
    seq_length = seq_length[shuffle_ix]
    target = target[shuffle_ix]
    item_id = item_id[shuffle_ix]

    print('------生成deepCTR训练数据集------')
    # 数据字典
    feature_dict = {
        'user': uid,  # uid
        'item_id': item_id,  # 候选商品id
        # 'cate_id': cate_id,
        'hist_item_id': hist_iid,  # 历史行为序列
        # 'hist_cate_id': hist_cid,
        'seq_length': seq_length  # 序列长度
    }

    # 特征配置信息
    feature_columns = [
        SparseFeat('user', user_id_max + 1, embedding_dim=10),
        # 暂时先这么做，如果是'bst'，正常embedding，如果是'bst_pre'，添加一个mask位
        SparseFeat('item_id', item_embed, embedding_dim=behavior_embed),
        # SparseFeat('cate_id', cate_embed, embedding_dim=behavior_embed),
    ]

    feature_columns += [
        VarLenSparseFeat(
            SparseFeat(
                'hist_item_id', vocabulary_size=item_embed,
                embedding_dim=behavior_embed,
                embedding_name='item_id'
            ),
            maxlen=maxlen, length_name='seq_length'
        )
    ]

    # 用户行为特征名列表
    # behavior_feature_list = ['item_id', 'cate_id']
    behavior_feature_list = ['item_id']

    # 生成 x
    x = {
        name: feature_dict[name]
        for name in get_feature_names(feature_columns)
    }

    y = target

    if pretrain is True:
        pretrain_fea_col = [
            VarLenSparseFeat(
                SparseFeat(
                    'hist_item_id', vocabulary_size=item_embed,
                    embedding_dim=behavior_embed,
                    embedding_name='item_id'
                ),
                maxlen=maxlen, length_name='seq_length'
            )
        ]
        return x, y, feature_columns, behavior_feature_list, item_id_max, pretrain_fea_col
    return x, y, feature_columns, behavior_feature_list, item_id_max


def create_sequence_amazon_movies_dataset(file, maxlen=20, behavior_embed=16, masked=True, pretrain=True):
    """读取amazon_books数据集，手动生成反例
    :return:
    """
    print('------开始处理数据------')
    # 依次读取存到pkl文件的数据
    rating_df = pd.read_pickle(file)

    rating_df.columns = ['user_id', 'item_id', 'time', 'category']
    
    # print(rating_df['user_id'].max())
  
    # 获取最大的item_id
    item_id_max = rating_df['item_id'].max()
    # 获取最大的user_id
    user_id_max = rating_df['user_id'].max()
    # category_max = rating_df['category'].max()

    if masked:
        item_embed = item_id_max + 2
    else:
        item_embed = item_id_max + 1
    
    print('------生成用户行为序列数据------')
    uid = []  # uid
    hist_iid = []  # 历史item id
    seq_length = []  # 序列长度
    target = []  # 标签
    item_id = []  # 目标物品id
    categories = []
    for user_id, df in tqdm(rating_df[['user_id', 'item_id', 'category']].groupby('user_id')):
        # df是分组后的所有行构成的DataFrame
        pos_ilist = df['item_id'].tolist()
        # categories_list = df['category'].tolist()

        # 生成反例的方法
        def get_neg():
            neg = pos_ilist[0]
            while neg in pos_ilist:
                neg = random.randint(1, item_id_max)
            return neg

        # 生成和正例个数相同的反例
        neg_list = [get_neg() for i in range(len(pos_ilist))]

        pos_temp_ihist = []
        pos_temp_chist = []
        # 从第二个正例（下标为1）开始生成历史记录（第一个正例没有历史记录）
        for i in range(1, len(pos_ilist)):
            # 把上一条浏览记录作为历史存入hist，格式为[item_id]
            pos_temp_ihist.append(pos_ilist[i - 1])
            if i > 20:  # 历史行为长度大于20时，做截断
                pos_temp_ihist = pos_temp_ihist[-20:]
#             if i > 25:
#                  break
            # 历史记录
            pos_hist_i = pos_temp_ihist.copy()
            # 添加通用项
            uid.extend([user_id] * 2)
            hist_iid.extend([pos_hist_i] * 2)
            seq_length.extend([len(pos_hist_i) if len(pos_hist_i) < 20 else 20] * 2)
            # categories.extend([categories_list[i]] * 2)
            # 添加正例
            item_id.append(pos_ilist[i])
            target.append(1)
            # 添加负例
            item_id.append(neg_list[i])
            target.append(0)

    print('------padding处理------')
    # 行为序列 padding
    hist_iid = pad_sequences(hist_iid, maxlen=maxlen, padding='post')

    uid = np.array(uid)  # uid
    hist_iid = np.array(hist_iid)  # 历史item id
    seq_length = np.array(seq_length)  # 序列长度
    target = np.array(target)  # 标签
    item_id = np.array(item_id)  # 目标物品
    # categories = np.array(categories)
    
    print('------shuffle处理------')
    # 打乱数据
    shuffle_ix = np.random.permutation(np.arange(len(uid)))
    uid = uid[shuffle_ix]
    hist_iid = hist_iid[shuffle_ix]
    seq_length = seq_length[shuffle_ix]
    target = target[shuffle_ix]
    item_id = item_id[shuffle_ix]
    # categories = categories[shuffle_ix]

    print('------生成deepCTR训练数据集------')
    # 数据字典
    feature_dict = {
        'user': uid,  # uid
        'item_id': item_id,  # 候选商品id
        'hist_item_id': hist_iid,  # 历史行为序列
        'seq_length': seq_length,  # 序列长度
        'category': categories
    }

    # 特征配置信息
    feature_columns = [
        SparseFeat('user', user_id_max + 1, embedding_dim=10),
        # 暂时先这么做，如果是'bst'，正常embedding，如果是'bst_pre'，添加一个mask位
        SparseFeat('item_id', item_embed, embedding_dim=behavior_embed),
#         SparseFeat('category', category_max + 1, embedding_dim=10),
    ]

    feature_columns += [
        VarLenSparseFeat(
            SparseFeat(
                'hist_item_id', vocabulary_size=item_embed,
                embedding_dim=behavior_embed,
                embedding_name='item_id'
            ),
            maxlen=maxlen, length_name='seq_length'
        )
    ]

    # 用户行为特征名列表
    # behavior_feature_list = ['item_id', 'cate_id']
    behavior_feature_list = ['item_id']

    # 生成 x
    x = {
        name: feature_dict[name]
        for name in get_feature_names(feature_columns)
    }

    y = target

    if pretrain is True:
        pretrain_fea_col = [
            VarLenSparseFeat(
                SparseFeat(
                    'hist_item_id', vocabulary_size=item_embed,
                    embedding_dim=behavior_embed,
                    embedding_name='item_id'
                ),
                maxlen=maxlen, length_name='seq_length'
            )
        ]
        return x, y, feature_columns, behavior_feature_list, item_id_max, pretrain_fea_col
    return x, y, feature_columns, behavior_feature_list, item_id_max


def create_sequence_epinions_dataset(file, maxlen=20, behavior_embed=16, masked=True, pretrain=True):
    """读取amazon_books数据集，手动生成反例
    :return:
    """
    print('------开始处理数据------')
    # 依次读取存到pkl文件的数据
    rating_df = pd.read_pickle(file)

    rating_df.columns = ['item_id', 'user_id', 'time']
    
    # print(rating_df['user_id'].max())
  
    # 获取最大的item_id
    item_id_max = rating_df['item_id'].max()
    # 获取最大的user_id
    user_id_max = rating_df['user_id'].max()
    # category_max = rating_df['category'].max()

    if masked:
        item_embed = item_id_max + 2
    else:
        item_embed = item_id_max + 1
    
    print('------生成用户行为序列数据------')
    uid = []  # uid
    hist_iid = []  # 历史item id
    seq_length = []  # 序列长度
    target = []  # 标签
    item_id = []  # 目标物品id
    categories = []
    for user_id, df in tqdm(rating_df[['user_id', 'item_id']].groupby('user_id')):
        # df是分组后的所有行构成的DataFrame
        pos_ilist = df['item_id'].tolist()
        if len(pos_ilist) > 20:
            pos_ilist = pos_ilist[:50]
        # categories_list = df['category'].tolist()

        # 生成反例的方法
        def get_neg():
            neg = pos_ilist[0]
            while neg in pos_ilist:
                neg = random.randint(1, item_id_max)
            return neg

        # 生成和正例个数相同的反例
        neg_list = [get_neg() for i in range(len(pos_ilist))]

        pos_temp_ihist = []
        pos_temp_chist = []
        # 从第二个正例（下标为1）开始生成历史记录（第一个正例没有历史记录）
        for i in range(1, len(pos_ilist)):
            # 把上一条浏览记录作为历史存入hist，格式为[item_id]
            pos_temp_ihist.append(pos_ilist[i - 1])
            if i > 20:  # 历史行为长度大于20时，做截断
                pos_temp_ihist = pos_temp_ihist[-20:]
#             if i > 20:
#                  break
            # 历史记录
            pos_hist_i = pos_temp_ihist.copy()
            # 添加通用项
            uid.extend([user_id] * 2)
            hist_iid.extend([pos_hist_i] * 2)
            seq_length.extend([len(pos_hist_i) if len(pos_hist_i) < 20 else 20] * 2)
            # categories.extend([categories_list[i]] * 2)
            # 添加正例
            item_id.append(pos_ilist[i])
            target.append(1)
            # 添加负例
            item_id.append(neg_list[i])
            target.append(0)

    print('------padding处理------')
    # 行为序列 padding
    hist_iid = pad_sequences(hist_iid, maxlen=maxlen, padding='post')

    uid = np.array(uid)  # uid
    hist_iid = np.array(hist_iid)  # 历史item id
    seq_length = np.array(seq_length)  # 序列长度
    target = np.array(target)  # 标签
    item_id = np.array(item_id)  # 目标物品
    # categories = np.array(categories)
    
    print('------shuffle处理------')
    # 打乱数据
    shuffle_ix = np.random.permutation(np.arange(len(uid)))
    uid = uid[shuffle_ix]
    hist_iid = hist_iid[shuffle_ix]
    seq_length = seq_length[shuffle_ix]
    target = target[shuffle_ix]
    item_id = item_id[shuffle_ix]
    # categories = categories[shuffle_ix]

    print('------生成deepCTR训练数据集------')
    # 数据字典
    feature_dict = {
        'user': uid,  # uid
        'item_id': item_id,  # 候选商品id
        'hist_item_id': hist_iid,  # 历史行为序列
        'seq_length': seq_length,  # 序列长度
        'category': categories
    }

    # 特征配置信息
    feature_columns = [
        SparseFeat('user', user_id_max + 1, embedding_dim=10),
        # 暂时先这么做，如果是'bst'，正常embedding，如果是'bst_pre'，添加一个mask位
        SparseFeat('item_id', item_embed, embedding_dim=behavior_embed),
#         SparseFeat('category', category_max + 1, embedding_dim=10),
    ]

    feature_columns += [
        VarLenSparseFeat(
            SparseFeat(
                'hist_item_id', vocabulary_size=item_embed,
                embedding_dim=behavior_embed,
                embedding_name='item_id'
            ),
            maxlen=maxlen, length_name='seq_length'
        )
    ]

    # 用户行为特征名列表
    # behavior_feature_list = ['item_id', 'cate_id']
    behavior_feature_list = ['item_id']

    # 生成 x
    x = {
        name: feature_dict[name]
        for name in get_feature_names(feature_columns)
    }

    y = target

    if pretrain is True:
        pretrain_fea_col = [
            VarLenSparseFeat(
                SparseFeat(
                    'hist_item_id', vocabulary_size=item_embed,
                    embedding_dim=behavior_embed,
                    embedding_name='item_id'
                ),
                maxlen=maxlen, length_name='seq_length'
            )
        ]
        return x, y, feature_columns, behavior_feature_list, item_id_max, pretrain_fea_col
    return x, y, feature_columns, behavior_feature_list, item_id_max


def create_sequence_bst_dataset(file, dataset_name, maxlen=20, behavior_embed=32, masked=True, pretrain=True):
    print('------开始处理数据------')
    # 依次读取存到pkl文件的数据
    rating_df = pd.read_pickle(file)
    if dataset_name in ['amazon_cds', 'amazon_beauty']:
        rating_df.columns = ['user_id', 'item_id', 'time', 'category']
    elif dataset_name == 'epinions':
        rating_df.columns = ['item_id','user_id', 'time']
    elif dataset_name == 'amazon_electric':
        with open(file, 'rb') as f:
            rating_df = pickle.load(f)
            cate_list = pickle.load(f)
            user_count, item_count, cate_count, example_count = pickle.load(f)
        rating_df.columns = ['user_id', 'item_id', 'time']
    # 获取最大的item_id
    item_id_max = rating_df['item_id'].max()
    # 获取最大的user_id
    user_id_max = rating_df['user_id'].max()

    if masked:
        item_embed = item_id_max + 2
    else:
        item_embed = item_id_max + 1

    print('------生成用户行为序列数据------')
    uid = []  # uid
    hist_iid = []  # 历史item id
    seq_length = []  # 序列长度
    target = []  # 标签
    item_id = []  # 目标物品id
    for user_id, df in tqdm(rating_df[['user_id', 'item_id']].groupby('user_id')):
        # df是分组后的所有行构成的DataFrame
        pos_list = df['item_id'].tolist()

        # 生成反例的方法
        def get_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
            return neg

        # 生成和正例个数相同的反例
        neg_list = [get_neg() for i in range(len(pos_list))]

        pos_temp_hist = []
        # 从第二个正例（下标为1）开始生成历史记录（第一个正例没有历史记录）
        for i in range(1, len(pos_list)):
            # 把上一条浏览记录作为历史存入hist，格式为[item_id]
            pos_temp_hist.append(pos_list[i - 1])
            if i > 20:  # 历史行为长度大于20时，做截断
                pos_temp_hist = pos_temp_hist[-20:]
            # 历史记录
            pos_hist_i = pos_temp_hist.copy()
            # 添加通用项
            uid.extend([user_id] * 2)
            seq_length.extend([len(pos_hist_i) + 1] * 2)
            # 添加正例
            hist_iid.append(pos_hist_i + [pos_list[i]])
            item_id.append(pos_list[i])
            target.append(1)
            # 添加负例
            hist_iid.append(pos_hist_i + [neg_list[i]])
            item_id.append(neg_list[i])
            target.append(0)

    print('------padding处理------')
    # 行为序列 padding
    hist_iid = pad_sequences(hist_iid, maxlen=maxlen + 1, padding='post')

    uid = np.array(uid)  # uid
    hist_iid = np.array(hist_iid)  # 历史item id
    seq_length = np.array(seq_length)  # 序列长度
    target = np.array(target)  # 标签
    item_id = np.array(item_id)  # 目标物品

    print('------shuffle处理------')
    # 打乱数据
    shuffle_ix = np.random.permutation(np.arange(len(uid)))
    uid = uid[shuffle_ix]
    hist_iid = hist_iid[shuffle_ix]
    seq_length = seq_length[shuffle_ix]
    target = target[shuffle_ix]
    item_id = item_id[shuffle_ix]

    print('------生成deepCTR训练数据集------')
    # 数据字典
    feature_dict = {
        'user': uid,  # uid
        'item_id': item_id,  # 候选商品id
        'hist_item_id': hist_iid,  # 历史行为序列
        'seq_length': seq_length  # 序列长度
    }

    # 特征配置信息
    feature_columns = [
        SparseFeat('user', user_id_max + 1, embedding_dim=10),
        # 暂时先这么做，如果是'bst'，正常embedding，如果是'bst_pre'，添加一个mask位
        SparseFeat('item_id', item_embed, embedding_dim=behavior_embed)
    ]

    feature_columns += [
        VarLenSparseFeat(
            SparseFeat(
                'hist_item_id', vocabulary_size=item_embed,
                embedding_dim=behavior_embed,
                embedding_name='item_id'
            ),
            maxlen=maxlen, length_name='seq_length'
        )
    ]

    # 用户行为特征名列表
    behavior_feature_list = ['item_id']

    # 生成 x
    x = {
        name: feature_dict[name]
        for name in get_feature_names(feature_columns)
    }

    y = target

    if pretrain is True:
        pretrain_fea_col = [
            VarLenSparseFeat(
                SparseFeat(
                    'hist_item_id', vocabulary_size=item_embed,
                    embedding_dim=behavior_embed,
                    embedding_name='item_id'
                ),
                maxlen=maxlen, length_name='seq_length'
            )
        ]
        return x, y, feature_columns, behavior_feature_list, item_id_max, pretrain_fea_col
    return x, y, feature_columns, behavior_feature_list, item_id_max


def create_sequence_dien_dataset(file, dataset_name, maxlen=20, behavior_embed=16, masked=True, pretrain=True):
    print('------开始处理数据------')
    # 依次读取存到pkl文件的数据
    rating_df = pd.read_pickle(file)
    if dataset_name in ['cds', 'beauty']:
        rating_df.columns = ['user_id', 'item_id', 'time', 'category']
    elif dataset_name == 'epinions':
        rating_df.columns = ['item_id', 'user_id', 'time']
    elif dataset_name == 'electric':
        with open(file, 'rb') as f:
            rating_df = pickle.load(f)
            cate_list = pickle.load(f)
            user_count, item_count, cate_count, example_count = pickle.load(f)
        rating_df.columns = ['user_id', 'item_id', 'time']

    # 获取最大的item_id
    item_id_max = rating_df['item_id'].max()
    # 获取最大的user_id
    user_id_max = rating_df['user_id'].max()

    if masked:
        item_embed = item_id_max + 2
    else:
        item_embed = item_id_max + 1

    print('------生成用户行为序列数据------')
    uid = []  # uid
    hist_id = []  # 历史item id
    neg_hist_id = []  # 反例item id
    seq_length = []  # 序列长度
    target = []  # 标签
    item_id = []  # 目标物品id
    for user_id, df in tqdm(rating_df[['user_id', 'item_id']].groupby('user_id')):
        # df是分组后的所有行构成的DataFrame
        pos_list = df['item_id'].tolist()

        # 生成反例的方法
        def get_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
            return neg

        # 生成和正例个数相同的反例
        neg_list = [get_neg() for i in range(len(pos_list))]

        pos_temp_hist = []
        # 从第二个正例（下标为1）开始生成历史记录（第一个正例没有历史记录）
        for i in range(1, len(pos_list)):
            # 把上一条浏览记录作为历史存入hist，格式为[item_id]
            neg_temp_list = pos_temp_hist.copy()
            neg_temp_list.append(neg_list[i - 1])
            # print(neg_temp_list)
            pos_temp_hist.append(pos_list[i - 1])
            if i > 20:  # 历史行为长度大于20时，做截断
                pos_temp_hist = pos_temp_hist[-20:]
            # 历史记录
            pos_hist_i = pos_temp_hist.copy()
            # 添加通用项
            uid.extend([user_id] * 2)
            seq_length.extend([len(pos_hist_i)] * 2)
            neg_hist_id.extend([neg_temp_list] * 2)
            hist_id.extend([pos_hist_i] * 2)
            # 添加正例
            item_id.append(pos_list[i])
            target.append(1)
            # 添加负例
            item_id.append(neg_list[i])
            target.append(0)

    print('------padding处理------')
    # 行为序列 padding
    hist_iid = pad_sequences(hist_id, maxlen=maxlen, padding='post')
    neg_hist_id = pad_sequences(neg_hist_id, maxlen=maxlen, padding='post')

    uid = np.array(uid)  # uid
    hist_iid = np.array(hist_iid)  # 历史item id
    neg_hist_id = np.array(neg_hist_id)
    seq_length = np.array(seq_length)  # 序列长度
    target = np.array(target)  # 标签
    item_id = np.array(item_id)  # 目标物品


    print('------shuffle处理------')
    # 打乱数据
    shuffle_ix = np.random.permutation(np.arange(len(uid)))
    uid = uid[shuffle_ix]
    hist_iid = hist_iid[shuffle_ix]
    neg_hist_id = neg_hist_id[shuffle_ix]
    seq_length = seq_length[shuffle_ix]
    target = target[shuffle_ix]
    item_id = item_id[shuffle_ix]

    print('------生成deepCTR训练数据集------')
    # 数据字典
    feature_dict = {
        'user': uid,  # uid
        'item_id': item_id,  # 候选商品id
        'hist_item_id': hist_iid,  # 历史行为序列
        'seq_length': seq_length,  # 序列长度
        'neg_hist_item_id': neg_hist_id
    }

    # 特征配置信息
    feature_columns = [
        SparseFeat('user', user_id_max + 1, embedding_dim=10),
        # 暂时先这么做，如果是'bst'，正常embedding，如果是'bst_pre'，添加一个mask位
        SparseFeat('item_id', item_embed, embedding_dim=behavior_embed)
    ]

    feature_columns += [
        VarLenSparseFeat(
            SparseFeat(
                'hist_item_id', vocabulary_size=item_embed,
                embedding_dim=behavior_embed,
                embedding_name='item_id'
            ),
            maxlen=maxlen, length_name='seq_length'
        ),
        VarLenSparseFeat(
            SparseFeat(
                'neg_hist_item_id', vocabulary_size=item_embed,
                embedding_dim=behavior_embed,
                embedding_name='item_id'
            ),
            maxlen=maxlen, length_name='seq_length'
        )
    ]

    # 用户行为特征名列表
    behavior_feature_list = ['item_id']

    # 生成 x
    x = {
        name: feature_dict[name]
        for name in get_feature_names(feature_columns)
    }

    y = target

    if pretrain is True:
        pretrain_fea_col = [
            VarLenSparseFeat(
                SparseFeat(
                    'hist_item_id', vocabulary_size=item_embed,
                    embedding_dim=behavior_embed,
                    embedding_name='item_id'
                ),
                maxlen=maxlen, length_name='seq_length'
            )
        ]
        return x, y, feature_columns, behavior_feature_list, item_id_max, pretrain_fea_col
    return x, y, feature_columns, behavior_feature_list, item_id_max


def create_srd_dataset(item_seq, seq_len, max_len, item_id_max, prob_rate=0.2):
    """
        创建四分类任务的数据，进行 原始|shuffle|random|delete 的四分类
        注意此方法需要增加 [mask] 位置，所以在 embedding 词典长度要 + 1。
        每个分类生成15%（或20%）
    """
    print('------Shuffle + Random + Delete 预训练数据创建开始-------')

    # 序列长度列表
    end_index = seq_len.tolist()
    # item序列
    item_seq = item_seq.tolist()

    labels = []
    result_sequences = []

    for i, instance in tqdm(enumerate(item_seq)):
        # 默认 label 全 0
        label = torch.zeros(max_len)

        # 非padding位index列表
        no_padd_index = list(range(0, end_index[i]))

        classify_index = 1
        # 四分类的padding位置当做一个单独的分类
        label[no_padd_index] = 1

        # 为保证每个分类至少有一个，对长度小于4的，只做二分类
        if end_index[i] < 5:
            result_sequences.append(instance)
            labels.append(np.array(label))
            continue

        # srd的数量，用来切片下面的 srd 位置列表
        split_num = int(end_index[i] * prob_rate)

        # 将非padding位的index打乱，作为选取 srd 位置的依据
        alter_index = no_padd_index.copy()
        random.shuffle(alter_index)

        '''shuffle操作'''
        # 将非padding位的index进行随机打乱
        shuffle_index = no_padd_index.copy()
        random.shuffle(shuffle_index)

        # 替换shuffle位置的元素为shuffle后的
        for place in alter_index[:split_num + 1]:
            instance[place] = shuffle_index[place]
            # 设置改位置 label 为1
            label[place] = classify_index + 1

        '''random操作'''
        for place in alter_index[split_num + 1: split_num * 2 + 1]:
            # 替换random位置的元素为random后的
            instance[place] = neg_sample(instance, item_id_max)
            # label位置为1
            label[place] = classify_index + 2

        '''delete操作'''

        for place in alter_index[split_num * 2 + 1: split_num * 3 + 1]:
            # 替换为[mask]，表示删除位
            instance[place] = item_id_max + 1
            # label位置为1
            label[place] = classify_index + 3

        result_sequences.append(instance)
        labels.append(np.array(label))

    result_sequences = np.array(result_sequences)
    # 把长度为1的seq_len过滤掉
    seq_len = np.array(seq_len)
    # seq_len = np.array(list(filter(lambda x: x != 1, seq_len)))
    labels = np.array(labels)

    print('------预训练数据创建完成-------')
    return result_sequences, seq_len, labels


# 生成预训练数据
def create_shuffle_random_dataset(item_seq, seq_len, max_len, item_id_max, prob_rate, classify=3):
    print('------Shuffle + Random 预训练数据创建开始-------')

    # 序列长度列表
    end_index = seq_len.tolist()
    # item序列
    item_seq = item_seq.tolist()
    # 创建对应的空列表
    labels = []
    result_sequences = []

    for i, instance in tqdm(enumerate(item_seq)):
        # 只有一个元素的序列就不生成了
        if end_index[i] < 2:
            continue
        # 复制一份【不在原序列上操作】
        label = torch.zeros(max_len)

        shuffle_index = list(range(0, end_index[i]))

        index = 0
        if classify == 4:
            label[shuffle_index] = 1
            index = 1

        # 对非padding位置进行shuffle(index)
        random.shuffle(shuffle_index)
        copy_shuffle = shuffle_index + [0] * (max_len - end_index[i])
        # 随机15%的位置
        shuffle_sample = random.sample(list(range(0, end_index[i])), math.ceil(end_index[i] * (prob_rate - 0.1)))
        # print('shuffle_sample')
        # print(shuffle_sample)
        if shuffle_sample is not None:
            # 替换shuffle位置的元素为shuffle后的
            for place in shuffle_sample:
                instance[place] = copy_shuffle[place]
                # label位置为1
                # !!!! index有问题！！！
                label[place] = index + 1

        # 注意random位置必须不能和shuffle重合
        copy_random = [neg_sample(instance, item_id_max) for j in range(end_index[i])]
        # 采样copy位置[去除和shuffle_sample重合的位置]
        random_sample = random.sample(list(range(0, end_index[i])), math.ceil(end_index[i] * prob_rate))
        random_sample = [sam for sam in random_sample if sam not in shuffle_sample]

        if random_sample is not None:
            for place in random_sample:
                # 替换random位置的元素为random后的
                instance[place] = copy_random[place]
                # label位置为1
                label[place] = index + 2

        result_sequences.append(instance)
        labels.append(np.array(label))

    result_sequences = np.array(result_sequences)
    # 把长度为1的seq_len过滤掉
    seq_len = np.array(list(filter(lambda x: x != 1, seq_len)))
    labels = np.array(labels)

    print('------预训练数据创建完成-------')
    return result_sequences, seq_len, labels


def create_delete_random_dataset(item_seq, seq_len, max_len, item_id_max, prob_rate, classify=4):
    # 覆盖D+R+P 4分类 和 D+R 3分类
    print('------Random + Delete 预训练数据创建开始-------')
    '''实验思路是增加random占到的比重，1:2:1的比例生成SRD数据，这个地方先写死，看实验效果再调整（prob_rate参数暂时不用）'''

    # 每个序列单独创建一个全0张量 label
    # 序列长度列表
    end_index = seq_len.tolist()
    # item序列
    item_seq = item_seq.tolist()
    # 创建对应的空列表
    result_sequences = []
    labels = []
    # 遍历每一条序列，生成dr数据
    for i, instance in tqdm(enumerate(item_seq)):
        # 只有一个元素的序列就不生成了
        """
        这里考虑增加处理方式，原始处理方式为长度为1的直接删除，
        长度为2及以上的根据有效长度（seq_len）生成DR位置（保证RD每个至少生成一位）  [如果要恢复代码，按照这个来]
        代码改成： （version 2）
            对于长度为2的，只Delete掉一位
            对于长度为3的，只Random掉一位
            对于长度为4的，D和R各一位
        """
        if end_index[i] is 1:
            continue
        # 生成全0 label
        label = torch.zeros(max_len)
        # 非padding位index
        seq_index = list(range(0, end_index[i]))
        # label index
        index = 0
        """
            两种分类策略（使用classify参数指定）：
            三分类：original:0/delete:1/random:2
            四分类：original:0/padding:1/delete:2/random:3
        """
        if classify == 4:
            label[seq_index] = 1
            index = 1

        if end_index[i] in [2, 4]:
            delete_sample = random.sample(
                list(range(0, end_index[i])),
                1
            )
        else:
            delete_sample = random.sample(
                list(range(0, end_index[i])),
                math.ceil(end_index[i] * prob_rate[0])
            )
        if delete_sample is not None:
            for place in delete_sample:
                # 替换random位置的元素为random后的
                instance[place] = item_id_max + 1
                # label位置为1
                label[place] = index + 1

        # 注意random位置必须不能和delete重合
        # copy_random = [neg_sample(instance, item_id_max) for j in range(end_index[i])]
        # 生成序列有效长度个反例
        if end_index[i] in [3, 4]:
            random_sample = random.sample(list(range(0, end_index[i])), 1)
        else:
            # 采样copy位置[去除和shuffle_sample重合的位置]
            random_sample = random.sample(list(range(0, end_index[i])), math.ceil(end_index[i] * prob_rate[1]))
        copy_random = [neg_sample(instance, item_id_max) for j in range(len(random_sample))]

        random_sample = list(set(random_sample) - set(delete_sample))
        flag = 0
        if random_sample is not None:
            for place in random_sample:
                # 替换random位置的元素为random后的
                instance[place] = copy_random[flag]
                flag += 1
                # label位置为1
                label[place] = index + 2

        # 注意random位置必须不能和shuffle重合

        result_sequences.append(instance)
        labels.append(np.array(label))

    result_sequences = np.array(result_sequences)
    # 把长度为1的seq_len过滤掉
    seq_len = np.array(list(filter(lambda x: x != 1, seq_len)))
    labels = np.array(labels)

    print('------预训练数据创建完成-------')
    return result_sequences, seq_len, labels
