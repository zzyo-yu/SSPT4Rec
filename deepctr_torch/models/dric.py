# -*- coding:utf-8 -*-
"""
Author:
    Yuhao Xu
"""
from ..layers.utils import slice_arrays
from deepctr_torch.layers.transformer import TransformerEncoder

from ..inputs import *
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.basemodel import BaseModel
from ..callbacks import History

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList


class DRIC(BaseModel):
    def __init__(
            self, dnn_feature_columns, history_feature_list, seq_max_len, classify=3,
            trm_num=1, trm_head_num=2, trm_hidden_size=64, trm_overlay=True,
            drop_prob=0.2, l2_reg_embedding=1e-6, dnn_dropout=0.0, init_std=0.0001,
            device='cpu', gpus=None, seed=1024, task='binary'
    ):
        super(DRIC, self).__init__(
            [], dnn_feature_columns,
            l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
            init_std=init_std, seed=seed, task=task,
            device=device, gpus=gpus
        )
        self.classify = classify
        self.seq_max_len = seq_max_len
        self.trm_hidden_size = trm_hidden_size

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)
        ) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []

        # ['item_id'] -> ['hist_item_id']
        self.history_fc_names = list(
            map(lambda x: "hist_" + x, history_feature_list)
        )

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        self.position_embedding = nn.Embedding(self.seq_max_len, trm_hidden_size)

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

        self.dropout = nn.Dropout(drop_prob)
        self.dense = nn.Linear(self.trm_hidden_size, classify)  # 0 (non-replaced), 1(padding), 2 (delete), 3 (random)
        self.loss_fct = nn.CrossEntropyLoss()
        self.to(device)
        
        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

    def forward(self, X):
        keys_emb_list = embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.history_feature_columns,
            return_feat_list=self.history_fc_names,
            to_list=True
        )  # feature_num * [B, T, e]

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

        attention_mask = self.get_attention_mask(keys_values[-1], bidirectional=True)

        position_ids = torch.arange(self.seq_max_len, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(len(X), self.seq_max_len).cuda()
        position_embedding = self.position_embedding(position_ids)

        keys_emb += position_embedding

        trm = self.transformer(
            keys_emb, attention_mask, output_all_encoded_layers=True
        )
        seq_output = trm[-1]

        # dropout
        seq_output = self.dropout(seq_output)
        logits = self.dense(seq_output)

        return logits

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,labels=[0, 1, 2, 3],
            validation_data=None, shuffle=True, callbacks=None, save_weight=None, save_path=None):

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)),
            torch.from_numpy(y)
        )

        if batch_size is None:
            batch_size = 256

        model = self.train()

        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=8
        )

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        
        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for i, (x_train, y) in t:
                        x = x_train.to(self.device).float()
                        y = y.to(self.device).long()
                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        loss = self.loss_fct(y_pred.view(-1, self.classify), y.view(-1))
                        reg_loss = self.get_regularization_loss()
                        total_loss = loss + reg_loss
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()
                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(
                                    metric_fun(
                                        torch.reshape(y, (-1,)).cpu().data.numpy(),
                                        torch.reshape(y_pred, (-1, self.classify)).cpu().data.numpy().astype("float64"),
                                        labels=labels
                                    )
                                )

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            epoch_logs["loss"] = total_loss_epoch / steps_per_epoch
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
            
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, labels=labels, batch_size = batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])

                print(eval_str)
                
            if save_weight is not None:
                if (epoch + 1) % save_weight == 0:
                    torch.save(
                        self.state_dict(), 
                        save_path + str(epoch + 1) + '.pth'
                    )
                    
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

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
    
    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)
        return np.concatenate(pred_ans).astype("float64")

    def evaluate(self, x, y, labels=[0,1,2,3], batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            if name == 'crossentropy':
                y = y.reshape(-1, )
                pred_ans = pred_ans.reshape(-1, len(labels))
                eval_result[name] = metric_fun(y, pred_ans, labels=labels)
            else:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result