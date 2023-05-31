# SSPT4Rec

## 说明

本仓库是论文 SSPT4Rec算法[1] 的开源代码。

SSPT4Rec基于DeepCTR-Torch库实现，直接使用原库中的DIN、DIEN进行试验，并根据论文和作者提供的源码复现了DSIN、BST、CORE、LightSANs等算法。

SSPT4Rec的训练分为两个阶段，SSPT4Rec/deepctr_torch/models/dric 下的DRIC是预训练阶段的模型实现，SSPT4Rec/deepctr_torch/models/trm4rec 下的 Trm4Rec 是微调阶段的模型实现。

SSPT4Rec/amazon_beauty_pretrain_epochs_log.ipynb 文件中展示了以 amazon beauty 数据集为例进行实验的步骤。

## 实验环境要求：

- python == 3.6

- torch == 1.12.1

- tensorflow == 2.4.2

- numpy == 1.22.4

> [1] Xu Yuhao, Wang Zhenhai, Wang Zhiru, Fan Rong, Wang Xing. A Recommendation Algorithm Based on a Self-supervised Learning Pretrain Transformer[J]. Neural Processing Letters, 2022. https://doi.org/10.1007/s11063-022-11053-8

