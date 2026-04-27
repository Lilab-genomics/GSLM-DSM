import torch
import torch.nn as nn
import torch.nn.functional as F
import math


###门控交叉融合 (Gated Cross Fusion), 通过门控机制动态控制信息流，增强重要特征，抑制噪声。

class TextCNN_block1(nn.Module):
    def __init__(self, vocab_size, embedding_size_DLM1, embedding_size_DLM2, DLM_seq_len1, DLM_seq_len2, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block1, self).__init__()
        # 初始化多分枝卷积层
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=embedding_size_DLM1,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])  # 对每个filter size创建一个卷积层
        # 定义全连接层
        self.fc1 = nn.Linear(1536, 64)  # 将卷积后的特征映射到512维
        self.fc = nn.Sequential(
            nn.Linear(1536, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        # 定义批归一化层
        self.batchnorm1 = nn.BatchNorm1d(64)

    def forward(self, DLM_fea1):
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded1 = DLM_fea1.permute(0, 2, 1)
        # 应用卷积层并使用Mish激活函数
        DLM_conved1 = [self.Mish1(conv(DLM_embedded1)) for conv in self.convs1]
        # 池化层，使用最大池化
        DLM_pooled1 = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 8)) for conv in DLM_conved1]
        # 多分支线性展开
        DLM_flatten1 = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled1]
        # 将各分支连接在一起
        DLM_cat1 = self.dropout1(torch.cat(DLM_flatten1, dim=1))

        return self.fc(DLM_cat1), DLM_cat1  # 返回1920维中间特征

        # 使用线性层进行维度变换，并应用批归一化
        # DLM_cat_i = self.fc1(DLM_cat)
        # DLM_cat_i = self.batchnorm1(DLM_cat_i)

        # 输出特征并分类
        # return self.fc(DLM_cat_i), DLM_cat_i  # 返回最终分类结果和中间特征


class TextCNN_block2(nn.Module):
    def __init__(self, vocab_size, embedding_size_DLM1, embedding_size_DLM2, DLM_seq_len1, DLM_seq_len2, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block2, self).__init__()
        # 初始化多分枝卷积层
        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels=embedding_size_DLM2,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])  # 对每个filter size创建一个卷积层
        # 定义全连接层
        self.fc2 = nn.Linear(1536, 64)  # 将卷积后的特征映射到512维
        self.fc = nn.Sequential(
            nn.Linear(1536, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout2 = nn.Dropout(dropout)
        self.Mish2 = nn.Mish()
        # 定义批归一化层
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, DLM_fea2):
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded2 = DLM_fea2.permute(0, 2, 1)
        # 应用卷积层并使用Mish激活函数
        DLM_conved2 = [self.Mish2(conv(DLM_embedded2)) for conv in self.convs2]
        # 池化层，使用最大池化
        DLM_pooled2 = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 8)) for conv in DLM_conved2]
        # 多分支线性展开
        DLM_flatten2 = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled2]
        # 将各分支连接在一起
        DLM_cat2 = self.dropout2(torch.cat(DLM_flatten2, dim=1))

        return self.fc(DLM_cat2), DLM_cat2  # 返回1536维中间特征

        # 使用线性层进行维度变换，并应用批归一化
        # DLM_cat_i = self.fc1(DLM_cat)
        # DLM_cat_i = self.batchnorm1(DLM_cat_i)

        # 输出特征并分类
        # return self.fc(DLM_cat_i), DLM_cat_i  # 返回最终分类结果和中间特征


class GatedCrossFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征投影
        self.proj1 = nn.Linear(1536, 512)
        self.proj2 = nn.Linear(1536, 512)

        # 门控单元
        self.gate = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),  # 添加非线性激活函数
            nn.Linear(512, 512),
            nn.Sigmoid()
        )

        # 残差连接
        self.residual = nn.Linear(1536 * 2, 512)  # 修改输入维度为 1536 * 2

        # 归一化层
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, feat1, feat2):
        # 特征投影
        proj1 = self.bn1(F.relu(self.proj1(feat1)))  # [B,512]
        proj2 = self.bn2(F.relu(self.proj2(feat2)))  # [B,512]

        # 计算门控权重
        gate = self.gate(torch.cat([proj1, proj2], dim=1))  # [B,512]

        # 加权融合
        fused = gate * proj1 + (1 - gate) * proj2

        # 残差连接
        residual = self.residual(torch.cat([feat1, feat2], dim=1))
        return fused + residual


class SplicePred(nn.Module):
    def __init__(self, vocab_size, embedding_size_DLM1, embedding_size_DLM2, DLM_seq_len1, DLM_seq_len2, n_filters,
                 filter_sizes, output_dim, dropout):
        super(SplicePred, self).__init__()

        # 实例化 TextCNN_block1，确保传递所有必需的参数并保持正确的顺序
        self.DLM_encoder1 = TextCNN_block1(vocab_size, embedding_size_DLM1, embedding_size_DLM2,
                                           DLM_seq_len1, DLM_seq_len2, n_filters, filter_sizes,
                                           output_dim, dropout)

        # 实例化 TextCNN_block2，确保传递所有必需的参数并保持正确的顺序
        self.DLM_encoder2 = TextCNN_block2(vocab_size, embedding_size_DLM1, embedding_size_DLM2,
                                           DLM_seq_len1, DLM_seq_len2, n_filters, filter_sizes,
                                           output_dim, dropout)

        self.hierarchical = GatedCrossFusion()
        # 定义全连接层用于分类
        self.fc3 = nn.Linear(512, output_dim)  # 假设data1和data2的维度都是1536   # 修改输入维度为512

    def forward(self, DLM_fea1, DLM_fea2):
        _, data1 = self.DLM_encoder1(DLM_fea1)  # GPN-MSA特征经过TextCNN
        _, data2 = self.DLM_encoder2(DLM_fea2)  # SpliceBERT特征经过TextCNN

        fea = self.hierarchical(data1, data2)
        return self.fc3(fea), fea
