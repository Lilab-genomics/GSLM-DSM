#!/usr/bin/env python
# coding=gbk

import argparse


def get_config():
    parse = argparse.ArgumentParser(description='default config')
    parse.add_argument('-a', type=str, default='模型')

    # 数据参数
    parse.add_argument('-vocab_size', type=int, default=4, help='The size of the vocabulary')  # 词汇表大小
    parse.add_argument('-output_size', type=int, default=1, help='Number of mutation functions')
    #    parse.add_argument('-CV', type=bool, default=True, help='是否进行交叉验证') ### True代表进行五折交叉验证，则输出五折交叉验证结果，详见./result/sta_test.csv
    parse.add_argument('-CV', type=bool, default=False,
                       help='是否进行交叉验证')  ### False代表不进行五折交叉验证，则输出测试集结果，详见./result/sta_test.csv

    # 训练参数
    parse.add_argument('-batch_size', type=int, default=128, help='Batch size')  # 批次大小
    parse.add_argument('-epochs', type=int, default=100)  # 迭代次数
    parse.add_argument('-learning_rate', type=float, default=0.0001)  # 学习率  raw-0.0001
    parse.add_argument('-threshold', type=float, default=0.5)  # 用于分类的阈值
    parse.add_argument('-early_stop', type=int, default=10)  # 早停步数

    # 模型参数
    parse.add_argument('-model_name', type=str, default='TextCNN', help='Name of the model')  # 选择模型
    #    parse.add_argument('-model_name', type=str, default='CNN', help='Name of the model')
    #    parse.add_argument('-model_name', type=str, default='ResNet', help='Name of the model')
    #    parse.add_argument('-model_name', type=str, default='RNN', help='Name of the model')
    #    parse.add_argument('-model_name', type=str, default='BiLSTM', help='Name of the model')
    #    parse.add_argument('-model_name', type=str, default='BiGRU', help='Name of the model')

    parse.add_argument('-embedding_size_DLM1', type=int, default=768,
                       help='Dimension of the embedding')  # DNA语言模型特征维度，默认为 768。

    parse.add_argument('-DLM_seq_len1', type=int, default=128,
                       help='Length of the sequence in DLM model')  #  DNA语言模型序列长度，默认为 128。

    parse.add_argument('-embedding_size_DLM2', type=int, default=512,
                       help='Dimension of the embedding')  # RNA语言模型特征维度，默认为 512。

    parse.add_argument('-DLM_seq_len2', type=int, default=503, help='Length of the sequence')  # RNA模型序列长度 503

    parse.add_argument('-dropout', type=float, default=0.6)  # Dropout 概率，默认为 0.6。
    parse.add_argument('-filter_num', type=int, default=64, help='Number of the filter')  # 卷积核数量，默认为 64。
    parse.add_argument('-filter_size', type=list, default=[3, 4, 5], help='Size of the filter')  # 卷积核大小，默认为 [3, 4, 5]。
    #parse.add_argument('-dropout2', type=float, default=0.6)  # Dropout 概率，默认为 0.6。
    #parse.add_argument('-filter_num2', type=int, default=64, help='Number of the filter')  # 卷积核数量，默认为 64。
    #parse.add_argument('-filter_size2', type=list, default=[3, 4, 5], help='Size of the filter')  # 卷积核大小，默认为 [3, 4, 5]。

    # 路径参数
    ## 训练集的特征文件
    parse.add_argument('-train_direction1', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/train_GPNMSA.npy',
                       help='The ref-seq feature of training set')

    parse.add_argument('-train_direction2', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/train_Splicebert.npy',
                       help='The alt-seq feature of training set')
    
    ## 训练集的标签文件。此处从两个文件中分别加载训练集特征和标签，所以应保证两个文件的样本顺序一致
    parse.add_argument('-train_label_direction', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/Balance_train_CB.csv',
                       help='Path of the label of training data')

    
    ## 测试集的特征文件
    #全部测试集
    parse.add_argument('-test_direction1', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/test_GPNMSA.npy',
                       help='The alt-seq feature of the test data')

    parse.add_argument('-test_direction2', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/test_Splicebert.npy',
                       help='The alt-seq feature of test set')
    #非经典剪接突变数据测试集
    parse.add_argument('-test_direction_non_class1', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/test_GPNMSA_non-canonical.npy',
                       help='The alt-seq feature of the test data')

    parse.add_argument('-test_direction_non_class2', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/test_Splicebert_non-canonical.npy',
                       help='The alt-seq feature of test set')

    #剪接突变数据测试集
    parse.add_argument('-test_direction_class1', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/test_GPNMSA_canonical.npy',
                       help='The alt-seq feature of the test data')

    parse.add_argument('-test_direction_class2', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/test_Splicebert_canonical.npy',
                       help='The alt-seq feature of test set')






    ## 测试集的标签文件。此处从两个文件中分别加载测试集特征和标签，所以应保证两个文件的样本顺序一致
    parse.add_argument('-test_label_direction', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/Balance_test_CB.csv',
                       help='Path of the label of test data')

    parse.add_argument('-test_label_non_class', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/test_noncanonical.csv',
                       help='Path of the label of test data')

    parse.add_argument('-test_label_class', type=str,
                       default='/data2/yanmengxiang/projects/GSLM-DSM/input/test_canonical.csv',
                       help='Path of the label of test data')

    config = parse.parse_args()
    return config
