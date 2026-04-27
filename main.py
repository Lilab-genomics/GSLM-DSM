#!/usr/bin/env python
# coding=gbk

import csv
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import estimate
from config import sta_config
from models.SplicePred import SplicePred
from train import DataTrain, predict, CosineScheduler, feature

""" 
estimate：用于评估模型性能的模块。
sta_config：配置文件，包含模型训练和测试的各种参数。
SplicePred：定义了模型的结构。
DataTrain：用于数据预处理和训练的类。
predict：用于模型预测的函数。
CosineScheduler：学习率调度器。
"""
torch.manual_seed(20230226)
# 设置了PyTorch的随机种子，确保每次运行程序时生成的随机数是相同的,这对于实验的可重复性非常重要。

torch.backends.cudnn.deterministic = True
# 设置CUDNN确定性模式，确保CUDNN在使用GPU加速时的行为是确定性的。虽然这可能会稍微降低性能，但可以确保实验结果的可重复性。

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义了一个设备变量DEVICE，根据系统是否支持CUDA来选择使用GPU（cuda)还是CPU。

bases = 'ATCG'


# 定义了一个字符串变量bases，表示DNA的四种碱基：腺嘌呤（A）、胸腺嘧啶（T）、胞嘧啶（C）和鸟嘌呤（G）


# 用于加载特征文件和带标签的数据文件
# 加载DNA语言模型的特征文件
# 若特征文件为.pth文件，则使用torch.load；若特征文件为.npy文件，则使用np.load
def getSequenceData(direction1, direction2, label_direction):
    # 检查文件扩展名并决定加载方法
    if direction1.endswith('.pth'):
        data1 = torch.load(direction1)
    elif direction1.endswith('.npy'):
        data1 = torch.from_numpy(np.load(direction1)).float()  # 假设.npy文件中的数据是浮点型的
    else:
        raise ValueError(f"Unsupported file format: {direction1}")

        # 检查文件扩展名并决定加载方法
    if direction2.endswith('.pth'):
        data2 = torch.load(direction2)
    elif direction2.endswith('.npy'):
        data2 = torch.from_numpy(np.load(direction2)).float()  # 假设.npy文件中的数据是浮点型的
    else:
        raise ValueError(f"Unsupported file format: {direction2}")

    # 使用 pandas 读取对应标签的CSV文件。然后提取标签 (Label) 字段的数据。
    Frame = pd.read_csv(label_direction)
    label = torch.tensor(Frame["Label"].values, dtype=torch.long)  # 带标签的文件中”Label“列为标签数据
    return data1, data2, label


# 用于加载和准备训练与测试数据集
# train_direction, chrom_train_direction, train_label_direction: 指向训练数据、染色体信息以及标签文件的位置。
# test_direction, chrom_test_direction, test_label_direction: 类似于训练数据，但这些是测试数据的相关路径。
# batch: DataLoader中的批量大小。
# encode: 数据表示形式，可以是'embedding'或'squence'。
# cv: 布尔值，决定是否使用交叉验证。
# SH: 在DataLoader中是否打乱数据。


def data_load(train_direction1, train_direction2, train_label_direction, test_direction1, test_direction2,
              test_label_direction, batch, encode='embedding', cv=True, SH=True):
    dataset_train, dataset_test = [], []
    # 定义两个空列表dataset_train和dataset_test来存储训练和测试的数据加载器。
    dataset_va = None
    # 如果启用了交叉验证(cv=True)，则还初始化一个空列表dataset_va用于保存验证数据加载器。
    assert encode in ['embedding', 'sequence'], 'There is no such representation!!!'
    # 检查encode参数是否为允许的值之一；如果不是，则抛出异常。

    # 当启用交叉验证时，通过调用getSequenceData获取训练数据，并将其划分为5折进行交叉验证。
    # 对每折数据，创建训练和验证数据集，并将它们转换为PyTorch的TensorDataset对象。
    # 为每个数据集创建DataLoader实例，并根据指定的批量大小和是否打乱数据来配置。
    # 将创建好的DataLoader添加到相应的列表中。
    if cv:
        dataset_va = []
        x_train1, x_train2, y_train = getSequenceData(train_direction1, train_direction2, train_label_direction)
        print(f"x_train1 shape: {x_train1.shape}, x_train2 shape: {x_train2.shape}, y_train shape: {y_train.shape}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

        for i, (train_index, test_index) in enumerate(cv.split(x_train1, y_train)):
            data_train1, data_train2, label_train = x_train1[train_index], x_train2[train_index], y_train[train_index]
            data_test1, data_test2, label_test = x_train1[test_index], x_train2[test_index], y_train[test_index]

            train_data = TensorDataset(torch.tensor(data_train1), torch.tensor(data_train2), torch.tensor(label_train))
            test_data = TensorDataset(torch.tensor(data_test1), torch.tensor(data_test2), torch.tensor(label_test))

            dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=SH))
            dataset_va.append(DataLoader(test_data, batch_size=batch, shuffle=SH))

    # 直接从给定的路径加载训练数据，并同样创建TensorDataset和DataLoader。
    else:
        print("encode train")
        x_train1, x_train2, y_train = getSequenceData(train_direction1, train_direction2, train_label_direction)
        print(f"x_train1 shape: {x_train1.shape}, x_train2 shape: {x_train2.shape}, y_train shape: {y_train.shape}")
        # Create datasets
        train_data = TensorDataset(x_train1, x_train2, y_train)
        dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=SH))

        # 加载训练数据
    print("encode test")
    x_test1, x_test2, y_test = getSequenceData(test_direction1, test_direction2, test_label_direction)
    print(f"x_test1 shape: {x_test1.shape}, x_test2 shape: {x_test2.shape}, y_train shape: {y_test.shape}")

    # Create datasets
    # 不论是否启用交叉验证，都直接从提供的路径加载测试数据，并构建相应的TensorDataset和DataLoader。
    test_data = TensorDataset(x_test1, x_test2, y_test)
    dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=False))
    # dataset_test.append(dataset_test)

    return dataset_train, dataset_va, dataset_test
    # 返回三个列表：包含训练数据加载器的dataset_train、可选的验证数据加载器dataset_va（如果启用了交叉验证）、以及测试数据加载器dataset_test。


# 从end减去start来得到总的秒数差，即epoch_time
def spent_time(start, end):
    epoch_time = end - start
    minute = int(epoch_time / 60)
    secs = int(epoch_time - minute * 60)
    return minute, secs


# 函数 save_results 用于将模型的评估结果保存到一个CSV文件中。
# model_name: 字符串，表示模型的名称。
# start: 开始的时间戳（浮点数）。
# end: 结束的时间戳（浮点数）。
# test_score: 一个包含多个评估指标值的列表，如召回率(Recall)、特异性(SPE)、精确度(Precision)、F1分数(F1)、马修斯相关系数(MCC)、准确率(Acc)、AUC和AUPR等。
# file_path: 要保存结果的CSV文件路径。
def save_results(model_name, start, end, test_score, file_path):
    #    title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']
    # title = ['Model', 'Recall', 'SPE', 'Precision', 'F1', 'MCC', 'Acc', 'AUC', 'AUPR',
    # 'RunTime', 'Test_Time']
    # 定义标题行
    title = ['Model', 'Recall', 'SPE', 'wujianlv', 'loujianlv', 'Precision', 'F1', 'MCC', 'Acc', 'AUC', 'AUPR',
             'RunTime', 'Test_Time']

    # 用time.strftime 获取当前的日期和时间，并格式化为字符串。
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # 创建一个列表 content，其中包含模型名称、各个评估指标的值（格式化为三位小数）、运行时间（以秒为单位）和当前时间。
    content = [[model_name,
                '%.3f' % test_score[0],
                '%.3f' % test_score[1],
                '%.3f' % test_score[2],
                '%.3f' % test_score[3],
                '%.3f' % test_score[4],
                '%.3f' % test_score[5],
                '%.3f' % test_score[6],
                '%.3f' % test_score[7],
                '%.3f' % test_score[8],
                '%.3f' % test_score[9],
                '%.3f' % (end - start),
                now]]

    # 如果文件已经存在，则读取文件的内容，检查第一行是否与预定义的标题行一致。
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None, encoding='gbk')
        one_line = list(data.iloc[0])
        # 如果一致，则直接追加新的结果；如果不一致，则先写入标题行再追加新结果。
        if one_line == title:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerows(content)
        else:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)
                writer.writerows(content)
    # 如果文件不存在，则创建新文件并写入标题行和新结果。
    else:
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)


# 主函数，用于执行模型训练、评估以及结果保存等任务。它还包含了一些辅助操作，如记录时间戳、保存模型参数、预测测试集并计算性能指标等。
def main(paths=None):
    # 打印当前操作的描述。
    print("doing: Splice predition")

    # 获取当前时间并记录到文件中。
    Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 打印命令行参数到文件中。
    parse_file = f"./result/sta_pares.txt"
    file1 = open(parse_file, 'a')
    file1.write(Time)
    file1.write('\n')
    print(args, file=file1)
    file1.write('\n')
    file1.close()

    # 设置结果文件路径。
    file_path = "{}/{}.csv".format('result', 'sta_test')

    # 调用 data_load 函数加载训练、验证（如果启用交叉验证）和测试数据集。
    print("Data is loading......")
    train_datasets, va_datasets, test_datasets = data_load(args.train_direction1, args.train_direction2,
                                                           args.train_label_direction, args.test_direction1,
                                                           args.test_direction2, args.test_label_direction,
                                                           args.batch_size, cv=args.CV)
    print("Data is loaded!")
    all_test_score = 0

    # 记录开始时间 start_time。
    start_time = time.time()

    # 如果 paths 为 None，则进入模型训练和评估部分。
    if paths is None:
        print(f"{args.model_name} is training......")
        a = len(train_datasets)
        # 对于每个训练数据集（如果使用交叉验证，则会有多个），创建模型实例。
        for i in range(len(train_datasets)):
            train_dataset = train_datasets[i]
            test_dataset = test_datasets[0]

            train_start = time.time()

            model = SplicePred(args.vocab_size, args.embedding_size_DLM1, args.embedding_size_DLM2, args.DLM_seq_len1,
                                   args.DLM_seq_len2, args.filter_num, args.filter_size, args.output_size, args.dropout)
            # 记录模型创建的时间，并将模型信息写入文件。
            model_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            file2 = open(models_file, 'a')
            file2.write(model_time)
            file2.write('\n')
            print(model, file=file2)
            file2.write('\n')
            file2.close()

            # 定义优化器、学习率调度器和损失函数。
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            lr_scheduler = CosineScheduler(10000, base_lr=args.learning_rate, warmup_steps=500)
            criterion = torch.nn.BCEWithLogitsLoss()

            # 创建 DataTrain 实例，用于执行训练过程。
            Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

            # 根据是否使用交叉验证，选择相应的验证或测试数据集进行训练。
            if va_datasets is None:
                Train.train_step(train_dataset, test_dataset, args.model_name, args.epochs,
                                 threshold=args.threshold)
            else:
                test_dataset = va_datasets[i]
                Train.train_step(train_dataset, va_datasets, args.model_name, args.epochs,
                                 threshold=args.threshold)

            # 保存训练好的模型参数到文件。
            PATH = os.getcwd()
            each_model = os.path.join(PATH, 'result', args.model_name + '.pth')
            torch.save(model.state_dict(), each_model)

            # 调用 predict 函数对测试数据集进行预测
            model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)

            # 转换为概率和真实标签数组
            # 处理预测结果（转换为概率）
            if isinstance(model_predictions, torch.Tensor):
                y_pred_prob = torch.sigmoid(model_predictions).numpy()
            else:
                # 假设 model_predictions 是列表或 NumPy 数组
                y_pred_prob = torch.sigmoid(torch.tensor(model_predictions)).numpy()

            # 处理真实标签
            if isinstance(true_labels, torch.Tensor):
                y_test = true_labels.numpy()
            else:
                y_test = true_labels  # 已经是 NumPy 数组，直接使用

            # 生成并保存ROC曲线数据
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
            roc_file_path = f"./result/{args.model_name}_roc_curve.csv"
            roc_df.to_csv(roc_file_path, index=False)
            print(f"ROC曲线数据已保存至: {roc_file_path}")

            # 生成并保存PR曲线数据
            precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_pred_prob)
            pr_df = pd.DataFrame({'Precision': precision_pr, 'Recall': recall_pr})
            pr_file_path = f"./result/{args.model_name}_pr_curve.csv"
            pr_df.to_csv(pr_file_path, index=False)
            print(f"PR曲线数据已保存至: {pr_file_path}")

            # 保存预测分数
            result = pd.DataFrame(model_predictions)  ###输出预测分数
            result.to_csv('./result/test_pred_score.txt', sep='\t', index=False, header=False)

            # 调用 predict 函数对测试数据集进行预测，返回模型的预测结果 model_predictions 和真实标签 true_labels。
            # device 参数指定了模型和数据所在的设备（如 'cuda' 或 'cpu'）。
            #model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)
            # 将模型的预测结果 model_predictions 转换为一个 Pandas DataFrame。
            # 将这个 DataFrame 保存到 ./result/test_pred_score.txt 文件中，使用制表符分隔，不包含索引和列头。
            #result = pd.DataFrame(model_predictions)  ###输出预测分数
            #result.to_csv('./result/test_pred_score.txt', sep='\t', index=False, header=False)
            # 调用 estimate.scores 函数计算性能指标，传入预测结果 model_predictions、真实标签 true_labels 和阈值 args.threshold。

            # 返回的 test_score 是一个包含多个性能指标（如召回率、特异性、精确度等）的列表。
            test_score = estimate.scores(model_predictions, true_labels, args.threshold)

            # 记录训练结束时间
            train_end = time.time()
            # 如果 train_datasets 的长度大于1（即使用了交叉验证），则在模型名称后面加上折叠编号 i，并调用 save_results 函数保存结果。
            if len(train_datasets) > 1:
                save_results(args.model_name + "fold " + str(i), train_start, train_end, test_score, file_path)
            # 否则，直接调用 save_results 函数保存结果，不加折叠编号。
            else:
                save_results(args.model_name, train_start, train_end, test_score, file_path)

            # 打印每个性能指标的值。
            print(f"{args.model_name}, test set:")
            metric = ["Recall", "SPE", "wujianlv", "loujianlv", "Precision", "F1", "MCC", "Acc", "AUC", "AUPR"]
            for k in range(len(metric)):
                print(f"{metric[k]}: {test_score[k]}\n")
            run_time = time.time()
            save_results('average', start_time, run_time, test_score, file_path)

            # 立即保存训练和测试集的特征，不再重新加载数据集
            print("Saving features after training...")

            os.makedirs('./save_feature', exist_ok=True)
            train_fea = feature(model, train_dataset)
            train_fea_df = pd.DataFrame(train_fea.cpu().detach().numpy())
            train_label = pd.read_csv(args.train_label_direction)["Label"]
            train_fea_df = pd.concat([pd.Series(train_label), train_fea_df], axis=1)  ##将样本特征和标签拼接在一起后输出，作为后续分类器的输入
            train_fea_df.to_csv('./save_feature/1-training_feature.txt', sep='\t', index=False, header=False)

            test_fea = feature(model, test_dataset)
            test_fea_df = pd.DataFrame(test_fea.cpu().detach().numpy())
            test_label = pd.read_csv(args.test_label_direction)["Label"]
            test_fea_df = pd.concat([pd.Series(test_label), test_fea_df], axis=1)
            test_fea_df.to_csv('./save_feature/2-testing_feature.txt', sep='\t', index=False, header=False)


# 作为主程序运行，设置模型详情文件路径，获取配置参数 args，并调用 main 函数。
if __name__ == '__main__':
    models_file = f'./result/model_details.txt'
    args = sta_config.get_config()
    main()

