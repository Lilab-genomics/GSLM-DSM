#!/usr/bin/env python
# coding=gbk

import time
import torch
import math
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
import os
import shutil

# import hiddenlayer as hl

# import tensorboard

class DataTrain:
    # 训练模型
    # model: 要训练的模型。
    # optimizer: 优化器，如Adam、SGD等。
    # criterion: 损失函数，如BCEWithLogitsLoss。
    # scheduler: 可选的学习率调度器，默认为None。
    # device: 设备，如'cuda'或'cpu'。
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda"):
        self.model = model.to(device)  # 将模型移动到指定设备
        self.optimizer = optimizer  # 保存优化器
        self.criterion = criterion  # 保存损失函数
        self.lr_scheduler = scheduler  # 保存学习率调度器
        self.device = device  # 保存设备信息

    # train_iter: 训练数据迭代器。
    # test_iter: 测试数据迭代器。
    # model_name: 模型名称。
    # epochs: 训练轮数。
    # threshold: 分类阈值，默认为0.5。
    def train_step(self, train_iter, test_iter, model_name, epochs=None, threshold=0.5):
        steps = 1  # 初始化步数
        train_fea = []  # 存储训练特征
        best_loss = 100000.  # 初始化最佳损失
        best_loss_acc = 0.  # 初始化最佳准确率
        bestlos_epoch = 0  # 最佳损失对应的epoch
        PATH = os.getcwd()  # 获取当前工作目录
        best_model = os.path.join(PATH, 'result', 'best.pth')  # 设置最佳模型保存路径
        early_stop = 10  # 早停机制的轮数
        #        history1 = hl.History()
        #         canvas1 = hl.Canvas()
        print_step = 100  # 打印步数间隔

        for epoch in range(1, epochs + 1):  # 遍历每个epoch
            start_time = time.time()  # 记录开始时间
            total_loss = 0  # 初始化总损失
            alpha = 0.4  # 权重参数
            i = 0  # 初始化索引
            for train_data1, train_data2, train_label in train_iter:  # 加载批量数据
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_data1, train_data2, train_label = train_data1.to(self.device), train_data2.to(self.device), train_label.to(self.device)
                # 模型预测
                y_hat, train_feature = self.model(train_data1, train_data2)
                # 计算损失
                loss = self.criterion(y_hat, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # 使用PyTorch内置的学习率调度器
                        self.lr_scheduler.step()
                    else:
                        # 使用自定义的学习率调度器
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()  # 累加损失
                steps += 1  # 增加步数

            # 完成一次迭代训练
            end_time = time.time()  # 记录结束时间
            epoch_time = end_time - start_time  # 计算epoch用时

            # 在训练集上进行预测
            model_predictions, true_labels = predict(self.model, train_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # 根据阈值分类
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc1 = accuracy_score(true_labels, y_hat)  # 计算准确率

            print(f'{model_name}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')  # 打印epoch信息
            print(f'Train loss:{total_loss / len(train_iter)}')  # 打印平均损失
            print(f'Train acc:{acc1}')  # 打印准确率

            train_loss = total_loss / len(train_iter)  # 计算平均损失
            if train_loss < best_loss:  # 如果当前损失小于最佳损失
                torch.save(self.model.state_dict(), best_model)  # 保存最佳模型
                best_loss = train_loss  # 更新最佳损失
                best_loss_acc = acc1  # 更新最佳准确率
                bestlos_epoch = epoch  # 更新最佳损失对应的epoch

            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):  # 早停机制
                break  # 如果连续early_stop个epoch损失没有改善，则停止训练

        # 加载最佳模型
        self.model.load_state_dict(torch.load(best_model))
        print("best_loss = " + str(best_loss))  # 打印最佳损失
        print("best_loss_acc = " + str(best_loss_acc))  # 打印最佳准确率

        # canvas1.save('./save_img/train_test' + model_name + '.pdf')


# model_predictions, true_labels
def predict(model, data, device="cuda"):
    # 模型预测
    model.to(device)  # 将模型移动到指定设备
    model.eval()  # 进入评估模式
    predictions = []  # 存储预测结果
    labels = []  # 存储真实标签

    with torch.no_grad():  # 取消梯度反向传播
        for x, x2, y in data:  # 根据实际数据格式进行调整
            x = x.to(device)  # 将数据移动到指定设备
            x2 = x2.to(device)
            y = y.to(device).unsqueeze(1)  # 真实标签


            score, _ = model(x, x2)  # 调用模型进行预测
            label = torch.sigmoid(score)  # 使用sigmoid函数将分数转换为概率
            predictions.extend(label.tolist())  # 将预测结果添加到列表
            labels.extend(y.tolist())  # 将真实标签添加到列表

    return np.array(predictions), np.array(labels)  # 返回预测结果和真实标签


def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    创建一个学习率调度器，在warmup阶段线性增加学习率，之后线性减少。

    参数:
        optimizer_ (torch.optim.Optimizer): 优化器实例。
        num_warmup_steps (int): warmup阶段的步数。
        num_training_steps (int): 总训练步数。
        last_epoch (int, optional): 上一个epoch的索引。默认为-1。
    """

    # 定义学习率计算函数
    def lr_lambda(current_step):
        # 在warmup阶段，学习率从0线性增加到初始学习率
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 在warmup阶段之后，学习率从初始学习率线性减少到最终学习率
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    # 返回LambdaLR调度器，该调度器会根据lr_lambda函数来调整学习率
    return LambdaLR(optimizer_, lr_lambda, last_epoch)


class CosineScheduler:
    """
    退化学习率调度器，支持warmup阶段。

    参数:
        max_update (int): 最大更新次数（即总训练步数）。
        base_lr (float, optional): 初始学习率。默认为0.01。
        final_lr (float, optional): 最终学习率。默认为0。
        warmup_steps (int, optional): warmup阶段的步数。默认为0。
        warmup_begin_lr (float, optional): warmup开始时的学习率。默认为0。
    """

    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr  # 保存初始学习率
        self.max_update = max_update  # 保存最大更新次数
        self.final_lr = final_lr  # 保存最终学习率
        self.warmup_steps = warmup_steps  # 保存warmup阶段的步数
        self.warmup_begin_lr = warmup_begin_lr  # 保存warmup开始时的学习率
        self.max_steps = self.max_update - self.warmup_steps  # 计算非warmup阶段的最大步数

    def get_warmup_lr(self, epoch):
        """
        计算warmup阶段的学习率。

        参数:
            epoch (int): 当前的epoch。

        返回:
            float: 当前epoch的学习率。
        """
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        """
        根据当前epoch返回学习率。

        参数:
            epoch (int): 当前的epoch。

        返回:
            float: 当前epoch的学习率。
        """
        if epoch < self.warmup_steps:
            # 在warmup阶段，使用get_warmup_lr方法计算学习率
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            # 在warmup阶段之后，使用余弦退火公式计算学习率
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def feature(model, data, device="cuda"):
    # 提取模型编码后的特征
    model.eval()  # 进入评估模式
    extract_fea = []  # 存储提取的特征

    with torch.no_grad():  # 取消梯度反向传播
        for x, x2, y in data:  # 根据实际数据格式进行调整
            x = x.to(device)  # 将数据移动到指定设备
            x2 = x2.to(device)
            _, fea = model(x, x2)  # 模型预测并获取中间特征
            extract_fea.extend(fea.tolist())  # 将特征添加到列表

    return torch.Tensor(extract_fea)  # 返回提取的特征
