#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)

from config import sta_config
from models.SplicePred import SplicePred


# =========================
# Device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =========================
# 路径配置
# =========================
BASE_DIR = "/data2/yanmengxiang/projects/GSLM-DSM"

MODEL_PATH = os.path.join(BASE_DIR, "result/best.pth")

TEST_LABEL = os.path.join(BASE_DIR, "input/test_noncanonical.csv")
TEST_GPN = os.path.join(BASE_DIR, "input/test_GPNMSA_noncanonical.npy")
TEST_SPLICEBERT = os.path.join(BASE_DIR, "input/test_Splicebert_noncanonical.npy")

SAVE_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 64
THRESHOLD = 0.5


# =========================
# 加载测试数据
# =========================
print("Loading test data...")

x_test1 = torch.from_numpy(np.load(TEST_GPN)).float()
x_test2 = torch.from_numpy(np.load(TEST_SPLICEBERT)).float()

label_df = pd.read_csv(TEST_LABEL)
y_test = torch.tensor(label_df["Label"].values, dtype=torch.float32)

print("x_test1:", x_test1.shape)
print("x_test2:", x_test2.shape)
print("y_test:", y_test.shape)

test_dataset = TensorDataset(x_test1, x_test2, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# 构建模型
# =========================
args = sta_config.get_config()

model = SplicePred(
    args.vocab_size,
    args.embedding_size_DLM1,
    args.embedding_size_DLM2,
    args.DLM_seq_len1,
    args.DLM_seq_len2,
    args.filter_num,
    args.filter_size,
    args.output_size,
    args.dropout
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# =========================
# 推理
# =========================
all_probs = []
all_labels = []
all_features = []

with torch.no_grad():
    for x1, x2, y in test_loader:
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)

        logits, fea = model(x1, x2)  # 明确解包

        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_labels.append(y)
        all_features.append(fea.cpu())


# 拼接结果
y_pred_prob = torch.cat(all_probs).numpy().flatten()
y_true = torch.cat(all_labels).numpy().flatten()
features = torch.cat(all_features).numpy()

y_pred = (y_pred_prob >= THRESHOLD).astype(int)


# =========================
# 计算指标
# =========================
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)
aupr = average_precision_score(y_true, y_pred_prob)
mcc = matthews_corrcoef(y_true, y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
spe = tn / (tn + fp)

print("\n===== Test Results =====")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {spe:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"MCC          : {mcc:.4f}")
print(f"AUC          : {auc:.4f}")
print(f"AUPR         : {aupr:.4f}")


# =========================
# 保存结果
# =========================

# 1️⃣ 保存预测概率
pd.DataFrame(y_pred_prob).to_csv(
    os.path.join(SAVE_DIR, "noncanonical_pred_score.txt"),
    sep="\t",
    index=False,
    header=False
)

# 2️⃣ 保存融合特征
pd.DataFrame(features).to_csv(
    os.path.join(SAVE_DIR, "noncanonical_fused_feature.txt"),
    sep="\t",
    index=False,
    header=False
)

# 3️⃣ ROC
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
pd.DataFrame({"FPR": fpr, "TPR": tpr}).to_csv(
    os.path.join(SAVE_DIR, "noncanonical_roc_curve.csv"),
    index=False
)

# 4️⃣ PR
precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_pred_prob)
pd.DataFrame({"Precision": precision_pr, "Recall": recall_pr}).to_csv(
    os.path.join(SAVE_DIR, "noncanonical_pr_curve.csv"),
    index=False
)

print("\nAll results saved successfully.")
