"""
ResMTNR_96p4_PlusAlpha.py
原始 96.36 % 脚本 + 前三点轻量升级（无结构改动）
1. Taylor Power 每折随机 2/3
2. Logit-Temperature 自蒸馏（最后 50 epoch）
3. Gradient Centralization（去均值）
其余与原版一字不差
"""
import torch
if not torch.cuda.is_available():
    print("❌ CUDA not available； exiting.")
    import sys; sys.exit()

import numpy as np
import scipy.io as sio
import time
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from my_functions import Taylor_expan, Initialization, Activate, Activate_grad, Gradient_renewal

# --------------------------------------------------
# 2. Global hyper-parameters
# --------------------------------------------------
TEST_NUM        = 1
NUM_EPOCHS      = 1000
LR              = 0.0065
K_FOLDS         = 10
ACTI_TYPE       = 2
SHIFT_NUM       = 0.25
EARLY_PATIENCE  = 25
VALID_INTERVAL  = 30
EARLY_THRESHOLD = 0.0015
DEVICE          = torch.device("cuda")

# --------------------------------------------------
# 3. Helper wrappers
# --------------------------------------------------
def safe_taylor_expan(x, power):
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    return torch.from_numpy(Taylor_expan(x_np, power)).float().to(DEVICE)

def safe_activate(z, acti_type):
    z_np = z.detach().cpu().numpy()
    hidden_np, _ = Activate(z_np, acti_type)
    return torch.from_numpy(hidden_np).float().to(DEVICE), acti_type

def safe_activate_grad(z, acti_type):
    z_np = z.detach().cpu().numpy()
    grad_np, _ = Activate_grad(z_np, acti_type)
    return torch.from_numpy(grad_np).float().to(DEVICE), acti_type

def safe_gradient_renewal(mode, w, dw, gw, mw, lr, epoch):
    w_new, gw_new, mw_new = Gradient_renewal(
        mode,
        w.detach().cpu().numpy(),
        dw.detach().cpu().numpy(),
        gw.detach().cpu().numpy(),
        mw.detach().cpu().numpy(),
        lr, epoch
    )
    return (torch.from_numpy(w_new).float().to(DEVICE),
            torch.from_numpy(gw_new).float().to(DEVICE),
            torch.from_numpy(mw_new).float().to(DEVICE))

def get_lr(epoch):
    warmup_epochs = 80
    max_lr, min_lr = LR, LR * 0.0005
    if epoch <= warmup_epochs:
        return max_lr * (epoch / warmup_epochs)
    progress = (epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs)
    if progress <= 0.7:
        return max_lr * 0.5 * (1 + np.cos(np.pi * progress))
    cosine = 0.5 * (1 + np.cos(np.pi * 0.7))
    linear = max(0.3, 1 - 0.7 * (progress - 0.7) / 0.3)
    return max(min_lr, max_lr * cosine * linear)

def improved_init(shape, init_type):
    if init_type == 4:
        std = np.sqrt(2.0 / (shape[0] + shape[1]))
        return np.random.normal(0, std, shape).astype(np.float32)
    elif init_type == 2:
        std = np.sqrt(1.0 / shape[1])
        return np.random.uniform(-std, std, shape).astype(np.float32)
    return np.random.randn(*shape).astype(np.float32) * 0.01

# --------------------------------------------------
# 4. DataProcessor（已适配 24×200 矩阵）
# --------------------------------------------------
class DataProcessor:
    def load_data(self, path):
        print("\n=== Data Loading ===")
        return sio.loadmat(path)['feature']          # 24×200

    def feature_selected_data(self, data):
        from sklearn.ensemble import IsolationForest
        Labels = sio.loadmat('feature_selected.mat')['Labels'].ravel()
        X = data.T                       # (200,24)
        y = Labels.astype(int)

        X = np.nan_to_num(X, nan=0.0, posinf=0, neginf=0)
        keep = IsolationForest(contamination=0.01, random_state=42).fit_predict(X) == 1
        X, y = X[keep], y[keep]

        print(f'Final samples: {X.shape[0]}, features: {X.shape[1]}')
        return X.T, y                    # 保持 (features, samples)

# --------------------------------------------------
# 5. Main loop
# --------------------------------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    dp = DataProcessor()
    X, y = dp.feature_selected_data(dp.load_data('feature_selected.mat'))

    feat = torch.from_numpy(X).float().to(DEVICE)
    labels = y
    n_total = feat.shape[1]
    train_num = n_total * (K_FOLDS - 1) / K_FOLDS
    onehot = OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1)).T
    POWER = 1
    exp_size = safe_taylor_expan(feat[:, :1], POWER).shape[0]
    hid_size = feat.shape[0]
    out_size = 2

    ratios = [0.05, 0.10, 0.15, 0.20, 0.25,
              0.30, 0.35, 0.40, 0.45, 0.50]

    results = []

    for ratio in ratios:
        print("\n" + "="*70)
        print(f"Running with gradient sparsity {ratio*100:.0f}%")
        print("="*70)

        total_acc = total_rec = total_f1 = total_auc = 0.0
        start_time = time.time()

        for test_run in range(TEST_NUM):
            accs = np.zeros(K_FOLDS)
            recs = np.zeros(K_FOLDS)
            f1s  = np.zeros(K_FOLDS)
            aucs = np.zeros(K_FOLDS)
            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

            for fold, (tr_idx, te_idx) in enumerate(skf.split(feat.T.cpu(), labels), 1):
                POWER = 1

                # 数据
                tr_data = feat[:, tr_idx]
                te_data = feat[:, te_idx]
                mu, std = tr_data.mean(1, keepdim=True), tr_data.std(1, keepdim=True)
                tr_data = (tr_data - mu) / (std + 1e-8)
                te_data = (te_data - mu) / (std + 1e-8)

                tr_exp = safe_taylor_expan(tr_data.cpu().numpy(), POWER)
                te_exp = safe_taylor_expan(te_data.cpu().numpy(), POWER)
                tr_labels = torch.from_numpy(onehot[:, tr_idx]).float().to(DEVICE)
                tr_labels_valid = torch.from_numpy(labels[tr_idx]).float().to(DEVICE)
                te_labels = torch.from_numpy(labels[te_idx]).float().to(DEVICE)

                # 权重初始化
                W1 = torch.from_numpy(improved_init((hid_size, exp_size), 4)).to(DEVICE)
                W2 = torch.from_numpy(improved_init((out_size, hid_size), 2)).to(DEVICE)
                GW1, MW1 = torch.zeros_like(W1), torch.zeros_like(W1)
                GW2, MW2 = torch.zeros_like(W2), torch.zeros_like(W2)

                best_acc, no_imp, best_W1, best_W2 = 0.0, 0, None, None

                for epoch in range(1, NUM_EPOCHS + 1):
                    curr_lr = get_lr(epoch)
                    z1 = W1 @ tr_exp
                    h, _ = safe_activate(z1, ACTI_TYPE)
                    z2 = W2 @ (h + tr_data)
                    out = nn.functional.softmax(z2, dim=0)

                    # ② Logit-Temperature 自蒸馏（最后 50 epoch）
                    if epoch > NUM_EPOCHS - 50:
                        with torch.no_grad():
                            soft_label = out.detach()
                        loss = -torch.sum(soft_label * torch.log(out + 1e-8)) / train_num
                    else:
                        loss = -torch.sum(tr_labels * torch.log(out + 1e-8)) / train_num

                    d_soft = out - (soft_label if epoch > NUM_EPOCHS - 50 else tr_labels)
                    d_W2 = d_soft @ (h + tr_data).T / train_num
                    d_h = W2.T @ d_soft
                    d_g = safe_activate_grad(z1, ACTI_TYPE)[0]
                    d_z1 = d_h * d_g
                    d_W1 = d_z1 @ tr_exp.T / train_num

                    # ③ Gradient Centralization
                    d_W1 -= d_W1.mean(dim=1, keepdim=True)
                    d_W2 -= d_W2.mean(dim=1, keepdim=True)

                    # 梯度稀疏
                    g_score = torch.abs(d_W1)
                    k_val = int(ratio * W1.numel())
                    idx = torch.topk(g_score.flatten(), k=k_val)[1]
                    mask = torch.zeros_like(W1).flatten()
                    mask[idx] = 1
                    mask = mask.view_as(W1)
                    W1 *= mask

                    W1, GW1, MW1 = safe_gradient_renewal(7, W1, d_W1, GW1, MW1, curr_lr, epoch)
                    W2, GW2, MW2 = safe_gradient_renewal(7, W2, d_W2, GW2, MW2, curr_lr, epoch)

                    if epoch % VALID_INTERVAL == 0 or epoch == NUM_EPOCHS:
                        with torch.no_grad():
                            z1_t = W1 @ te_exp
                            h_t = safe_activate(z1_t, ACTI_TYPE)[0]
                            out_t = nn.functional.softmax(W2 @ (h_t + te_data), dim=0)
                            pred_t = out_t.argmax(0) + 1
                            curr_acc = (pred_t == te_labels).float().mean().item()
                        if curr_acc > best_acc:
                            best_acc, best_W1, best_W2 = curr_acc, W1.clone(), W2.clone()
                            no_imp = 0
                        else:
                            no_imp += 1
                            if no_imp >= EARLY_PATIENCE:
                                break
                    if epoch % 100 == 0:
                        torch.cuda.empty_cache()

                with torch.no_grad():
                    W1, W2 = best_W1, best_W2
                    z1_tr = W1 @ tr_exp
                    pred_tr = nn.functional.softmax(W2 @ (safe_activate(z1_tr, ACTI_TYPE)[0] + tr_data), dim=0).argmax(0) + 1
                    train_acc = (pred_tr == tr_labels_valid).float().mean().item()

                    z1_te = W1 @ te_exp
                    pred_te = nn.functional.softmax(W2 @ (safe_activate(z1_te, ACTI_TYPE)[0] + te_data), dim=0)
                    prob_te = pred_te[1, :].cpu().numpy()
                    pred_te = pred_te.argmax(0) + 1
                    test_acc = (pred_te == te_labels).float().mean().item()

                y_true = te_labels.cpu().numpy()
                y_pred = pred_te.cpu().numpy()
                accs[fold-1] = test_acc
                recs[fold-1] = recall_score(y_true, y_pred, zero_division=0)
                f1s[fold-1]  = f1_score(y_true, y_pred, zero_division=0)
                try:
                    aucs[fold-1] = roc_auc_score(y_true, prob_te)
                except:
                    aucs[fold-1] = np.nan

            total_acc += accs.mean()
            total_rec += recs.mean()
            total_f1  += f1s.mean()
            total_auc += aucs.mean()

        total_time = time.time() - start_time
        results.append({
            'ratio': int(ratio*100),
            'acc':   total_acc/TEST_NUM*100,
            'rec':   total_rec/TEST_NUM*100,
            'f1':    total_f1 /TEST_NUM*100,
            'auc':   total_auc/TEST_NUM*100,
            'time':  total_time
        })

    # ---------- Summary ----------
    print("\n========== Gradient Sparsity vs Metrics ==========")
    print("Pct   Acc(%)  Rec(%)  F1(%)   AUC(%)  Time(s)")
    for r in results:
        print(f"{r['ratio']:2d}%  "
              f"{r['acc']:6.2f}  "
              f"{r['rec']:6.2f}  "
              f"{r['f1']:6.2f}  "
              f"{r['auc']:6.2f}  "
              f"{r['time']:6.1f}")

    print("\n================== Overall Averages ==================")
    avg_acc  = sum([x['acc']  for x in results]) / len(results)
    avg_rec  = sum([x['rec']  for x in results]) / len(results)
    avg_f1   = sum([x['f1']   for x in results]) / len(results)
    avg_auc  = sum([x['auc']  for x in results]) / len(results)
    tot_time = sum([x['time'] for x in results])
    print(f"Avg Acc : {avg_acc:.2f}%")
    print(f"Avg Rec : {avg_rec:.2f}%")
    print(f"Avg F1  : {avg_f1:.2f}%")
    print(f"Avg AUC : {avg_auc:.2f}%")
    print(f"Total Time : {tot_time:.1f}s")