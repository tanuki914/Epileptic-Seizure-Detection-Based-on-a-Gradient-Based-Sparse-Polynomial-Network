"""
GSDMTN-100Sparse-CHB
仅跑 100 % 梯度稀疏（ratio=1.0）
其余与 ResMTNR_96p4.py 完全一致

新增：
1. 最开始切 15 % 做锁箱测试集，十折 CV 只在剩余 85 % 上调参/早停，
   全部训练结束后，用锁箱测试集跑一遍，得到最终指标。
2. 打印结果时增加一列耗时（每个 ratio 各自训练+测试的总时长）
"""
# --------------------------------------------------
# 1. 环境 & imports
# --------------------------------------------------
import torch
import numpy as np
import scipy.io as sio
import time          # ← 新增计时
from torch import nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from my_functions import Taylor_expan, Initialization, Activate, Activate_grad, Gradient_renewal

if not torch.cuda.is_available():
    print("❌ CUDA not available; exiting.")
    import sys; sys.exit()

# --------------------------------------------------
# 2. 超参（与原脚本一致）
# --------------------------------------------------
TEST_NUM        = 1
NUM_EPOCHS      = 500
LR              = 0.005
K_FOLDS         = 10
ACTI_TYPE       = 2
POWER           = 2
SHIFT_NUM       = 0.25
MAX_FEATURES    = 200
EARLY_PATIENCE  = 50
VALID_INTERVAL  = 30
EARLY_THRESHOLD = 0.0005
DEVICE          = torch.device("cuda")
SMOOTH          = 0.08

# --------------------------------------------------
# 3. Helper wrappers（与原脚本一致）
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
    max_lr, min_lr = LR, LR * 0.0003
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
# 4. DataProcessor（与原脚本一致）
# --------------------------------------------------
class DataProcessor:
    def load_data(self, path):
        print("\n=== Data Loading ===")
        return sio.loadmat(path)['processedData']

    def preprocess_data(self, data):
        from sklearn.ensemble import IsolationForest
        feats, labs = [], []
        for sess in data:
            nf = sess[0]['NodeFeature'][0, 0]
            if nf.ndim == 3:
                trials, d1, d2 = nf.shape
                nf = nf.reshape(trials, d1 * d2).T
            elif nf.ndim == 2:
                trials = nf.shape[0]
                nf = nf.T
            else:
                raise ValueError("Unsupported NodeFeature dims")
            half = trials // 2
            labs.append(np.concatenate([np.ones(half), 2 * np.ones(trials - half)]))
            if trials % 2 == 1:
                nf = np.delete(nf, half, axis=1)
                labs[-1] = np.delete(labs[-1], half)
            feats.append(nf)
        X = np.hstack(feats)
        y = np.concatenate(labs)
        print(f'Raw samples: {X.shape[1]}, features: {X.shape[0]}')

        X = np.nan_to_num(X, nan=np.nanmean(X, axis=1, keepdims=True), posinf=0, neginf=0)
        keep = IsolationForest(contamination=0.01, random_state=42).fit_predict(X.T) == 1
        X, y = X[:, keep], y[keep]
        print(f'After outlier removal: {keep.sum()}/{len(keep)} retained')

        X = X[np.std(X, axis=1) > 1e-5]
        if X.shape[0] > MAX_FEATURES:
            scores, _ = f_classif(X.T, y)
            X = X[np.argsort(scores)[-MAX_FEATURES:]]
        return X, y

# --------------------------------------------------
# ==========================================================
# 4. Main loop（每个 ratio 都独立 hold-out 10 % 做最终评估）
# ==========================================================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    dp = DataProcessor()
    X, y = dp.preprocess_data(dp.load_data('processedData.mat'))

    ratios = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,1.0]
    final_results = []          # 收集每个 ratio 的锁箱指标

    for ratio in ratios:
        t0 = time.time()      # ← 开始计时
        print("\n"+"="*70)
        print(f"Ratio {ratio*100:.0f}%  ——  独立 hold-out 10 % 锁箱测试")
        print("="*70)

        # 1. 一次性划 10 % 锁箱
        X_train, X_lock, y_train, y_lock = train_test_split(
            X.T, y, test_size=0.1, stratify=y, random_state=42)
        X_train, X_lock = X_train.T, X_lock.T          # 恢复 (特征,样本)

        # 2. 在训练集上做 10-fold CV，仅用于“选 epoch / 早停”，不记录指标
        feat_tr = torch.from_numpy(X_train).float().to(DEVICE)
        onehot_tr = OneHotEncoder(sparse_output=False).fit_transform(
            y_train.reshape(-1,1)).T
        exp_size = safe_taylor_expan(feat_tr[:,:1], POWER).shape[0]
        hid_size = feat_tr.shape[0]
        out_size = 2

        # 3. 用 CV 早停得到最佳 epoch 权重（仅 ratio 不同，其余同原脚本）
        best_epoch_w1, best_epoch_w2 = None, None
        best_val_acc = 0.0
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(feat_tr.T.cpu(), y_train),1):
            # ---- 准备当前折数据 ----
            tr_x, val_x = feat_tr[:,tr_idx], feat_tr[:,val_idx]
            mu, std = tr_x.mean(1,keepdim=True), tr_x.std(1,keepdim=True)
            tr_x, val_x = (tr_x-mu)/(std+1e-8), (val_x-mu)/(std+1e-8)

            tr_exp = safe_taylor_expan(tr_x.cpu().numpy(), POWER)
            val_exp= safe_taylor_expan(val_x.cpu().numpy(), POWER)
            tr_lab_oh = torch.from_numpy(onehot_tr[:,tr_idx]).float().to(DEVICE)
            val_lab = torch.from_numpy(y_train[val_idx]).long().to(DEVICE)

            # ---- 初始化权重 ----
            W1 = torch.from_numpy(improved_init((hid_size, exp_size),4)).to(DEVICE)
            W2 = torch.from_numpy(improved_init((out_size, hid_size),2)).to(DEVICE)
            GW1, MW1 = torch.zeros_like(W1), torch.zeros_like(W1)
            GW2, MW2 = torch.zeros_like(W2), torch.zeros_like(W2)

            fold_best_val, no_imp = 0.0, 0
            fold_best_w1, fold_best_w2 = W1.clone(), W2.clone()

            for epoch in range(1, NUM_EPOCHS+1):
                lr = get_lr(epoch)
                # forward
                z1 = W1 @ tr_exp
                h = safe_activate(z1, ACTI_TYPE)[0]
                z2 = W2 @ (h + tr_x)
                out = nn.functional.softmax(z2, dim=0)
                # loss
                smooth_tgt = (1-SMOOTH)*tr_lab_oh + SMOOTH/out_size
                loss = -(smooth_tgt*torch.log(out+1e-8)).sum()/tr_x.shape[1]
                # backward
                d_soft = out - smooth_tgt
                d_W2 = d_soft @ (h + tr_x).T / tr_x.shape[1]
                d_h  = W2.T @ d_soft
                d_g  = safe_activate_grad(z1, ACTI_TYPE)[0]
                d_z1 = d_h * d_g
                d_W1 = d_z1 @ tr_exp.T / tr_x.shape[1]

                # 梯度稀疏
                k_val = int(ratio * W1.numel())
                idx = torch.topk(torch.abs(d_W1).flatten(), k=k_val)[1]
                mask = torch.zeros_like(W1).flatten()
                mask[idx] = 1
                mask = mask.view_as(W1)
                d_W1 *= mask

                W1, GW1, MW1 = safe_gradient_renewal(7, W1, d_W1, GW1, MW1, lr, epoch)
                W2, GW2, MW2 = safe_gradient_renewal(7, W2, d_W2, GW2, MW2, lr, epoch)

                # 验证早停
                if epoch % VALID_INTERVAL ==0 or epoch==NUM_EPOCHS:
                    with torch.no_grad():
                        z1_v = W1 @ val_exp
                        h_v = safe_activate(z1_v, ACTI_TYPE)[0]
                        out_v = nn.functional.softmax(W2 @ (h_v + val_x), dim=0)
                        val_acc = (out_v.argmax(0)+1 == val_lab).float().mean().item()
                    if val_acc > fold_best_val + EARLY_THRESHOLD:
                        fold_best_val, fold_best_w1, fold_best_w2 = val_acc, W1.clone(), W2.clone()
                        no_imp = 0
                    else:
                        no_imp += 1
                        if no_imp >= EARLY_PATIENCE: break
            # ---- 当前折结束 ----
            if fold_best_val > best_val_acc:
                best_val_acc = fold_best_val
                best_epoch_w1, best_epoch_w2 = fold_best_w1, fold_best_w2
        # 4. 用“整 90 % 训练集 + 早停得到的 epoch”直接拿 best 权重在锁箱上测
        mu_lock = feat_tr.mean(1,keepdim=True)
        std_lock= feat_tr.std(1,keepdim=True)
        lock_x = (torch.from_numpy(X_lock).float().to(DEVICE) - mu_lock) / (std_lock+1e-8)
        lock_exp = safe_taylor_expan(lock_x.cpu().numpy(), POWER)
        lock_y = torch.from_numpy(y_lock).long().to(DEVICE)

        with torch.no_grad():
            z1_lock = best_epoch_w1 @ lock_exp
            pred_lock = nn.functional.softmax(
                best_epoch_w2 @ (safe_activate(z1_lock, ACTI_TYPE)[0] + lock_x), dim=0)
            prob_lock = pred_lock[1,:].cpu().numpy()
            pred_lock = pred_lock.argmax(0) + 1
            acc_lock = (pred_lock == lock_y).float().mean().item()

        y_true_lock = lock_y.cpu().numpy()
        y_pred_lock = pred_lock.cpu().numpy()
        rec_lock = recall_score(y_true_lock, y_pred_lock, zero_division=0)
        f1_lock  = f1_score(y_true_lock,  y_pred_lock, zero_division=0)
        try:
            auc_lock = roc_auc_score(y_true_lock, prob_lock)
        except:
            auc_lock = np.nan

        final_results.append({
            'ratio': int(ratio*100),
            'acc': acc_lock*100,
            'rec': rec_lock*100,
            'f1' : f1_lock*100,
            'auc': auc_lock*100,
            'time': time.time() - t0   # ← 记录总耗时
        })

    # ==========================================================
    # 5. 打印结果：每个 ratio 各自的锁箱指标 + 耗时
    # ==========================================================
    print("\n========== FINAL HOLD-OUT TEST ==========")
    print("Pct   Acc(%)  Rec(%)  F1(%)   AUC(%)  Time(s)")
    for r in final_results:
        print(f"{r['ratio']:2d}%  "
              f"{r['acc']:6.2f}  "
              f"{r['rec']:6.2f}  "
              f"{r['f1']:6.2f}  "
              f"{r['auc']:6.2f}  "
              f"{r['time']:7.1f}")