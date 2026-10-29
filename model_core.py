"""模型、搜索空间、数据组织和评价指标，由 experiments.py 调用。"""
import os
import sys
import json
import time
import warnings
# SVR 触及 max_iter 上限时会告警。这是刻意设置的耗时上限，
# 此类配置在内层验证上得分差，SSA 会自动避开该区域，无需干预。
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from config import (TAB_DIR, OUT_DIR, ISO_FEATURES, TARGETS, SEED, MODELS,
                    PROTOCOLS, LOG_TARGET, SSA_POP, SSA_ITER, N_SEEDS,
                    AXES, DIR_FEATURES, LONG_FORMAT, USE_DIR, USE_PERP,
                    OPTIMIZERS as OPT_NAMES)
from ssa import Space, OPTIMIZERS

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("未安装 xgboost，将跳过 XGB：pip install xgboost")

INNER_FOLDS = 3        # 训练集内部分组 K 折的折数
TUNE_FOLD = 0          # LOCO 协议用于寻优的折号，其最优参数复用于全部折
SVR_MAX_ITER = 200000  # SVR 单次拟合的迭代上限，防止大 C 下不收敛导致耗时失控
MAX_DIR_MISSING = 0.05  # 方向特征缺失超过该比例即整项弃用，否则中位数填补
SVR_TUNE_N = 2000      # 寻优阶段 SVR 的训练子集上限。SVR 复杂度约 O(n²)~O(n³)，
                       # 全量 4500 样本下单次拟合过慢；降采样只影响寻优，
                       # 阶段 2 的最终评估仍使用全部训练样本


# ==================================================== 搜索空间
def build_space(model):
    """搜索空间。连续量用对数尺度，上界放宽到最优值不会贴边。"""
    if model == "DNN":
        # 上界经一轮放宽：首轮 loco 协议下 h2/h3/alpha/learning_rate_init 四项
        # 同时贴上界，blocked 协议下 batch_size 贴上界。放宽后重跑可区分两种情形：
        # 最优值退回区间内部说明原区间设窄；仍取上界则说明是模型的真实倾向
        # （跨岩心任务下主动选择最强正则化，退化为常数预测器）。
        # 第二轮放宽（2026-08）：loco 下 alpha / learning_rate_init /
        # batch_size 三项同时贴界，random_sub 下前两项贴界。
        # alpha 上界 1e1→1e3、下界 1e-6→1e-8；lr 下界 1e-4→1e-5、
        # 上界 3e-1→5e-1；batch_size 上界 512→2048（长表下训练行数
        # 为子样本数×3，2048 仍远小于样本量，不会退化为全批量）。
        return Space([("h1", 16, 512, "int"), ("h2", 16, 512, "int"),
                      ("h3", 8, 256, "int"),
                      ("alpha", 1e-8, 1e3, "logfloat"),
                      ("learning_rate_init", 1e-5, 5e-1, "logfloat"),
                      ("batch_size", 32, 2048, "int")])
    if model == "KNN":
        return Space([("n_neighbors", 1, 50, "int"),
                      ("leaf_size", 10, 100, "int"), ("p", 1, 3, "int"),
                      ("weights", ["uniform", "distance"], None, "cat")])
    if model == "RF":
        # 上界依据首轮观测收紧：RF 最优为 122 棵树、深度 16，
        # 600/25 已留足余量，且最优值不会贴边
        return Space([("n_estimators", 50, 600, "int"),
                      ("max_depth", 3, 25, "int"),
                      ("min_samples_split", 2, 20, "int"),
                      ("min_samples_leaf", 1, 10, "int"),
                      ("max_features", 0.1, 1.0, "float")])
    if model == "SVR":
        # C 上界由 1e4 收窄至 1e3：SMO 在大 C 下收敛极慢，单次拟合可达数十分钟。
        # gamma 上界由 1e1 收窄至 1e0：过大的 gamma 使核矩阵近乎对角，无实用意义。
        # poly 核在大 C 下最慢，已移除；其表达能力可由 rbf 覆盖。
        return Space([("C", 1e-2, 1e3, "logfloat"),
                      ("gamma", 1e-4, 1e0, "logfloat"),
                      ("epsilon", 1e-3, 1.0, "logfloat"),
                      ("kernel", ["rbf", "sigmoid"], None, "cat")])
    if model == "XGB":
        return Space([("n_estimators", 50, 800, "int"),
                      ("max_depth", 2, 10, "int"),
                      ("learning_rate", 1e-3, 0.3, "logfloat"),
                      ("subsample", 0.5, 1.0, "float"),
                      ("colsample_bytree", 0.5, 1.0, "float"),
                      ("reg_alpha", 1e-4, 100.0, "logfloat"),
                      ("reg_lambda", 1e-4, 100.0, "logfloat")])
    raise ValueError(model)


def make_model(name, p, seed, multioutput=True):
    """multioutput=False 用于长表模式：一行=(子样本,方向)，标签是标量，
    KNN/SVR/XGB 不能再套 MultiOutputRegressor。"""
    def wrap(est):
        return MultiOutputRegressor(est) if multioutput else est

    if name == "DNN":
        return MLPRegressor(hidden_layer_sizes=(p["h1"], p["h2"], p["h3"]),
                            alpha=p["alpha"], learning_rate_init=p["learning_rate_init"],
                            batch_size=p["batch_size"], max_iter=400,
                            early_stopping=False, n_iter_no_change=15, random_state=seed)
    if name == "KNN":
        return wrap(KNeighborsRegressor(
            n_neighbors=p["n_neighbors"], leaf_size=p["leaf_size"],
            p=p["p"], weights=p["weights"]))
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"], min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"], random_state=seed, n_jobs=2)
    if name == "SVR":
        # max_iter 必须设上限。sklearn 默认 -1 表示不限迭代，
        # 在大 C 下 SMO 可能长时间不收敛，导致单次拟合耗时不可控。
        return wrap(SVR(
            C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"], kernel=p["kernel"],
            max_iter=SVR_MAX_ITER, cache_size=500))
    if name == "XGB":
        return wrap(XGBRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"], reg_alpha=p["reg_alpha"],
            reg_lambda=p["reg_lambda"], random_state=seed, n_jobs=2, verbosity=0))
    raise ValueError(name)


# 下界即该参数理论最小值的维度：撞下界是合理最优解，不代表区间设窄，
# 因为无法再往下放宽。例如决策树节点至少需 2 个样本才能分裂，
# 叶节点至少含 1 个样本，最近邻至少取 1 个。
NATURAL_FLOOR = {"min_samples_split", "min_samples_leaf", "n_neighbors", "p",
                 # 小批量本就常是最优，且 32 以下没有放宽的意义，
                 # 否则放宽上界后 batch_size 会一直报「触下界」
                 "batch_size"}


def at_boundary(params, space, tol=0.02):
    """最优值贴边界即说明区间设窄，须放宽后重跑。

    两类例外不计入告警：
      1. 取值极少的整数维（区间跨度 < 10），必然贴边
      2. 下界为理论最小值的维度（NATURAL_FLOOR），撞下界时只检查上界
    """
    flags = []
    for name, low, high, kind in space.dims:
        if kind == "cat" or (kind == "int" and (high - low) < 10):
            continue
        v = params[name]
        if kind == "logfloat":
            v, low, high = np.log10(v), np.log10(low), np.log10(high)
        span = high - low
        if span <= 0:
            continue
        hit_low = (v - low) < tol * span and name not in NATURAL_FLOOR
        hit_high = (high - v) < tol * span
        if hit_low or hit_high:
            # 标注贴的是哪一侧：顶上界说明区间设窄，应继续放宽；
            # 触下界（且不属 NATURAL_FLOOR）说明模型倾向于该维度的极小值，
            # 例如 loco 下选最强正则化、最小学习率，往往是在退化为常数预测器。
            flags.append(name + ("_上界" if hit_high else "_下界"))
    return flags


# ==================================================== 数据与管道
class Data:
    """特征矩阵、标签与划分。

    宽表模式（long=False）：一行 = 子样本，标签为 (KX,KY,KZ)，三个方向共用一个
    多输出模型或三个独立模型。全部特征方向无关，模型无从预测各向异性。
    保留作对照。

    长表模式（long=True）：一行 = (子样本, 方向)，标签为该方向的 K，
    三个方向共用一个模型。特征 = 各向同性特征 + 该方向的方向特征
    (+ 两个垂直方向同名特征的均值)。这样做的三点理由：
      1. 训练样本数×3；
      2. 强制三个方向共享同一套结构—渗透率映射，而不是各学各的整体水平；
      3. 方向特征的置换重要性直接等于"各向异性被解释掉多少"。

    不存在新的信息泄漏：splits.csv 是按子样本（row_id）划分的，长表在划分
    之后才展开，同一子样本的三行必然同属训练集或同属测试集。
    """

    def __init__(self, long=LONG_FORMAT, use_dir=USE_DIR, use_perp=USE_PERP,
                 dir_only=None, verbose=True,
                 dir_shuffle=None, dir_isomean=False, shuffle_seed=SEED):
        """dir_only：只启用指定的若干方向特征，供 09_ablation.py 分组消融。

        dir_shuffle / dir_isomean 是两个零假设对照（09_ablation.py 的 A2s / A2p）。
        A1→A2 一次改了两件事：列数 7→13，以及有没有方向信息。只比 A1 与 A2，
        无法排除「增量只是因为多了 6 列、模型容量变大」这一替代解释。

          dir_shuffle="derange"  逐样本把三个轴的方向特征整体轮换一位或两位。
              轮换量在 {1,2} 中取，没有不动点，任何一个方向都拿不到自己的值。
              轮换对同一样本的全部方向特征一致施加，因此这是一次「坐标轴重标定」：
              每列的边缘分布、方向特征彼此之间的联合结构完全不变，
              唯一被破坏的是方向特征与 K 的方向之间的对应关系。
              置换逐样本独立抽取，5615 个样本上已自然平均，
              不必再对 shuffle_seed 做多次重复。
          dir_shuffle="full"     逐样本抽一个完整随机置换（含恒等，1/6 概率保持原样）。
              比 derange 松，会残留约 1/6 的真实方向信号，一般不用。
          dir_isomean=True       每个轴一律取三轴均值。列数与 A2 相同，
              但三个方向的输入向量完全一致，预测的 Kmax/Kmin 恒为 1、
              R2_dev 必为 0，是「列数匹配的零方向」下界参照。

        两个开关都关闭时（默认），Xax 与打补丁前逐位相同。
        """
        df = pd.read_csv(os.path.join(TAB_DIR, "features_all.csv"))
        df = df[df["valid"] == 1].reset_index(drop=True)
        self.long, self.use_dir, self.use_perp = long, use_dir, use_perp

        # 可用的方向特征：三个轴的列必须齐备。少量缺失（如个别样本沿某方向
        # 无贯通路径导致 tau_geo 为 nan）用训练全集中位数填补，缺失过多才整项弃用。
        self.dir_names, self.dir_dropped = [], []
        if long and use_dir:
            for n in (DIR_FEATURES if dir_only is None else dir_only):
                cols = [f"{n}_{a}" for a in AXES]
                if not set(cols) <= set(df.columns):
                    raise ValueError(f"Requested feature is absent: {cols}")
                self.dir_names.append(n)
            if self.dir_dropped and verbose:
                print(f"  以下方向特征未启用：{self.dir_dropped}")
                print(f"  （运行 directional_core.py 生成并 --merge 并入）")

        self.df = df
        self.splits = pd.read_csv(os.path.join(TAB_DIR, "splits.csv"))
        self.Yraw = df[TARGETS].to_numpy(float)
        self.Y = np.log10(np.clip(self.Yraw, 1e-6, None)) if LOG_TARGET else self.Yraw

        iso = df[ISO_FEATURES].to_numpy(float)
        if not long:
            self.X = iso
            self.Xax = None
            self.feat_names = list(ISO_FEATURES)
        else:
            # ---- 取轴规则 srcmap[i, a]：第 i 个样本的第 a 个方向该读哪个源轴 ----
            # normal ：srcmap[:, a] = a，即读自己（默认，与补丁前完全一致）
            # derange：srcmap[i, a] = (a + k_i) % 3，k_i ∈ {1,2}，无不动点
            # full   ：逐样本一个完整随机置换
            n_row = len(df)
            rs = np.random.default_rng(shuffle_seed)
            if dir_shuffle == "derange":
                shift = rs.integers(1, 3, size=n_row)
                srcmap = (np.arange(3)[None, :] + shift[:, None]) % 3
            elif dir_shuffle == "full":
                srcmap = np.argsort(rs.random((n_row, 3)), axis=1)
            elif dir_shuffle in (None, False, "none"):
                srcmap = np.tile(np.arange(3), (n_row, 1))
            else:
                raise ValueError(f"dir_shuffle 只能是 None / 'derange' / 'full'，收到 {dir_shuffle!r}")
            self.srcmap, self.dir_shuffle, self.dir_isomean = srcmap, dir_shuffle, dir_isomean

            # Xax[a] 为第 a 个方向的特征矩阵，行序与 df 一致
            self.Xax, self.feat_names = [], None
            idx = np.arange(n_row)
            for a, ax in enumerate(AXES):
                blocks, names = [iso], list(ISO_FEATURES)
                selfs, perps = [], []
                for n in self.dir_names:
                    V = df[[f"{n}_{b}" for b in AXES]].to_numpy(float)
                    if dir_isomean:
                        s = V.mean(axis=1)
                        p = s
                    else:
                        s = V[idx, srcmap[:, a]]
                        # 「另外两个轴的均值」= (三轴之和 − 自身) / 2。
                        # srcmap 是逐样本的置换（双射），故这一式在三种口径下都成立，
                        # 且 normal 口径下与补丁前的 df[perp].mean(axis=1) 逐位相等。
                        p = (V.sum(axis=1) - s) / 2.0
                    selfs.append(s)
                    perps.append(p)
                for n, s in zip(self.dir_names, selfs):
                    blocks.append(s[:, None])
                    names.append(f"{n}_自身方向")
                if use_perp:
                    for n, p in zip(self.dir_names, perps):
                        blocks.append(p[:, None])
                        names.append(f"{n}_垂直方向均值")
                self.Xax.append(np.hstack(blocks))
                self.feat_names = names
            if verbose and self.dir_names:
                if dir_isomean:
                    print("  [对照档] 方向特征已各向同性化（三轴均值），"
                          "三方向输入相同，R2_dev 应恒为 0")
                elif dir_shuffle:
                    print(f"  [对照档] 方向特征已逐样本轴置换（{dir_shuffle}，"
                          f"seed={shuffle_seed}），列数与分布不变，只断开轴对应")
            self.X = self.Xax[0]        # 供 fit_pipeline 推断维度

        if verbose:
            mode = "长表（一行=子样本×方向）" if long else "宽表（一行=子样本）"
            print(f"有效样本 {len(df)}，{mode}，特征 {len(self.feat_names)} 个")
            if long:
                print(f"  各向同性 {len(ISO_FEATURES)} 项；方向特征 "
                      f"{len(self.dir_names)} 项 {self.dir_names}"
                      + ("（含垂直方向均值）" if use_perp and self.dir_names else ""))

    def stack(self, rows):
        """长表模式下把子样本下标展开为 (3n, d) 特征与 (3n,) 标签。"""
        X = np.vstack([self.Xax[a][rows] for a in range(3)])
        y = np.concatenate([self.Y[rows, a] for a in range(3)])
        return X, y

    def fold(self, protocol, fold):
        sp = self.splits[(self.splits.protocol == protocol) & (self.splits.fold == fold)]
        tr = sp.loc[sp.split == "train", "row_id"].to_numpy()
        te = sp.loc[sp.split == "test", "row_id"].to_numpy()
        grp = self.df.iloc[tr]["core_id"].to_numpy()  # keep overlapping windows in one inner fold
        return tr, te, grp

    def folds_of(self, protocol):
        return sorted(self.splits[self.splits.protocol == protocol].fold.unique())


def fit_pipeline(X, y):
    """所有预处理只在训练数据上拟合。"""
    if np.isinf(X).any() or not np.isfinite(y).all():
        raise ValueError("Non-finite inputs/targets")
    if np.isnan(X).all(axis=0).any():
        raise ValueError("A feature is entirely missing in this training fold")
    imputer = SimpleImputer(strategy="median").fit(X)
    sx = StandardScaler().fit(imputer.transform(X))
    sy = StandardScaler().fit(y)

    def tx(A):
        return sx.transform(imputer.transform(A))
    return tx, sy


def evaluate_aniso(y_true, y_pred, eps=1e-6):
    """各向异性的预测精度。逐样本在 lg 尺度上把三向渗透率分解为

        lg K_ia = L_i + d_ia ,   L_i = mean_a lg K_ia ,  sum_a d_ia = 0

    L 是"这个子样本整体有多渗"，d 是"方向之间怎么分配"。原有的 R²(K_a) 几乎
    完全由 L 决定，即使各向异性一点没学到也能很高，故必须把两者拆开报告。

    返回：
      R2_level     整体水平 L 的 R²
      R2_dev       方向偏差 d 的 R²（各向异性的核心指标）
      R2_aniso_lg  各向异性比 lg(Kmax/Kmin) 的 R²
      MAE_aniso    Kmax/Kmin 本身的平均绝对误差（倍数，便于直观理解）
      hit_rate     最高渗方向判对的样本比例（随机猜为 1/3）
      mean_aniso_true / mean_aniso_pred  实测与预测的 Kmax/Kmin 均值
    """
    a = np.log10(np.clip(y_true, eps, None))
    p = np.log10(np.clip(y_pred, eps, None))
    La, Lp = a.mean(axis=1), p.mean(axis=1)
    da, dp = a - La[:, None], p - Lp[:, None]
    ra = a.max(axis=1) - a.min(axis=1)          # lg(Kmax/Kmin)
    rp = p.max(axis=1) - p.min(axis=1)
    return dict(
        target="_aniso",
        R2_level=float(r2_score(La, Lp)),
        R2_dev=float(r2_score(da.ravel(), dp.ravel())),
        R2_aniso_lg=float(r2_score(ra, rp)),
        RMSE_aniso_lg=float(np.sqrt(mean_squared_error(ra, rp))),
        MAE_aniso=float(np.mean(np.abs(10 ** ra - 10 ** rp))),
        hit_rate=float(np.mean(a.argmax(axis=1) == p.argmax(axis=1))),
        mean_aniso_true=float(np.mean(10 ** ra)),
        mean_aniso_pred=float(np.mean(10 ** rp)))


def evaluate(y_true, y_pred):
    """同时在原始 mD 尺度与 log10 尺度上计算指标。

    渗透率跨多个数量级时，原始尺度的 R² 由高渗样本主导，少数样本的偏差即可
    造成很大的负值；而模型是在 log10 空间训练的，该尺度下的指标与优化目标
    一致，且对各数量级的样本赋予相近权重。两个尺度并列输出，不做选择性报告。
    std_implied 用于一致性自检：固定测试集上 σ(y)=RMSE/√(1−R²) 应为常数。
    """
    rows = []
    eps = 1e-6
    for j, t in enumerate(TARGETS):
        a, p = y_true[:, j], y_pred[:, j]
        r2 = float(r2_score(a, p))
        rmse = float(np.sqrt(mean_squared_error(a, p)))
        la = np.log10(np.clip(a, eps, None))
        lp = np.log10(np.clip(p, eps, None))
        r2l = float(r2_score(la, lp))
        rmsel = float(np.sqrt(mean_squared_error(la, lp)))
        rows.append(dict(
            target=t, R2=r2, RMSE=rmse,
            MAE=float(np.mean(np.abs(a - p))),
            std_implied=rmse / np.sqrt(max(1 - r2, 1e-12)),
            std_actual=float(np.std(a)),
            R2_log=r2l, RMSE_log=rmsel,
            MAE_log=float(np.mean(np.abs(la - lp))),
            std_implied_log=rmsel / np.sqrt(max(1 - r2l, 1e-12)),
            std_actual_log=float(np.std(la))))
    rows.append(evaluate_aniso(y_true, y_pred))
    return rows


def fit_eval(D, tr, te, model_name, params, seed, return_pred=False):
    if D.long:
        Xtr, ytr = D.stack(tr)
        ytr = ytr[:, None]
        tx, sy = fit_pipeline(Xtr, ytr)
        m = make_model(model_name, params, seed, multioutput=False)
        m.fit(tx(Xtr), sy.transform(ytr).ravel())
        t0 = time.perf_counter()
        # 三个方向依次推理后拼回 (n_test, 3)，与宽表输出形状一致
        pred = np.column_stack([
            sy.inverse_transform(
                m.predict(tx(D.Xax[a][te])).reshape(-1, 1)).ravel()
            for a in range(3)])
        infer = (time.perf_counter() - t0) / max(len(te), 1)
    else:
        tx, sy = fit_pipeline(D.X[tr], D.Y[tr])
        m = make_model(model_name, params, seed)
        m.fit(tx(D.X[tr]), sy.transform(D.Y[tr]))
        t0 = time.perf_counter()
        pred = sy.inverse_transform(m.predict(tx(D.X[te])))
        infer = (time.perf_counter() - t0) / max(len(te), 1)
    if LOG_TARGET:
        pred = 10 ** pred
    res = evaluate(D.Yraw[te], pred)
    return (res, infer, pred) if return_pred else (res, infer)


