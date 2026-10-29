# -*- coding: utf-8 -*-
"""
make_splits.py：数据划分协议

设计说明
--------
1000³ 岩心配 256³ 窗口，每轴至多 3~8 个窗口位置，不足以在单块岩心内部
切出"训练/验证/测试三段 + 隔离带"。因此外层只切 train/test，
超参数寻优改用训练集内部的分组 K 折交叉验证（experiments.py 实现），即嵌套交叉验证。
这样既无泄漏，也不必为验证集牺牲整块岩心。

三种协议
--------
random ：全局随机划分。在滑窗重叠条件下有泄漏，保留它是为了
         定量展示"精度虚高多少"，不作为结论依据。
blocked：岩心内空间分块。沿 z 方向把窗口按层分成两段，段间丢弃 n_guard 层，
         保证测试窗口与任一训练窗口体积不相交。
         n_guard = ceil(WINDOW/STRIDE) - 1 + EXTRA_GUARD
         回答"能否预测同一岩心未采样区域"。
loco   ：留一岩心交叉验证，整块岩心留出做测试。
         回答"能否预测一块全新岩心"，最严格的泛化检验。

每个协议同时输出 group 列：
random → 随机分组；blocked → z 层号；loco → 岩心编号。

输出：TAB_DIR/splits.csv，列 = [row_id, protocol, fold, split, group]
"""
import os
import math
import numpy as np
import pandas as pd
from config import TAB_DIR, WINDOW, STRIDE, SEED

EXTRA_GUARD = 1        # 在几何不重叠之外再多丢弃的层数，抑制空间自相关
TEST_FRAC = 0.30       # blocked 协议中测试段占 z 层的比例
RNG = np.random.default_rng(SEED)


def _z_layers(df):
    """把 oz 坐标映射为层号 0,1,2,..."""
    uz = np.sort(df["oz"].unique())
    return df["oz"].map({z: i for i, z in enumerate(uz)}), len(uz)


def split_random(df):
    idx = df.index.to_numpy().copy()
    RNG.shuffle(idx)
    n_test = int(len(idx) * 0.2)
    split = pd.Series("train", index=df.index)
    split.loc[idx[:n_test]] = "test"
    group = pd.Series(RNG.integers(0, 5, len(df)), index=df.index)   # 随机分组，供内层CV
    return [("random", 0, split, group)]


def split_random_sub(df, n_train):
    """随机划分，但训练集降采样至 n_train 个，与 blocked 协议的训练规模一致。

    用途：blocked 协议的训练集只占部分 z 层，样本数少于随机划分。
    若直接比较两者的 R²，精度差中同时包含"泄漏"与"训练集规模"两个因素。
    本协议控制训练集规模不变，使 random_sub 与 blocked 的差值仅归因于泄漏。
    """
    idx = df.index.to_numpy().copy()
    RNG.shuffle(idx)
    n_test = int(len(idx) * 0.2)
    split = pd.Series("drop", index=df.index)
    split.loc[idx[:n_test]] = "test"
    split.loc[idx[n_test:n_test + n_train]] = "train"
    group = pd.Series(RNG.integers(0, 5, len(df)), index=df.index)
    return [("random_sub", 0, split, group)]


def split_blocked(df):
    layer, n_layer = _z_layers(df)
    n_guard = math.ceil(WINDOW / STRIDE) - 1 + EXTRA_GUARD
    n_test = max(1, int(round(n_layer * TEST_FRAC)))
    n_train = n_layer - n_test - n_guard

    print(f"blocked：每轴 {n_layer} 层，隔离 {n_guard} 层，"
          f"训练 {n_train} 层，测试 {n_test} 层")
    if n_train < 1:
        raise ValueError(
            f"层数不足（{n_layer} 层），无法在隔离 {n_guard} 层后留出训练段。\n"
            f"解决办法：把 config.STRIDE 减小到 128 或 96 以增加层数，"
            f"或放弃 blocked 协议、只用 loco。")
    if n_train < 2:
        print("  警告：训练段仅 1 层，blocked 结果的代表性有限")

    split = pd.Series("drop", index=df.index)
    split[layer < n_train] = "train"
    split[layer >= n_train + n_guard] = "test"
    return [("blocked", 0, split, layer)]


def split_loco(df):
    folds = []
    cores = sorted(df["core_id"].unique())
    for k, held in enumerate(cores):
        split = pd.Series("train", index=df.index)
        split[df["core_id"] == held] = "test"
        folds.append(("loco", k, split, df["core_id"]))
    return folds


def verify_no_overlap(df, sp, protocol, fold):
    """核验训练窗口与测试窗口体积不相交。两窗口相交 <=> 三轴坐标差均 < WINDOW。"""
    b = sp[(sp["protocol"] == protocol) & (sp["fold"] == fold)]
    tr = df.loc[b.loc[b["split"] == "train", "row_id"]]
    te = df.loc[b.loc[b["split"] == "test", "row_id"]]
    bad = 0
    for core in te["core_id"].unique():
        a = tr[tr["core_id"] == core][["ox", "oy", "oz"]].to_numpy()
        c = te[te["core_id"] == core][["ox", "oy", "oz"]].to_numpy()
        if len(a) == 0 or len(c) == 0:
            continue
        for w in c:
            bad += int(np.all(np.abs(a - w) < WINDOW, axis=1).sum())
    return bad


def main():
    df = pd.read_csv(os.path.join(TAB_DIR, "features_all.csv"))
    df = df[df["valid"] == 1].reset_index(drop=True)
    print(f"有效样本 {len(df)}，岩心 {df['core_id'].nunique()} 块")
    print(f"窗口 {WINDOW}，步长 {STRIDE}，单轴重叠率 {max(0,(WINDOW-STRIDE)/WINDOW):.1%}\n")

    # 先算出 blocked 的训练集规模，供 random_sub 对齐
    _, _, bsplit, _ = split_blocked(df)[0]
    n_blocked_train = int((bsplit == "train").sum())
    print(f"\nrandom_sub 协议将训练集降采样至 {n_blocked_train} 个，与 blocked 一致\n")

    records = []
    makers = (split_random,
              lambda d: split_random_sub(d, n_blocked_train),
              split_blocked, split_loco)
    for maker in makers:
        for proto, fold, split, group in maker(df):
            for row_id in df.index:
                records.append(dict(row_id=row_id, protocol=proto, fold=fold,
                                    split=split[row_id], group=group[row_id]))

    sp = pd.DataFrame(records)
    sp = sp[sp["split"] != "drop"]
    out = os.path.join(TAB_DIR, "splits.csv")
    sp.to_csv(out, index=False)

    print("\n各协议样本分布：")
    print(pd.crosstab([sp["protocol"], sp["fold"]], sp["split"]))

    print("\n泄漏自检（训练-测试相交窗口对数，应为 0）：")
    for proto in ["random", "random_sub", "blocked", "loco"]:
        for fold in sorted(sp[sp["protocol"] == proto]["fold"].unique()):
            bad = verify_no_overlap(df, sp, proto, fold)
            tag = "← 预期不为 0，这正是要展示的泄漏" if proto.startswith("random") and bad else ""
            print(f"  {proto} f{fold}: {bad} {tag}")
            if not proto.startswith("random"):
                assert bad == 0, f"{proto} f{fold} 仍存在重叠，请增大 EXTRA_GUARD"

    print(f"\n划分已保存：{out}")


if __name__ == "__main__":
    main()