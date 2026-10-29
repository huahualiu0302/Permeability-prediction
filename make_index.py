# -*- coding: utf-8 -*-
"""
make_index.py：为子样本文件生成岩心编号与窗口坐标索引 data/subsample_index.csv

子样本目录结构为 <RAW_DIR>/<岩心名>/0.hdf5 … 511.hdf5，共 11 个岩心，由 make_subsamples.py 生成。
512 = 8³，对应 1000³ 体素、256³ 窗口、100 体素步长的三重嵌套滑窗，每轴起点为 0、100、…、700。
第 n 个子样本的起点为 (ox, oy, oz) = (n//64, n%64//8, n%8) × 100，数组第 0、1、2 轴对应 KX、KY、KZ 的方向。
"""
import os
import numpy as np
import pandas as pd
from config import RAW_DIR, TAB_DIR, CORES, FULL_SIZE, WINDOW, STRIDE

N_PER_AXIS = (FULL_SIZE - WINDOW) // STRIDE + 1        # = 8


def verify_mapping(feature_csv_dir=None):
    """可选：用已有的分岩心特征表复核 lag=1/8/64 指纹是否成立。"""
    if not feature_csv_dir:
        return
    cols = ["fractal_dimension", "Porosity", "pore_mean", "throat_mean",
            "mean_coordination", "KX", "KY", "KZ"]
    step = N_PER_AXIS
    print("映射复核（lag=1, %d, %d 处应明显低于相邻 lag）：" % (step, step ** 2))
    for fn in sorted(os.listdir(feature_csv_dir)):
        if not fn.endswith(".csv"):
            continue
        d = pd.read_csv(os.path.join(feature_csv_dir, fn))
        if not set(cols) <= set(d.columns) or len(d) != N_PER_AXIS ** 3:
            continue
        Z = ((d[cols] - d[cols].mean()) / d[cols].std()).to_numpy()
        rng = np.random.default_rng(0)
        p = rng.integers(0, len(Z), (500, 2))
        base = np.linalg.norm(Z[p[:, 0]] - Z[p[:, 1]], axis=1).mean()
        s = []
        for lag in [1, 2, 4, step, 2 * step, step ** 2, 2 * step ** 2]:
            a = np.arange(0, len(Z) - lag)
            s.append(f"lag{lag}={np.linalg.norm(Z[a]-Z[a+lag],axis=1).mean()/base:.2f}")
        print(f"  {fn:<24s} " + "  ".join(s))


def main():
    origins = list(range(0, FULL_SIZE - WINDOW + 1, STRIDE))
    assert len(origins) == N_PER_AXIS, "步长与窗口不匹配，请检查 config"
    print(f"每轴 {N_PER_AXIS} 个窗口位置，单岩心 {N_PER_AXIS**3} 个子样本")
    print(f"单轴重叠率 {(WINDOW-STRIDE)/WINDOW:.1%}\n")

    records = []
    for core in CORES:
        folder = os.path.join(RAW_DIR, core)
        if not os.path.isdir(folder):
            print(f"  [跳过] 未找到目录 {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".hdf5")]
        n_expect = N_PER_AXIS ** 3
        if len(files) != n_expect:
            print(f"  [警告] {core} 有 {len(files)} 个文件，预期 {n_expect}，映射可能不成立")
        for f in files:
            n = int(os.path.splitext(f)[0])          # 文件名即索引
            i, j, k = n // (N_PER_AXIS ** 2), (n % (N_PER_AXIS ** 2)) // N_PER_AXIS, n % N_PER_AXIS
            records.append(dict(core_id=core, idx=n,
                                ox=origins[i], oy=origins[j], oz=origins[k],
                                file=os.path.join(core, f)))
        print(f"  {core}: {len(files)} 个子样本")

    df = pd.DataFrame(records).sort_values(["core_id", "idx"]).reset_index(drop=True)
    out = os.path.join(TAB_DIR, "subsample_index.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n索引已保存：{out}，共 {len(df)} 个子样本，{df.core_id.nunique()} 块岩心")
    print("下一步：运行 pnm_core.py 计算常规参数与渗透率")


if __name__ == "__main__":
    main()
    # verify_mapping(TAB_DIR)     # 需要时取消注释，用已有分岩心 CSV 复核映射