# -*- coding: utf-8 -*-
"""
directional_core.py —— 方向相关的孔隙结构特征

背景
----
原有 7 项特征（分形维数、孔隙度、最大/平均孔隙直径、最大/平均喉道直径、
平均配位数）全部是方向无关的标量，无法刻画孔喉网络的空间定向性，因而
三个方向只能靠三个独立模型去拟合各自的整体水平，各向异性本身无从预测。
本脚本补充随坐标轴置换而置换的方向特征。

三条设计约束
------------
1. 必须随坐标轴置换而置换。"各向异性程度"这类标量对 KX/KY/KZ 是同一个值，
   模型照样分不出方向，不可用。
2. 计算成本必须远低于标签。若特征本身需要解一次 Laplace 或 Stokes 方程，
   用特征代替流动模拟就省不了时间。
   ⚠ 因此不使用 porespy 的 dns.tortuosity——它解扩散方程，成本与 Stokes 同量级。
3. 不能是流动解的函数。用流动求解得到的迂曲度去预测渗透率属循环论证。
   本脚本全部特征均为纯几何/拓扑量。

两组特征与其代价
----------------
图像域（只需分割图像）：
  phi_min       垂直于该轴的全部切片中孔隙度的最小值        亚秒级
  phi_slice_cv  该轴切片孔隙度的变异系数                    亚秒级
  chord_mean    沿该轴的孔隙相平均弦长（MIL 张量对角元）    亚秒级
  corr_len      沿该轴的两点相关长度（FFT 自相关）          亚秒级
  perc_frac     沿该轴贯通的孔隙团占总体积的比例            亚秒级
  tau_geo       沿该轴的测地迂曲度（快速行进法）            本组最贵，见下
网络域（需重跑 snow2 提取；但不必重解三方向 Stokes）：
  k_bundle      Σ(r⁴/l)·cos²θ 的并联束估计，量纲即渗透率
  orient_M      以 d⁴/L 加权的喉道取向张量对角元（三轴和为 1）
  throat_dir    以 |u_a|·d⁴ 加权的喉道直径调和平均
  coord_dir     每孔沿该轴 ±45° 锥内的喉道数均值

关于迂曲度的取法
----------------
迂曲度必须走测地距离/快速行进这条路，不能用 porespy 的 tortuosity_fd。
后者给每个孔隙体素建一个节点再解 Laplace 方程：256³ 窗口、20% 孔隙度即约
330 万个未知数，三个方向解三次。这比整条 PNM 流水线还贵——原 86.8 s 中的
大头是 snow2 网络提取，而在 2365 孔 / 3909 喉的网络上解三方向 Stokes 只需
亚秒级。用 tortuosity_fd 就省不了时间。
本脚本用 skimage.graph.MCP_Geometric（Dijkstra 型快速行进），复杂度 O(N log N)，
且是纯几何量，不解任何方程，不构成循环论证。

用法
----
  python directional_core.py                 只算图像域特征（推荐先跑）
  python directional_core.py --no-tortuosity 图像域但跳过 tau_geo（最快）
  python directional_core.py --with-network  同时重跑 snow2 算网络域特征

实测代价（256³ 子样本，单样本）
------------------------------
  除 tau_geo 外的图像域特征   合计 < 1 s
  tau_geo（三方向）           约 16 s   ← 本组唯一较贵的一项
  网络域（含 snow2 重提取）   约 30 s
参照：原 PNM 流水线单样本 86.8 s，其中大头是 snow2 提取。
若先只要最便宜的一档，用 --no-tortuosity，5615 个样本 1 小时内可跑完。
  python directional_core.py --merge         并入 features_all.csv
  python directional_core.py --check-axes    核对轴映射（见下）

轴映射核对
----------
子样本按数组第 0、1、2 轴切分，对应 KX、KY、KZ（见 make_index.py）；若切分顺序改动，
轴的物理含义会互换。方向特征一旦与 KX/KY/KZ 错位，全部是噪声。
--check-axes 输出 phi_min_a 与 K_b 的 3×3 相关矩阵，对角元应显著强于非对角元
（phi_min 越小越堵，故相关应为正且对角最大）。这一步不通过就不要往下做。

输出：<TAB_DIR>/dir_features.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd

from config import (RAW_DIR, TAB_DIR, VOXEL_SIZE, M2_TO_MD, AXES,
                    DIR_FEATURES_IMAGE, DIR_FEATURES_NET, TARGETS)

CKPT = os.path.join(TAB_DIR, "dir_features.csv")
COS45 = np.sqrt(0.5)


# ================================================== 图像域
def phi_slice_stats(pore, axis):
    """垂直于 axis 的逐切片孔隙度，返回 (最小值, 变异系数)。

    最小截面孔隙度对应的机制：垂向流动必须穿越层界，
    有效渗透率受最堵的那个截面控制。
    """
    other = tuple(i for i in range(3) if i != axis)
    phi = pore.mean(axis=other)
    mu = float(phi.mean())
    return float(phi.min()), float(phi.std() / mu) if mu > 0 else np.nan


def mean_chord_length(pore, axis):
    """沿 axis 的孔隙相平均弦长（体素）。

    平均弦长 = 孔隙体素总数 / 孔隙游程个数。用差分定位游程起点，无需循环。
    注意：贴边的游程被截断，会略微低估，各样本口径一致，不影响横向比较。
    """
    p = np.moveaxis(pore, axis, -1)
    p = p.reshape(-1, p.shape[-1])
    z = np.zeros((p.shape[0], 1), dtype=np.int8)
    d = np.diff(np.concatenate([z, p.astype(np.int8), z], axis=1), axis=1)
    n_run = int((d == 1).sum())
    n_vox = int(p.sum())
    return n_vox / n_run if n_run else np.nan


def correlation_lengths(pore):
    """三个方向的两点相关长度（体素），一次 FFT 同时得到。

    S2(r) 由自相关给出：S2(r) = <I(x)I(x+r)>。沿各轴取剖面，
    定义相关长度为 S2 首次降到 φ² + (φ−φ²)/e 所对应的 r（线性插值）。
    采用周期自相关（FFT），在相关长度远小于窗口边长时该近似成立。
    """
    f = pore.astype(np.float32)
    phi = float(f.mean())
    F = np.fft.rfftn(f)
    S = np.fft.irfftn(F * np.conj(F), s=f.shape).real / f.size
    thr = phi ** 2 + (phi - phi ** 2) / np.e
    out = []
    for axis in range(3):
        idx = [0, 0, 0]
        prof = np.empty(f.shape[axis], dtype=float)
        for r in range(f.shape[axis]):
            idx[axis] = r
            prof[r] = S[tuple(idx)]
        below = np.where(prof <= thr)[0]
        if len(below) == 0:
            out.append(float(f.shape[axis]))       # 未衰减到阈值，取窗口边长封顶
            continue
        r1 = int(below[0])
        if r1 == 0:
            out.append(0.0)
            continue
        y0, y1 = prof[r1 - 1], prof[r1]
        frac = (y0 - thr) / (y0 - y1) if y0 != y1 else 0.0
        out.append(float(r1 - 1 + frac))
    return out


def percolating_fraction(pore, axis):
    """沿 axis 贯通的孔隙团占子样本总体积的比例。

    与方向无关的 isolated_frac 不同：这里度量的是"有孔隙但过不去"，
    即同时触及该轴两个端面的连通团。一次连通组分标记即可，O(N)。
    """
    from scipy import ndimage
    lab, n = ndimage.label(pore)
    if n == 0:
        return 0.0
    lo = np.moveaxis(lab, axis, 0)[0]
    hi = np.moveaxis(lab, axis, 0)[-1]
    spanning = np.intersect1d(np.unique(lo[lo > 0]), np.unique(hi[hi > 0]))
    if spanning.size == 0:
        return 0.0
    return float(np.isin(lab, spanning).sum() / lab.size)


def geodesic_tortuosity(pore, axis):
    """沿 axis 的测地迂曲度 τ = <测地路径长> / <直线距离>。

    从 axis=0 端面的全部孔隙体素出发做多源快速行进，取到达 axis=L-1 端面的
    孔隙体素的测地距离均值，除以 L-1。无贯通路径时返回 nan。
    MCP_Geometric 在 26 邻域下按真实几何步长累计，不是 chamfer 近似。
    """
    try:
        from skimage.graph import MCP_Geometric
    except ImportError:
        return np.nan
    p = np.moveaxis(pore, axis, 0)
    L = p.shape[0]
    costs = np.where(p, 1.0, np.inf)
    starts = np.argwhere(p[0])
    if len(starts) == 0:
        return np.nan
    starts = np.column_stack([np.zeros(len(starts), int), starts])
    mcp = MCP_Geometric(costs, fully_connected=True)
    dist, _ = mcp.find_costs(starts.tolist())
    end = dist[L - 1][p[L - 1]]
    end = end[np.isfinite(end)]
    if end.size == 0:
        return np.nan
    return float(end.mean() / (L - 1))


def image_features(pore, with_tortuosity=True):
    """返回 {特征名_轴: 值}，长度为 4×3。"""
    rec = {}
    corr = correlation_lengths(pore)
    for a, ax in enumerate(AXES):
        pmin, pcv = phi_slice_stats(pore, a)
        rec[f"phi_min_{ax}"] = pmin
        rec[f"phi_slice_cv_{ax}"] = pcv
        rec[f"chord_mean_{ax}"] = mean_chord_length(pore, a) * VOXEL_SIZE * 1e6   # µm
        rec[f"corr_len_{ax}"] = corr[a] * VOXEL_SIZE * 1e6                        # µm
        rec[f"perc_frac_{ax}"] = percolating_fraction(pore, a)
        rec[f"tau_geo_{ax}"] = geodesic_tortuosity(pore, a) if with_tortuosity \
            else np.nan
    return rec


# ================================================== 网络域
def network_features(coords, conns, td, side_m):
    """由孔隙网络算方向特征。side_m 为子样本边长（米）。

    coords (n_pore,3) 米，conns (n_throat,2)，td (n_throat,) 喉道等效直径（米）。
    喉道单位方向向量 u = (c_j − c_i)/‖·‖，长度 l = ‖c_j − c_i‖，半径 r = d/2。

      k_bundle_a  = (π/8)·Σ (r⁴/l)·u_a² · (L/A)，L=side_m，A=side_m²
                    量纲为 m²，即渗透率。把每条喉道按与坐标轴的夹角投影后
                    并联求和，是一个纯几何的方向渗透率估计（并联意义上的上界，
                    未计串联瓶颈与迂曲）。这是本组特征中物理含义最直接的一项。
                    ⚠ 采用"各喉道压降正比于其轴向投影"的均匀梯度闭合；换用别的
                    闭合，l 的幂次会变（严格均匀梯度下 l 会消掉）。它是代理量，
                    不是推导所得的渗透率。
      orient_M_a  = Σ w u_a² / Σ w，w = d⁴/l    沿 a 轴的传导份额，三轴和为 1
                    与 k_bundle 的区别在于归一化：k_bundle 带量级，orient_M 只留
                    取向。两者都保留——前者预测整体水平，后者解释各向异性。
      throat_dir_a= Σ w₂ / Σ (w₂/d)，w₂ = |u_a|·d⁴   方向调和平均喉道直径
      coord_dir_a = 每孔 |u_a| > cos45° 的喉道数均值
    调和平均而非算术平均，是因为串联通道的总传导受最窄环节控制，
    """
    rec = {}
    n_pore = len(coords)
    if len(conns) == 0 or n_pore == 0:
        for ax in AXES:
            for n in DIR_FEATURES_NET:
                rec[f"{n}_{ax}"] = np.nan
        return rec

    v = coords[conns[:, 1]] - coords[conns[:, 0]]
    L = np.linalg.norm(v, axis=1)
    ok = L > 0
    v, L, td_ = v[ok], L[ok], np.asarray(td)[ok]
    u = np.abs(v / L[:, None])                 # 取绝对值：方向无正负之分
    d4 = td_ ** 4
    w = d4 / L

    r4_over_l = (td_ / 2.0) ** 4 / L
    for a, ax in enumerate(AXES):
        k_m2 = (np.pi / 8.0) * float(np.sum(r4_over_l * u[:, a] ** 2)) / side_m
        rec[f"k_bundle_{ax}"] = k_m2 * M2_TO_MD
        rec[f"orient_M_{ax}"] = float(np.sum(w * u[:, a] ** 2) / np.sum(w))
        w2 = u[:, a] * d4
        s = np.sum(w2)
        rec[f"throat_dir_{ax}"] = float(s / np.sum(w2 / td_)) * 1e6 if s > 0 else np.nan
        near = u[:, a] > COS45
        rec[f"coord_dir_{ax}"] = float(2.0 * near.sum() / n_pore)   # 每条喉道计入两个孔
    return rec


def _patch_numpy_reshape():
    """给 numpy < 2.1 补上 np.reshape 的 shape= 关键字。

    np.reshape 的 shape= 是 numpy 2.1 引入的，之前叫 newshape。
    porespy 3.0.4 的 _snow2.py 第 261 行写的是
        L = np.reshape(L, shape=boundary_width.shape)
    在 numpy < 2.1 上直接抛 TypeError，而 02h 把异常吞成 NaN，
    表现为"算了一整天全是 NaN"。

    这里只补这一个关键字，不动 numpy 版本，因为同一环境里的
    sklearn / xgboost / torch / pandas 以及已有的全部建模结果
    都建立在当前版本上。

    幂等；numpy >= 2.1 上检测到原生支持即直接返回，不做任何改动。
    02h 用 ProcessPoolExecutor 起子进程，每个子进程都要单独打一次，
    故本函数在 extract_network 内部调用而非模块顶部。
    """
    if getattr(np, "_reshape_shape_kw_shim", False):
        return False
    try:
        np.reshape(np.zeros(2), shape=(2,))
        return False                      # 原生支持，无需补
    except TypeError:
        pass

    _orig = np.reshape

    def reshape(a, newshape=None, order="C", *, shape=None, copy=None, **kw):
        if newshape is None:
            newshape = shape
        if newshape is None:
            raise TypeError("reshape() 缺少 shape/newshape 参数")
        return _orig(a, newshape, order=order)

    np.reshape = reshape
    np._reshape_shape_kw_shim = True
    return True


def _pick_key(net, cands, what):
    """按候选键名取值，支持带相名前缀的变体。

    snow2 的输出键名随 porespy 版本变化（throat.equivalent_diameter /
    throat.inscribed_diameter 都出现过），传字典相时还会加相名前缀。
    硬编码单个键名会在换版本后抛 KeyError；若调用方把异常吞掉，
    就变成"算了一整天全是 NaN"。故这里逐个候选尝试，
    全部失败时抛出带完整键名清单的异常，让问题当场暴露。
    """
    for c in cands:
        if c in net:
            return np.asarray(net[c])
    tails = [c.split(".")[-1] for c in cands]
    for k in net.keys():
        if any(k.endswith(t) for t in tails):
            return np.asarray(net[k])
    raise KeyError(
        f"snow2 输出中找不到{what}。候选键名 {cands} 均不存在。\n"
        f"实际键名：{sorted(net.keys())}\n"
        f"请检查 snow2 输出的键名，并把正确的键名补进候选表。")


def extract_network(pore):
    """重跑 snow2 提取网络，只取方向特征所需的三样东西。

    返回 (coords 米, conns, td 米)。键名按候选表自动匹配，见 _pick_key。
    """
    _patch_numpy_reshape()
    import dask
    dask.config.set(scheduler="threads")
    import porespy as ps
    net = ps.networks.snow2(phases=pore, voxel_size=VOXEL_SIZE).network

    coords = _pick_key(net, ["pore.coords", "pore.center", "pore.centroid"],
                       "孔隙坐标")
    conns = _pick_key(net, ["throat.conns"], "喉道连接")
    try:
        td = _pick_key(net, ["throat.equivalent_diameter",
                             "throat.inscribed_diameter",
                             "throat.diameter"], "喉道等效直径")
    except KeyError:
        # 退路：由喉道横截面积换算等效圆直径 d = 2√(A/π)，
        # 与 equivalent_diameter 的定义一致。
        area = _pick_key(net, ["throat.cross_sectional_area", "throat.area"],
                         "喉道横截面积（用于换算等效直径）")
        td = 2.0 * np.sqrt(np.asarray(area, float) / np.pi)
    return np.asarray(coords), np.asarray(conns), np.asarray(td, float)


# ================================================== 主流程
def main(with_network=False, with_tortuosity=True):
    import h5py
    index = pd.read_csv(os.path.join(TAB_DIR, "subsample_index.csv"))
    done = set()
    if os.path.exists(CKPT):
        prev = pd.read_csv(CKPT)
        prev = prev[prev["core_id"] != "core_id"]
        done = set(zip(prev["core_id"], prev["ox"], prev["oy"], prev["oz"]))
        print(f"发现检查点，已完成 {len(done)} 个样本，将跳过")

    rows, t_img, t_net, n = [], 0.0, 0.0, 0
    for i, r in index.iterrows():
        if (r["core_id"], r["ox"], r["oy"], r["oz"]) in done:
            continue
        with h5py.File(os.path.join(RAW_DIR, r["file"]), "r") as f:
            im = f["data"][:]
        pore = (im == 0)

        t0 = time.time()
        rec = dict(core_id=r["core_id"], ox=r["ox"], oy=r["oy"], oz=r["oz"])
        rec.update(image_features(pore, with_tortuosity=with_tortuosity))
        t_img += time.time() - t0

        if with_network:
            t0 = time.time()
            try:
                rec.update(network_features(*extract_network(pore),
                                            side_m=pore.shape[0] * VOXEL_SIZE))
            except Exception as e:
                print(f"  {r['file']} 网络提取失败：{type(e).__name__}: {e}")
            t_net += time.time() - t0

        rows.append(rec)
        n += 1
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(CKPT, mode="a", index=False,
                                      header=not os.path.exists(CKPT),
                                      encoding="utf-8-sig", float_format="%.6g")
            rows = []
            print(f"  [{i+1}/{len(index)}] 图像域累计 {t_img:.0f}s"
                  + (f"，网络域累计 {t_net:.0f}s" if with_network else ""))

    if rows:
        pd.DataFrame(rows).to_csv(CKPT, mode="a", index=False,
                                  header=not os.path.exists(CKPT),
                                  encoding="utf-8-sig", float_format="%.6g")
    if n:
        print(f"\n新增 {n} 个样本；单样本图像域 {t_img/n:.2f} s"
              + (f"，网络域 {t_net/n:.1f} s" if with_network else ""))
    print(f"已保存：{CKPT}")
    print("下一步：python directional_core.py --merge")


def merge_into_features():
    """把方向特征并入 features_all.csv（按 core_id/ox/oy/oz 左连接）。"""
    fp = os.path.join(TAB_DIR, "features_all.csv")
    df = pd.read_csv(fp)
    dd = pd.read_csv(CKPT)
    dd = dd[dd["core_id"] != "core_id"].drop_duplicates(
        ["core_id", "ox", "oy", "oz"], keep="first")
    key = ["core_id", "ox", "oy", "oz"]
    for c in dd.columns:
        if c not in key:
            dd[c] = pd.to_numeric(dd[c], errors="coerce")
    new = [c for c in dd.columns if c not in key]
    df = df.drop(columns=[c for c in new if c in df.columns])
    out = df.merge(dd, on=key, how="left")
    miss = out[new[0]].isna().sum() if new else 0
    print(f"并入 {len(new)} 列方向特征；{miss} 个样本无对应方向特征（将为 NaN）")
    if miss:
        print("  这些样本在长表模式下会被剔除，请先补齐 02g 的计算")
    out.to_csv(fp, index=False, encoding="utf-8-sig", float_format="%.6g")
    print(f"已写回：{fp}")


def check_axes():
    """核对方向特征的轴与 KX/KY/KZ 是否对齐。"""
    fp = os.path.join(TAB_DIR, "features_all.csv")
    df = pd.read_csv(fp)
    df = df[df["valid"] == 1]
    for name in DIR_FEATURES_IMAGE + DIR_FEATURES_NET:
        cols = [f"{name}_{a}" for a in AXES]
        if not set(cols) <= set(df.columns):
            continue
        lk = np.log10(df[TARGETS].clip(lower=1e-6))
        M = np.array([[df[c].corr(lk[t]) for t in TARGETS] for c in cols])
        print(f"\n{name}：与 lg K 的相关矩阵（行=特征轴，列=渗透率方向）")
        print(pd.DataFrame(M, index=AXES, columns=TARGETS).round(3).to_string())
        diag = np.nanmean(np.diag(M))
        off = np.nanmean(M[~np.eye(3, dtype=bool)])
        flag = "✓ 对角占优" if abs(diag) > abs(off) + 0.02 else "✗ 对角不占优"
        print(f"  原始尺度   对角均值 {diag:+.3f}  非对角均值 {off:+.3f}  {flag}")

        # ---- 去均值口径 ----
        # 原始尺度上三个轴共享一个很大的整体量级成分（chord_mean 三轴极差
        # 只有均值的 12%），对角与非对角的差被淹在小数点后第三位，
        # 既不能证明轴对齐、也不能证明没对齐。逐样本把三轴中心化之后，
        # 每行之和为 0，剩下的纯粹是方向偏差分量：若轴映射正确，
        # 对角元应同时是所在行与所在列的最大者，非对角约为对角的 −1/2。
        F = df[cols].to_numpy(float)
        K = lk.to_numpy(float)
        Fc = F - np.nanmean(F, axis=1, keepdims=True)
        contrast = float(np.nanmean(np.nanmax(F, axis=1) - np.nanmin(F, axis=1)))
        scale = float(np.nanmean(np.abs(F))) or 1.0
        # 三轴几乎恒等的特征（如 perc_frac），中心化后只剩 1e-16 量级的浮点残差，
        # 相关系数在数值上仍算得出来，但那是纯噪声，必须先挡掉再打印。
        resid = float(np.nanstd(Fc))
        if not np.isfinite(resid) or resid <= 1e-9 * scale:
            print(f"  去均值后 该特征三轴数值几乎恒等（三轴极差仅为均值的 "
                  f"{contrast/scale:.2%}），中心化后只剩浮点残差，相关系数无意义。")
            print(f"  ⚠ 该项完全没有方向对比度：它在原始尺度上的高相关全部来自整体水平，"
                  f"不应计入「方向特征」对各向异性的贡献")
            continue
        Fc = pd.DataFrame(Fc, columns=cols)
        Kc = pd.DataFrame(K - K.mean(axis=1, keepdims=True), columns=TARGETS)
        Mc = np.array([[Fc[c].corr(Kc[t]) for t in TARGETS] for c in cols])
        print("  去均值后的相关矩阵（逐样本三轴中心化）")
        print(pd.DataFrame(Mc, index=AXES, columns=TARGETS).round(3).to_string())
        dc = np.nanmean(np.diag(Mc))
        oc = np.nanmean(Mc[~np.eye(3, dtype=bool)])
        if (np.isfinite(Mc).all() and
                all(np.argmax(np.abs(Mc[i])) == i for i in range(3)) and
                all(np.argmax(np.abs(Mc[:, j])) == j for j in range(3))):
            fc = "✓ 对角为各行各列最大，轴映射正确"
        else:
            fc = "✗ 对角不是行列最大 —— 轴可能错位，或该项本身无方向信号"
        print(f"  去均值后 对角均值 {dc:+.3f}  非对角均值 {oc:+.3f}  {fc}")
        if contrast / scale < 0.02:
            print(f"  ⚠ 三轴极差仅为均值的 {contrast/scale:.1%}，方向对比度很弱，"
                  f"去均值后的相关即使显著也很不稳健")
    print("\n说明：只要有若干项方向特征稳定呈对角占优，即可确认 ox/oy/oz 与 "
          "KX/KY/KZ 的轴对应正确；若普遍不占优，请核对 make_index.py 中"
          "编号与坐标的对应关系。")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--merge" in args:
        merge_into_features()
    elif "--check-axes" in args:
        check_axes()
    else:
        main(with_network=("--with-network" in args),
             with_tortuosity=("--no-tortuosity" not in args))