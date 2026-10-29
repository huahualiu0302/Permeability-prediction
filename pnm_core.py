# -*- coding: utf-8 -*-
"""
pnm_core.py：常规孔隙结构参数提取与三方向渗透率计算（等效直径）

* 孔隙度为孔隙体素比；渗透率全程用 SI 单位计算，K[m²] 除以 9.869233e-16 得 mD，不含任何调节系数。
* snow2 提取孔隙网络时传入 voxel_size=2.25e-6 m，网络直接以 SI 单位建立。
* 输出表保留岩心编号与子样本起点坐标（core_id、ox、oy、oz），供空间分块和留一岩心划分使用。
* 记录孤立孔隙占比、求解收敛状态和进出口流量不平衡度，按物理准则标记无效子样本并写明理由。
* 记录单个子样本各步骤的耗时。

本文件得到的 KX、KY、KZ 按等效直径计算传导率；作为预测目标的内切直径渗透率由 pnm_inscribed.py 在同一网络上重算。
平均孔径、最大孔径、平均喉径、最大喉径取等效直径的统计值。
"""
__version__ = "2026-08-04-r6"   # 版本戳

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import h5py
# ---- 必须在 import porespy 之前执行 ----
# porespy 3.x 的 snow2 默认走并行分支（parallel_kw={}），内部使用 dask。
# dask 选择调度器时会尝试 import distributed，该链路加载 tornado 并读取
# Windows 证书库；证书库中若有畸形证书会抛 ssl.SSLError: ASN1 NOT_ENOUGH_DATA。
# 显式指定本地调度器即可跳过对 distributed 的探测。
# 另一种做法：pip uninstall -y distributed
import dask
dask.config.set(scheduler="threads")

import porespy as ps
from porespy_compat import compatible_snow2
snow2 = compatible_snow2(ps.networks.snow2)
import openpnm as op
from sklearn.linear_model import LinearRegression

from config import (RAW_DIR, TAB_DIR, CORES, VOXEL_SIZE, FULL_SIZE, WINDOW,
                    STRIDE, MU, P_IN, P_OUT, M2_TO_MD)

# ==================== 水力形状因子模型 ====================
# 选用 cones_and_cylinders（孔隙为收敛-发散锥体、喉道为圆柱），依据：
#   1. 锥柱型是孔隙网络模型的经典几何假设，文献基础最厚；
#   2. 不受"喉道直径须小于所连孔隙"的约束——该比例逐样本记录于
#      throat_gt_pore_frac 列；
#      spheres_and_cylinders 在该比例大于 0 时会直接抛异常，无法用于全部样本；
#   3. 相比 cubes_and_cuboids 的方截面假设（截面积取 d² 而非圆形 πd²/4），
#      对真实孔喉形态的近似更合理。
# 注意：选型依据是物理合理性与文献惯例，不是"哪个更接近实验室实测值"。
# 各模型的量级差异（Kirby 子样本均值）：
#   cubes_and_cuboids 2993 mD | pyramids_and_cuboids 1659 mD | cones_and_cylinders 1071 mD
SIZE_FACTOR_MODEL = "cones_and_cylinders"

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- 分形维数
def fractal_dimension_box_counting(pore_mask: np.ndarray):
    """三维盒计数法。返回 (分形维数, 拟合R²)。

    同时返回拟合优度，便于检查拟合质量。
    """
    box_sizes = np.array([2, 4, 8, 16, 32, 64])
    voxels = np.column_stack(np.where(pore_mask))
    if len(voxels) == 0:
        return np.nan, np.nan

    Ns = []
    for size in box_sizes:
        min_boxes = np.inf
        for offset in np.linspace(0, size, 3, dtype=int):
            bins = [np.arange(0, dim + size, size) - offset for dim in pore_mask.shape]
            H, _ = np.histogramdd(voxels, bins=bins)
            min_boxes = min(min_boxes, np.sum(H > 0))
        Ns.append(min_boxes)

    X = np.log(1.0 / box_sizes).reshape(-1, 1)
    y = np.log(np.asarray(Ns, dtype=float))
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    return float(model.coef_[0]), float(r2)


# ---------------------------------------------------------------- 单方向渗透率
def solve_direction(pn, phase, axis: str, shape, voxel_size):
    """沿指定轴求解 Stokes 流动并按达西定律换算绝对渗透率。

    返回 (K_mD, 相对流量不平衡度)。全程 SI 单位，无任何调节系数。
    """
    lo, hi = pn.pores(f"{axis}min"), pn.pores(f"{axis}max")
    if len(lo) == 0 or len(hi) == 0:
        return np.nan, np.nan

    alg = op.algorithms.StokesFlow(network=pn, phase=phase)
    alg.set_value_BC(pores=lo, values=P_OUT, mode="overwrite")
    alg.set_value_BC(pores=hi, values=P_IN, mode="overwrite")
    # OpenPNM 3.6.2 run() reads Workspace.default_solver when no solver is passed.
    # The historical settings['solver']='cg' was not used by run(). Make the
    # actual installed default explicit; record it in the result below.
    solver_name = op.Workspace().settings.default_solver
    alg.run(solver=getattr(op.solvers, solver_name)())

    # OpenPNM 3.x 的 rate() 可能返回逐孔流量数组，必须求和取总流量。
    q_in = float(np.abs(np.sum(alg.rate(pores=lo, mode="group"))))    # m³/s，网络已是 SI 单位
    q_out = float(np.abs(np.sum(alg.rate(pores=hi, mode="group"))))
    imbalance = abs(q_in - q_out) / max(q_in, 1e-30)   # 质量守恒自检

    idx = {"x": 0, "y": 1, "z": 2}[axis]
    L = shape[idx] * voxel_size                                    # m
    A = np.prod([shape[i] for i in range(3) if i != idx]) * voxel_size ** 2   # m²
    dP = P_IN - P_OUT

    K_m2 = q_in * L * MU / (A * dP)
    return K_m2 * M2_TO_MD, imbalance


# ---------------------------------------------------------------- 主流程
def process_one(path, core_id, ox, oy, oz, im=None, audit_dir=None):
    """path 为 hdf5 路径；也可直接传入 im（内存中的分割图像数组），
    此时忽略 path。后者用于不落盘子样本的场景，
    两条路径共用完全相同的提取逻辑，保证结果一致。"""
    t0 = time.time()
    if im is None:
        with h5py.File(path, "r") as f:
            im = f["data"][:]

    pore = (im == 0)                       # 孔隙相为 True
    porosity = float(np.mean(pore))        # 直接体素比，不做任何平移
    t_input = time.time()

    fd, fd_r2 = fractal_dimension_box_counting(pore)
    t_box = time.time()

    # ---- 孔隙网络提取：务必传 voxel_size，网络直接以 SI 单位建立 ----
    net = snow2(phases=pore, voxel_size=VOXEL_SIZE)
    t_snow = time.time()

    pd_arr = np.asarray(net["network"]["pore.equivalent_diameter"])    # m
    td_arr = np.asarray(net["network"]["throat.equivalent_diameter"])  # m
    if len(pd_arr) == 0 or len(td_arr) == 0:
        return None

    conns = np.asarray(net["network"]["throat.conns"])
    # 喉道直径大于所连孔隙直径的比例：SNOW 等效直径的已知特性
    throat_gt_pore = float(np.mean(
        td_arr > np.minimum(pd_arr[conns[:, 0]], pd_arr[conns[:, 1]])))
    n_pore = net["network"]["pore.coords"].shape[0]
    coord = np.zeros(n_pore, dtype=int)
    np.add.at(coord, conns[:, 0], 1)
    np.add.at(coord, conns[:, 1], 1)
    mean_coord = float(coord.mean()) if n_pore else 0.0
    t_stats = time.time()

    # ---- 网络健康检查：孤立孔占比作为质量控制指标 ----
    proj = op.Project()
    pn = op.network.Network(project=proj)
    pn.update(net.network)
    health = op.utils.check_network_health(network=pn)
    disconnected = sorted(set(list(health["isolated_pores"]) + list(health["disconnected_pores"])))
    iso_frac = len(disconnected) / max(n_pore, 1)
    if disconnected:
        op.topotools.trim(network=pn, pores=disconnected)

    if pn.Np < 10:                        # 修剪后网络过小，无法求解
        return dict(core_id=core_id, ox=ox, oy=oy, oz=oz, valid=0,
                    drop_reason="连通孔隙数不足10", Porosity=porosity)

    water = op.phase.Water(network=pn)
    water["pore.viscosity"] = MU
    water["throat.viscosity"] = MU
    pn["throat.hydraulic_size_factors"] = getattr(
        op.models.geometry.hydraulic_size_factors, SIZE_FACTOR_MODEL)(
        network=pn,
        pore_diameter="pore.equivalent_diameter",
        throat_diameter="throat.equivalent_diameter",
    )
    water["throat.hydraulic_conductance"] = op.models.physics.hydraulic_conductance.generic_hydraulic(
        phase=water, pore_viscosity="pore.viscosity",
        throat_viscosity="throat.viscosity", size_factors="throat.hydraulic_size_factors",
    )

    shape = im.shape
    t_setup = time.time()
    kx, ib_x = solve_direction(pn, water, "x", shape, VOXEL_SIZE)
    ky, ib_y = solve_direction(pn, water, "y", shape, VOXEL_SIZE)
    kz, ib_z = solve_direction(pn, water, "z", shape, VOXEL_SIZE)
    t_flow = time.time()
    if audit_dir is not None:
        os.makedirs(audit_dir, exist_ok=True)
        geometry = {k: np.asarray(v) for k, v in net.network.items()
                    if k in ('pore.equivalent_diameter', 'pore.inscribed_diameter',
                             'throat.equivalent_diameter', 'throat.inscribed_diameter',
                             'pore.coords', 'throat.conns', 'pore.boundary')}
        np.savez_compressed(os.path.join(audit_dir, 'geometry.npz'), **geometry)

    # ---- 物理准则筛样本（取代事后对全部变量做 IQR 一刀切）----
    # 判据只保留"物理上无法计算"的情形。
    # 曾设有"孤立孔隙占比>30%"一条，已移除：孤立孔在求解前已被 trim，
    # 不参与渗流，其占比高低不影响渗透率计算的正确性；孤立孔多是致密砂岩的
    # 真实属性，按此剔除会系统性删去低渗端样本（Parker 曾因此被剔 27.5%）。
    # 该指标改为记录于 isolated_frac 列，可作为输入特征。
    valid, reason = 1, ""
    if np.any(np.isnan([kx, ky, kz])):
        valid, reason = 0, "存在方向无贯通路径"
    elif np.nanmax([ib_x, ib_y, ib_z]) > 0.01:
        valid, reason = 0, "进出口流量不平衡>1%，求解未收敛"
    elif porosity <= 0:
        valid, reason = 0, "孔隙度非正"

    return dict(
        core_id=core_id, ox=ox, oy=oy, oz=oz,
        fractal_dimension=fd, fractal_fit_r2=fd_r2,
        Porosity=porosity,
        pore_max=float(pd_arr.max()) * 1e6,       # 统一以 µm 记录，便于制表
        pore_mean=float(pd_arr.mean()) * 1e6,
        throat_max=float(td_arr.max()) * 1e6,
        throat_mean=float(td_arr.mean()) * 1e6,
        mean_coordination=mean_coord,
        KX=kx, KY=ky, KZ=kz,
        isolated_frac=iso_frac, throat_gt_pore_frac=throat_gt_pore,
        size_factor_model=SIZE_FACTOR_MODEL,
        solver_name=op.Workspace().settings.default_solver,
        imbalance_max=float(np.nanmax([ib_x, ib_y, ib_z])),
        n_pores=int(pn.Np), n_throats=int(pn.Nt),
        pnm_time_sec=round(t_flow - t0, 6),
        input_read_mask_sec=t_input-t0,
        boxcount_sec=t_box-t_input,
        network_extract_sec=t_snow-t_box,
        geometry_statistics_sec=t_stats-t_snow,
        network_setup_sec=t_setup-t_stats,
        flow_solve_sec=t_flow-t_setup,
        valid=valid, drop_reason=reason,
    )


# ---------------------------------------------------------------- 合并检查点
# 按岩心开多进程时，每块岩心写自己的
# features_ckpt_<岩心>.csv，全部跑完后用 --merge 合成一张 features_all.csv。
MERGE_KEY = ["core_id", "ox", "oy", "oz"]
MERGE_NUM = ["Porosity", "fractal_dimension", "pore_max", "pore_mean",
             "throat_max", "throat_mean", "mean_coordination", "KX", "KY", "KZ"]


def merge_ckpts():
    print("读取各岩心检查点：")
    parts = []
    for f in sorted(glob.glob(os.path.join(TAB_DIR, "features_ckpt*.csv"))):
        try:
            d = pd.read_csv(f)
            d = d[d["core_id"] != "core_id"]      # 追加写入可能混入重复表头
            parts.append(d)
            print(f"  {os.path.basename(f):<34s} {len(d):5d} 行")
        except Exception as e:
            print(f"  [跳过] {os.path.basename(f)}: {e}")
    if not parts:
        print("\n未找到任何检查点文件")
        return
    df = pd.concat(parts, ignore_index=True)
    for c in MERGE_NUM + ["valid"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df.drop_duplicates(MERGE_KEY, keep="first")   # 断点续跑可能重复写入
    if n0 != len(df):
        print(f"\n去除重复样本 {n0-len(df)} 个")

    origins = list(range(0, FULL_SIZE - WINDOW + 1, STRIDE))
    print(f"\n合计 {len(df)} 个样本，预期 {len(CORES)*len(origins)**3}")
    have = set(map(tuple, df[MERGE_KEY].values))
    missing = {c: sum(1 for a in origins for b in origins for z in origins
                      if (c, a, b, z) not in have) for c in CORES}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        print("  仍缺失：")
        for k, v in missing.items():
            print(f"    {k:<14s} {v}")
    else:
        print("  覆盖完整")

    out = os.path.join(TAB_DIR, "features_all.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.6g")
    print(f"\n已保存：{out}")
    if "valid" in df:
        print(f"有效样本 {int(df['valid'].sum())}，剔除 {int((df['valid']==0).sum())}")


# ---------------------------------------------------------------- 主流程
def main(only_core=None):
    """支持断点续跑：中途中断后重新运行会跳过已完成的样本。
    也可用 only_core 指定单块岩心，便于按岩心并行（开多个进程各跑一块）。"""
    index = pd.read_csv(os.path.join(TAB_DIR, "subsample_index.csv"))
    if only_core:
        index = index[index["core_id"] == only_core]

    ckpt = os.path.join(TAB_DIR, f"features_ckpt{'_'+only_core if only_core else ''}.csv")
    done = set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        done = set(zip(prev["core_id"], prev["ox"], prev["oy"], prev["oz"]))
        print(f"发现检查点，已完成 {len(done)} 个样本，将跳过")

    rows, failed = [], 0
    for i, r in index.iterrows():
        tag = (r["core_id"], r["ox"], r["oy"], r["oz"])
        if tag in done:
            continue
        path = os.path.join(RAW_DIR, r["file"])
        print(f"[{i+1}/{len(index)}] {r['file']}")
        try:
            rec = process_one(path, r["core_id"], r["ox"], r["oy"], r["oz"])
            if rec is not None:
                rows.append(rec)
                if len(rows) % 20 == 0:      # 每 20 个落盘一次
                    pd.DataFrame(rows).to_csv(ckpt, mode="a", index=False,
                                              header=not os.path.exists(ckpt),
                                              encoding="utf-8-sig", float_format="%.6g")
                    rows = []
        except Exception as e:
            failed += 1
            print(f"  失败：{e}")

    if rows:
        pd.DataFrame(rows).to_csv(ckpt, mode="a", index=False,
                                  header=not os.path.exists(ckpt),
                                  encoding="utf-8-sig", float_format="%.6g")

    df = pd.read_csv(ckpt)
    out = os.path.join(TAB_DIR, "features_all.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.6g")

    print(f"\n完成：成功 {len(df)}，失败 {failed}")
    print(f"有效样本 {int(df['valid'].sum())}，剔除 {int((df['valid']==0).sum())}")
    print("\n各岩心剔除明细：")
    print(pd.crosstab(df["core_id"], df["drop_reason"]))
    print(f"\n单样本 PNM 平均耗时 {df['pnm_time_sec'].mean():.2f} s "
          f"")
    print(f"结果已保存：{out}")


if __name__ == "__main__":
    import sys
    # 用法：python pnm_core.py            全部岩心，串行
    #       python pnm_core.py Kirby      只跑 Kirby，便于多进程并行
    #       python pnm_core.py --merge    合并各岩心检查点为 features_all.csv
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--merge":
        merge_ckpts()
    else:
        main(arg)
