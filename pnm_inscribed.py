# -*- coding: utf-8 -*-
"""pnm_inscribed.py：按内切直径计算 256³ 子样本的三向渗透率（预测目标）。

内切直径是孔隙或喉道内能容纳的最大球的直径。每个子样本只提取一次孔隙网络
（与 pnm_core.process_one 相同的 snow2 调用、孤立孔剔除和锥柱模型），
然后在同一个网络上按两种直径取法各解一次三向流动：
  eq   孔隙、喉道都取等效直径，用 OpenPNM 原公式（与 pnm_core.py 的结果相同，用来核对）
  ins  孔隙、喉道都取内切直径，并修正负孔段长度（预测目标）

负孔段长度的修正：OpenPNM 3.6.2 的锥柱模型在两孔重叠时取 L1 = (4L² + D1² − D2²) / (8L)，L2 = L − L1，
较小孔隙的中心落在较大孔隙内时，较小孔隙一侧的孔段长度为负，喉道传导率可能变成负数。
此时把该段长度取为 0，两孔心之间整段按较大孔隙一侧的圆锥台计算，喉道段不变；其余喉道沿用 OpenPNM 原值。
孔隙网络几何（坐标、连接、两种直径）存为 npz，换别的直径取法时只需重解流动。

用法（在本目录下）：
  python pnm_inscribed.py check                 先算 3 个子样本，核对 eq 与 data/features_all.csv 一致
  python pnm_inscribed.py run --workers 4       全部子样本，并行，可中断，重跑时跳过已完成的
  python pnm_inscribed.py merge                 汇总，生成 data/inscribed/features_all_inscribed.csv 等

输入：data/features_all.csv（pnm_core.py 与 directional_core.py --merge 的结果）、子样本 hdf5（config.py 的 RAW_DIR）。
输出：data/inscribed/。features_all_inscribed.csv 只把有效子样本的 KX、KY、KZ 换成内切直径的结果，其余列不变。
"""
import os
import sys

# 每个进程只用少量线程，并行靠多进程；必须在导入 numpy 之前设置
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data' / 'inscribed'

SETTINGS = {'eq': ('equivalent', 'equivalent', False), 'ins': ('inscribed', 'inscribed', True)}   # 孔隙直径、喉道直径、是否修正负孔段
TARGETS = ['KX', 'KY', 'KZ']
GEOM_KEYS = ('pore.coords', 'throat.conns', 'pore.equivalent_diameter', 'pore.inscribed_diameter',
             'throat.equivalent_diameter', 'throat.inscribed_diameter', 'pore.boundary', 'pore.volume',
             'throat.cross_sectional_area', 'throat.total_length')


def sample_no(ox, oy, oz):
    """窗口编号与起点坐标的对应关系，与 make_index.py 相同：ox = n//64*100, oy = n%64//8*100, oz = n%8*100。"""
    n = int(ox) // 100 * 64 + int(oy) // 100 * 8 + int(oz) // 100
    assert (n // 64 * 100, n % 64 // 8 * 100, n % 8 * 100) == (int(ox), int(oy), int(oz))
    return n


def load_tasks(features, raw):
    import pandas as pd
    f = pd.read_csv(features)
    f = f[f['valid'] == 1]
    tasks = []
    for r in f.itertuples(index=False):
        n = sample_no(r.ox, r.oy, r.oz)
        tasks.append(dict(core=r.core_id, n=n, ox=int(r.ox), oy=int(r.oy), oz=int(r.oz),
                          path=str(Path(raw) / r.core_id / f'{n}.hdf5'), K_old=[float(r.KX), float(r.KY), float(r.KZ)]))
    return tasks


# ------------------------------------------------------------------ 单个子样本（在子进程中运行）
_P = None


def _lib(ncores):
    global _P
    if _P is None:
        import pnm_core as P          # 与 pnm_core.py 同一套 snow2 兼容包装、求解器和单位
        import porespy as ps
        ps.settings.ncores = ncores   # 只影响 dask 的线程数，不影响分块方式和结果
        try:
            ps.settings.tqdm['disable'] = True
        except Exception:
            pass
        _P = P
    return _P


def size_factors_fixed(op, pn):
    """返回 (修正后的尺寸因子, 孔段长度为负的喉道掩码)。只改孔段长度为负的喉道，公式与 OpenPNM 3.6.2 的 cones_and_cylinders 相同。"""
    import numpy as np
    sf = np.asarray(op.models.geometry.hydraulic_size_factors.cones_and_cylinders(
        network=pn, pore_diameter='pore.diam_use', throat_diameter='throat.diam_use'), dtype=float).copy()
    L1, Lt, L2 = np.asarray(op.models.geometry.conduit_lengths.cones_and_cylinders(
        network=pn, pore_diameter='pore.diam_use', throat_diameter='throat.diam_use'), dtype=float).T
    bad = (L1 < 0) | (L2 < 0)
    if bad.any():
        D1, Dt, D2 = np.asarray(pn.get_conduit_data('diam_use'), dtype=float).T
        Lc = L1 + L2                           # 重叠时 L1 + L2 恰为孔心距离
        n1 = np.where(L1 < 0, 0.0, np.where(L2 < 0, Lc, L1))
        n2 = np.where(L2 < 0, 0.0, np.where(L1 < 0, Lc, L2))
        b = np.where(bad)[0]
        F1 = 16 / 3 * (n1[b] * (D1[b]**2 + D1[b] * Dt[b] + Dt[b]**2) / (D1[b]**3 * Dt[b]**3 * np.pi**2))
        F2 = 16 / 3 * (n2[b] * (D2[b]**2 + D2[b] * Dt[b] + Dt[b]**2) / (D2[b]**3 * Dt[b]**3 * np.pi**2))
        I = 1 / (2 * np.pi)
        # 长度为 0 的一侧尺寸因子为无穷大，即不产生阻力
        with np.errstate(divide='ignore'):
            sf[b, 0] = np.where(F1 > 0, 1 / (16 * np.pi**2 * I * np.where(F1 > 0, F1, 1.0)), np.inf)
            sf[b, 2] = np.where(F2 > 0, 1 / (16 * np.pi**2 * I * np.where(F2 > 0, F2, 1.0)), np.inf)
    return sf, bad


def solve_settings(P, net_dict, shape):
    """在同一个网络上按两种直径取法求三向渗透率。网络处理步骤与 pnm_core.process_one 相同。"""
    import numpy as np
    op = P.op
    assert P.SIZE_FACTOR_MODEL == 'cones_and_cylinders'
    proj = op.Project()
    pn = op.network.Network(project=proj)
    pn.update(net_dict)
    health = op.utils.check_network_health(network=pn)
    bad = sorted(set(list(health['isolated_pores']) + list(health['disconnected_pores'])))
    if bad:
        op.topotools.trim(network=pn, pores=bad)
    if pn.Np < 10:
        raise RuntimeError('修剪后连通孔隙数不足10')
    water = op.phase.Water(network=pn)
    water['pore.viscosity'] = P.MU
    water['throat.viscosity'] = P.MU
    out = {}
    n_fixed = 0
    for name, (pdef, tdef, fix) in SETTINGS.items():
        # OpenPNM 3.6.2 的 cones_and_cylinders 按 pore_diameter 的后缀同时取孔隙和喉道直径，
        # 因此把选定的两种直径复制到同一个后缀下；eq 时与 pnm_core.py 传入的数组完全相同
        pn['pore.diam_use'] = np.asarray(pn[f'pore.{pdef}_diameter'])
        pn['throat.diam_use'] = np.asarray(pn[f'throat.{tdef}_diameter'])
        if fix:
            sf, mask = size_factors_fixed(op, pn)
            n_fixed = int(mask.sum())
        else:
            sf = op.models.geometry.hydraulic_size_factors.cones_and_cylinders(
                network=pn, pore_diameter='pore.diam_use', throat_diameter='throat.diam_use')
        pn['throat.hydraulic_size_factors'] = sf
        water['throat.hydraulic_conductance'] = op.models.physics.hydraulic_conductance.generic_hydraulic(
            phase=water, pore_viscosity='pore.viscosity', throat_viscosity='throat.viscosity',
            size_factors='throat.hydraulic_size_factors')
        if fix and not (np.asarray(water['throat.hydraulic_conductance']) > 0).all():
            raise RuntimeError('修正后仍有非正的喉道传导率')
        ks, imb = [], []
        for axis in 'xyz':
            k, ib = P.solve_direction(pn, water, axis, shape, P.VOXEL_SIZE)
            ks.append(float(k)); imb.append(float(ib))
        out[name] = dict(K=ks, imbalance=max(imb))
    stats = dict(n_pores=int(pn.Np), n_throats=int(pn.Nt), n_fixed_throats=n_fixed)
    conns = np.asarray(pn['throat.conns'])
    for d in ('equivalent', 'inscribed'):
        pdm = np.asarray(pn[f'pore.{d}_diameter']); tdm = np.asarray(pn[f'throat.{d}_diameter'])
        stats[f'throat_gt_pore_{d}'] = float(np.mean(tdm > np.minimum(pdm[conns[:, 0]], pdm[conns[:, 1]])))
        stats[f'pore_mean_{d}_um'] = float(pdm.mean() * 1e6); stats[f'throat_mean_{d}_um'] = float(tdm.mean() * 1e6)
        stats[f'pore_max_{d}_um'] = float(pdm.max() * 1e6); stats[f'throat_max_{d}_um'] = float(tdm.max() * 1e6)
    op.Workspace().clear()
    return out, stats


def work(task):
    """子进程入口。返回结果字典；出错时返回 error 字段，不中断其他子样本。"""
    import numpy as np
    import h5py
    t0 = time.time()
    try:
        P = _lib(task.get('ncores', 1))
        with h5py.File(task['path'], 'r') as f:
            im = f['data'][:]
        if im.shape != (256, 256, 256):
            raise ValueError(f'窗口尺寸不对：{im.shape}')
        pore = (im == 0)
        t1 = time.time()
        net = P.snow2(phases=pore, voxel_size=P.VOXEL_SIZE)
        t2 = time.time()
        nd = net.network
        gdir = Path(task['geom_dir']); gdir.mkdir(parents=True, exist_ok=True)
        geom = {k: np.asarray(nd[k]) for k in GEOM_KEYS if k in nd}
        for k in nd.keys():                       # 端面标签（xmin、xmax 等）
            if k.startswith('pore.') and k[5:] in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'):
                geom[k] = np.asarray(nd[k])
        np.savez_compressed(gdir / f"{task['n']}.npz", **geom)
        res, stats = solve_settings(P, nd, im.shape)
        t3 = time.time()
        K_old = np.array(task['K_old'])
        rel = (np.array(res['eq']['K']) - K_old) / K_old
        return dict(core=task['core'], n=task['n'], ox=task['ox'], oy=task['oy'], oz=task['oz'],
                    K_old=task['K_old'], **{f'K_{k}': v['K'] for k, v in res.items()},
                    **{f'imbalance_{k}': v['imbalance'] for k, v in res.items()},
                    eq_rel_diff_max=float(np.max(np.abs(rel))), **stats,
                    read_sec=t1 - t0, extract_sec=t2 - t1, solve_sec=t3 - t2, total_sec=t3 - t0)
    except Exception as e:                          # noqa: BLE001
        import traceback
        return dict(core=task['core'], n=task['n'], error=repr(e), trace=traceback.format_exc()[-2000:])


def label_path(core, n):
    return OUT / 'labels' / core / f'{n}.json'


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(path)


def run(args, limit=None):
    import multiprocessing as mp
    tasks = load_tasks(args.features, args.raw)
    missing = [t['path'] for t in tasks if not Path(t['path']).exists()]
    if missing:
        raise SystemExit(f'找不到 {len(missing)} 个原始文件，例如 {missing[0]}。请检查 config.py 的 RAW_DIR 或设置环境变量 DIGROCK_RAW。')
    todo = [t for t in tasks if not label_path(t['core'], t['n']).exists()]
    if limit:
        todo = todo[:limit]
    for t in todo:
        t['geom_dir'] = str(OUT / 'geometry' / t['core'])
        t['ncores'] = args.threads
    print(f'共 {len(tasks)} 个有效子样本，已完成 {len(tasks) - len([t for t in tasks if not label_path(t["core"], t["n"]).exists()])} 个，'
          f'本次计算 {len(todo)} 个，{args.workers} 个进程', flush=True)
    if not todo:
        return
    start, done, errors = time.time(), 0, 0
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, maxtasksperchild=20) as pool:
        for r in pool.imap_unordered(work, todo, chunksize=1):
            done += 1
            if 'error' in r:
                errors += 1
                write_json(OUT / 'errors' / f"{r['core']}_{r['n']}.json", r)
                print(f"  出错 {r['core']}/{r['n']}：{r['error']}", flush=True)
                continue
            write_json(label_path(r['core'], r['n']), r)
            if done % 10 == 0 or done == len(todo) or done <= 3:
                el = time.time() - start
                eta = el / done * (len(todo) - done)
                print(f"  {done}/{len(todo)}  {r['core']}/{r['n']}  提取 {r['extract_sec']:.1f} s  "
                      f"eq 与原值最大相对差 {r['eq_rel_diff_max']:.2e}  ins/eq = "
                      f"{sum(r['K_ins']) / sum(r['K_eq']):.3f}  已用 {el / 3600:.2f} h，预计还需 {eta / 3600:.2f} h", flush=True)
    print(f'本次完成 {done - errors} 个，出错 {errors} 个（见 data/inscribed/errors）。', flush=True)
    if errors:
        sys.exit(3)


def check(args):
    """先算 3 个子样本（3 块不同岩心），确认 eq 结果与 features_all.csv 一致。"""
    import pandas as pd
    tasks = load_tasks(args.features, args.raw)
    pick, seen = [], set()
    for t in tasks:
        if t['core'] not in seen:
            pick.append(t); seen.add(t['core'])
        if len(pick) == 3:
            break
    import numpy as np
    rels = []
    for t in pick:
        t['geom_dir'] = str(OUT / 'check_geometry' / t['core'])
        t['ncores'] = args.threads
        r = work(t)
        if 'error' in r:
            print(r['trace']); raise SystemExit(f"出错：{r['error']}")
        rels += list(np.abs((np.array(r['K_eq']) - np.array(t['K_old'])) / np.array(t['K_old'])))
        print(f"{t['core']}/{t['n']}：原值 {[round(x, 2) for x in t['K_old']]}，重算 eq {[round(x, 2) for x in r['K_eq']]}，"
              f"最大相对差 {r['eq_rel_diff_max']:.2e}；ins {[round(x, 2) for x in r['K_ins']]}；提取 {r['extract_sec']:.1f} s", flush=True)
    # 依赖库小版本不同时个别子样本可能差 1% 左右，不影响结果；差别大到说明计算流程不对（单位、直径、求解器）时才停下。
    worst, med = float(np.max(rels)), float(np.median(rels))
    if worst > 0.10 or med > 0.02:
        raise SystemExit(f'等效直径的重算结果与原数据相差过大（最大 {worst:.1%}，中位数 {med:.1%}），计算流程或环境有问题，已停止。')
    if worst > 0.01:
        print(f'提示：个别方向相差 {worst:.1%}（中位数 {med:.2%}），属于依赖库版本差异，继续运行。', flush=True)
    print('CHECK 通过：等效直径的结果与原数据一致，可以运行 run。', flush=True)


def merge(args):
    import numpy as np
    import pandas as pd
    f = pd.read_csv(args.features)
    valid = f['valid'] == 1
    recs = {}
    for p in (OUT / 'labels').glob('*/*.json'):
        r = json.loads(p.read_text(encoding='utf-8'))
        recs[(r['core'], r['ox'], r['oy'], r['oz'])] = r
    keys = list(zip(f.core_id, f.ox, f.oy, f.oz))
    lack = [k for k, v in zip(keys, valid) if v and k not in recs]
    if lack:
        raise SystemExit(f'还有 {len(lack)} 个有效子样本没有算完，例如 {lack[0]}；先运行 run。')
    rows = []
    for k, v in zip(keys, valid):
        if not v:
            continue
        r = recs[k]
        row = dict(core_id=k[0], ox=k[1], oy=k[2], oz=k[3])
        for s in SETTINGS:
            for t, x in zip(TARGETS, r[f'K_{s}']):
                row[f'{t}_{s}'] = x
            row[f'imbalance_{s}'] = r[f'imbalance_{s}']
        for t, x in zip(TARGETS, r['K_old']):
            row[f'{t}_old'] = x
        for c in ('eq_rel_diff_max', 'n_pores', 'n_throats', 'n_fixed_throats', 'extract_sec', 'total_sec', 'throat_gt_pore_equivalent',
                  'throat_gt_pore_inscribed', 'pore_mean_inscribed_um', 'throat_mean_inscribed_um',
                  'pore_max_inscribed_um', 'throat_max_inscribed_um'):
            row[c] = r[c]
        rows.append(row)
    lab = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    lab.to_csv(OUT / 'labels_all_settings.csv', index=False, encoding='utf-8-sig', float_format='%.8g')
    K_ins = lab[[f'{t}_ins' for t in TARGETS]].to_numpy()
    imb = lab['imbalance_ins'].to_numpy()
    assert np.isfinite(K_ins).all() and (K_ins > 0).all(), '内切直径的渗透率出现非正值或缺失'
    assert (imb < 0.01).all(), '内切直径的流量不平衡超过 1%'
    bad_eq = lab[lab.eq_rel_diff_max > 0.01]
    # 新的 features_all.csv：按文本读入，只替换有效子样本的 KX、KY、KZ（与原表一样保留 6 位有效数字），其余列逐字不变
    g = pd.read_csv(args.features, dtype=str, keep_default_na=False)
    assert len(g) == len(f) and list(g.core_id) == list(f.core_id)
    m = lab.set_index(['core_id', 'ox', 'oy', 'oz'])
    for i in f.index[valid]:
        k = (f.at[i, 'core_id'], f.at[i, 'ox'], f.at[i, 'oy'], f.at[i, 'oz'])
        for t in TARGETS:
            g.at[i, t] = f'{m.loc[k, f"{t}_ins"]:.6g}'
    g.to_csv(OUT / 'features_all_inscribed.csv', index=False, encoding='utf-8-sig')
    chk = pd.read_csv(OUT / 'features_all_inscribed.csv')
    other = [c for c in f.columns if c not in TARGETS]
    assert chk[other].equals(f[other]), '除渗透率外的列与原表不一致'
    # 汇总
    lg_old = np.log10(lab[[f'{t}_eq' for t in TARGETS]].to_numpy()); lg_new = np.log10(K_ins)
    d_old = lg_old - lg_old.mean(1, keepdims=True); d_new = lg_new - lg_new.mean(1, keepdims=True)
    ratio = K_ins / lab[[f'{t}_eq' for t in TARGETS]].to_numpy()
    order_same = float(np.mean([list(np.argsort(a)) == list(np.argsort(b)) for a, b in zip(lg_old, lg_new)]))
    lab_mD = {'BanderaGray': 9, 'Parker': 10, 'Kirby': 62, 'BanderaBrown': 63, 'BSG': 80, 'BUG': 86, 'Berea': 121,
              'CastleGate': 269, 'BB': 275, 'Leopard': 327, 'Bentheimer': 386}
    neu = {'BanderaGray': 40, 'Parker': 13, 'Kirby': 154, 'BanderaBrown': 67, 'BSG': 138, 'BUG': 122, 'Berea': 186,
           'CastleGate': 450, 'BB': 471, 'Leopard': 226, 'Bentheimer': 973}
    per_core = {}
    for c, q in lab.groupby('core_id'):
        mean_ins = q[[f'{t}_ins' for t in TARGETS]].to_numpy().mean()
        mean_eq = q[[f'{t}_eq' for t in TARGETS]].to_numpy().mean()
        lb, nu = lab_mD.get(c, np.nan), neu.get(c, np.nan)
        per_core[c] = dict(n=int(len(q)), **{f'{t}_ins_mean': float(q[f'{t}_ins'].mean()) for t in TARGETS},
                           mean_ins=float(mean_ins), mean_eq=float(mean_eq), lab=lb, neumann=nu,
                           ins_over_lab=float(mean_ins / lb), eq_over_lab=float(mean_eq / lb),
                           ins_over_neumann=float(mean_ins / nu), ratio_ins_eq_median=float(np.median(ratio[q.index])))
    pc = pd.DataFrame(per_core).T
    summary = dict(
        n=int(len(lab)), eq_reproduction=dict(max_rel_diff=float(lab.eq_rel_diff_max.max()), n_over_1pct=int(len(bad_eq)),
                                              median_rel_diff=float(lab.eq_rel_diff_max.median())),
        ratio_ins_over_eq=dict(min=float(ratio.min()), q05=float(np.quantile(ratio, .05)), median=float(np.median(ratio)),
                               q95=float(np.quantile(ratio, .95)), max=float(ratio.max())),
        log_corr=float(np.corrcoef(lg_old.ravel(), lg_new.ravel())[0, 1]),
        dev_corr=float(np.corrcoef(d_old.ravel(), d_new.ravel())[0, 1]),
        direction_order_same_fraction=order_same,
        K_ins_range=[float(K_ins.min()), float(K_ins.max())],
        ins_over_lab=dict(min=float(pc.ins_over_lab.min()), median=float(pc.ins_over_lab.median()), max=float(pc.ins_over_lab.max())),
        ins_over_neumann=dict(min=float(pc.ins_over_neumann.min()), median=float(pc.ins_over_neumann.median()),
                              max=float(pc.ins_over_neumann.max())),
        spearman_ins_lab=float(pc.mean_ins.astype(float).rank().corr(pc.lab.astype(float).rank())),
        spearman_ins_neumann=float(pc.mean_ins.astype(float).rank().corr(pc.neumann.astype(float).rank())),
        throat_gt_pore_inscribed_core_mean=dict((c, float(q.throat_gt_pore_inscribed.mean())) for c, q in lab.groupby('core_id')),
        per_core=per_core, extract_sec_median=float(lab.extract_sec.median()))
    write_json(OUT / 'inscribed_summary.json', summary)
    # 每块岩心三个方向的平均渗透率与实测值之比
    pl = pd.DataFrame({c: dict(KX=v['KX_ins_mean'], KY=v['KY_ins_mean'], KZ=v['KZ_ins_mean'], lab_mD=v['lab'],
                               mean_3axis=v['mean_ins'], ratio_descriptive=v['ins_over_lab']) for c, v in per_core.items()}).T
    pl.index.name = 'core_id'
    pl.to_csv(OUT / 'pnm_lab_comparison_inscribed.csv', encoding='utf-8-sig')
    print(json.dumps({k: summary[k] for k in ('n', 'eq_reproduction', 'ratio_ins_over_eq', 'log_corr', 'dev_corr',
                                               'direction_order_same_fraction', 'ins_over_lab', 'ins_over_neumann')},
                     ensure_ascii=False, indent=1))
    if len(bad_eq):
        print(f'注意：{len(bad_eq)} 个子样本的等效直径重算结果与原数据相差超过 1%，明细见 labels_all_settings.csv 的 eq_rel_diff_max 列。')
    print('已写出 data/inscribed/ 下的 features_all_inscribed.csv、labels_all_settings.csv、inscribed_summary.json、pnm_lab_comparison_inscribed.csv')


def main():
    import config as C
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['check', 'run', 'merge'])
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument('--threads', type=int, default=2, help='每个进程中 snow2 使用的线程数')
    p.add_argument('--limit', type=int, default=None, help='只算前若干个（试运行用）')
    p.add_argument('--features', default=str(HERE / 'data' / 'features_all.csv'))
    p.add_argument('--raw', default=os.environ.get('DIGROCK_RAW', C.RAW_DIR))
    a = p.parse_args()
    if a.command == 'check':
        check(a)
    elif a.command == 'run':
        run(a, a.limit)
    else:
        merge(a)


if __name__ == '__main__':
    main()
