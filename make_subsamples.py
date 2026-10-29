# -*- coding: utf-8 -*-
"""make_subsamples.py：把一块岩心的 1000³ 二值分割图像切成 512 个 256³ 子样本。

每轴起点为 0、100、…、700；第 n 个子样本的起点为 ox = n//64×100，oy = n%64//8×100，oz = n%8×100。
子样本取 volume[ox:ox+256, oy:oy+256, oz:oz+256]，存为 <RAW_DIR>/<岩心名>/<n>.hdf5，数据集名为 data，取值不变（0 为孔隙）。
已存在的子样本跳过；先写临时文件再改名，中断后重跑不会留下半截文件。

用法：python make_subsamples.py <岩心名> <二值图像 .raw 文件>
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from config import CORES, FULL_SIZE, RAW_DIR, STRIDE, WINDOW


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in CORES:
        raise SystemExit(f'用法：python make_subsamples.py <岩心名> <二值图像 .raw 文件>，岩心名取 {CORES}')
    core, raw = sys.argv[1], sys.argv[2]
    vol = np.fromfile(raw, dtype=np.uint8)
    if vol.size != FULL_SIZE ** 3:
        raise SystemExit(f'体素数为 {vol.size}，应为 {FULL_SIZE}³')
    vol = vol.reshape(FULL_SIZE, FULL_SIZE, FULL_SIZE)
    out = Path(RAW_DIR) / core
    out.mkdir(parents=True, exist_ok=True)
    m = (FULL_SIZE - WINDOW) // STRIDE + 1
    for n in range(m ** 3):
        fp = out / f'{n}.hdf5'
        if fp.exists():
            continue
        ox, oy, oz = n // (m * m) * STRIDE, n % (m * m) // m * STRIDE, n % m * STRIDE
        tmp = fp.with_suffix('.tmp')
        with h5py.File(tmp, 'w') as f:
            f.create_dataset('data', data=vol[ox:ox + WINDOW, oy:oy + WINDOW, oz:oz + WINDOW], compression='gzip')
        os.replace(tmp, fp)
    print(f'{core}：{m ** 3} 个子样本已写入 {out}', flush=True)


if __name__ == '__main__':
    main()
