"""全部路径相对于本文件所在目录。256³ 子样本 hdf5 默认放在 data/subsamples/<岩心>/<编号>.hdf5，
可用环境变量 DIGROCK_RAW 改到别处。"""
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent
ROOT = str(BASE)
RAW_DIR = os.environ.get('DIGROCK_RAW', str(BASE / 'data' / 'subsamples'))
TAB_DIR = str(BASE / 'data')
OUT_DIR = str(BASE / 'results')
Path(TAB_DIR).mkdir(exist_ok=True)
Path(OUT_DIR).mkdir(exist_ok=True)
CORES = ['BanderaBrown', 'BanderaGray', 'BB', 'Bentheimer', 'Berea', 'BSG',
         'BUG', 'CastleGate', 'Kirby', 'Leopard', 'Parker']
VOXEL_SIZE=2.25e-6
FULL_SIZE=1000
WINDOW=256
STRIDE=100
MU=1e-3
P_IN=101325.
P_OUT=0.
M2_TO_MD=1/9.869233e-16
ISO_FEATURES=['fractal_dimension','Porosity','pore_max','pore_mean',
              'throat_max','throat_mean','mean_coordination']
FEATURES=ISO_FEATURES
TARGETS=['KX','KY','KZ']
AXES=['x','y','z']
DIR_FEATURES_IMAGE=['phi_min','phi_slice_cv','chord_mean','corr_len','perc_frac','tau_geo']
DIR_FEATURES_NET=['k_bundle','orient_M','throat_dir','coord_dir']
DIR_FEATURES=DIR_FEATURES_IMAGE
LONG_FORMAT=True
USE_DIR=True
USE_PERP=False
SEED=42
LOG_TARGET=True
MODELS=['DNN','KNN','RF','SVR','XGB']
PROTOCOLS=['blocked','loco']
SSA_POP=6
SSA_ITER=12
N_SEEDS=10
OPTIMIZERS=['SSA','RandomSearch','TPE']
