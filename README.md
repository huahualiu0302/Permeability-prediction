# 砂岩数字岩心三向渗透率预测

由 256³ 数字岩心子样本计算孔隙结构参数和孔隙网络模拟渗透率，再训练和评价机器学习模型，预测 X、Y、Z 三个方向的渗透率。仓库只含代码，不含数据。

Code for predicting directional permeability (KX, KY, KZ) of sandstone digital cores from pore-structure parameters with pore-network simulation and machine learning. No data are included.

## 环境

Python 3.11：

```
pip install -r requirements.txt
```

## 数据

数字岩心为 Digital Rocks Portal 的公开数据集：

> Neumann R, Andreeta M, Lucas-Oliveira E. 11 Sandstones: raw, filtered and segmented data. Digital Rocks Portal, 2020. doi:10.17612/F4H1-W124

使用其中 11 块砂岩的二值分割图像（1000³，uint8，体素 2.25 µm，0 为孔隙）。程序中的岩心名：

| 编号 | `core_id` | 名称 |
| --- | --- | --- |
| 1 | `BanderaGray` | Bandera Gray |
| 2 | `Parker` | Parker |
| 3 | `Kirby` | Kirby |
| 4 | `BanderaBrown` | Bandera Brown |
| 5 | `BSG` | Berea Sister Gray |
| 6 | `BUG` | Berea Upper Gray |
| 7 | `Berea` | Berea |
| 8 | `CastleGate` | Castlegate |
| 9 | `BB` | Buff Berea |
| 10 | `Leopard` | Leopard |
| 11 | `Bentheimer` | Bentheimer |

每块岩心切成 512 个 256³ 子样本，每轴起点为 0、100、…、700，数组第 0、1、2 轴对应 KX、KY、KZ 的方向。子样本默认存在 `data/subsamples/<core_id>/<n>.hdf5`，存到别处时设置环境变量 `DIGROCK_RAW`。

## 运行顺序

命令都在仓库根目录下运行，结果写在 `data/` 和 `results/` 下。中断后重跑会跳过已完成的部分。

| 步骤 | 命令 | 作用 |
|---|---|---|
| 1 | `python make_subsamples.py <core_id> <二值图像>`，每块岩心一次 | 切 256³ 子样本 |
| 2 | `python make_index.py` | 子样本的岩心编号与起点坐标索引 |
| 3 | `python pnm_core.py`，再 `python pnm_core.py --merge` | SNOW 提取孔隙网络，计算 7 项常规参数（孔隙度、分形维数、平均与最大孔径、平均与最大喉径、平均配位数），写 `data/features_all.csv`。可按岩心并行：`python pnm_core.py Berea` |
| 4 | `python directional_core.py`，再 `python directional_core.py --merge` | 6 项方向参数（连通孔隙度、迂曲度、平均弦长、相关长度、最小切片孔隙度、切片孔隙度变异系数），并入 `data/features_all.csv` |
| 5 | `python pnm_inscribed.py check`、`run`、`merge` | 按内切直径和锥柱模型计算三向渗透率（预测目标），写 `data/inscribed/features_all_inscribed.csv` |
| 6 | 把 `data/features_all.csv` 另存为 `data/features_all_equivalent.csv`，再把 `data/inscribed/features_all_inscribed.csv` 复制为 `data/features_all.csv` | 后续建模用内切直径的渗透率 |
| 7 | `python make_splits.py` | 空间分块划分和留一岩心划分，写 `data/splits.csv` |
| 8 | 见下 | 训练与评价 |

第 5 步的锥柱模型中，较小孔隙的中心落在较大孔隙内时，OpenPNM 3.6.2 给出的较小孔隙一侧孔段长度为负。程序把该段长度取为 0，两孔心之间整段按较大孔隙一侧的圆锥台计算。

训练与评价（`experiments.py`、`fast6.py`）：

```
# 5 种模型，仅常规参数（base7）与加入方向参数（image13），空间分块划分，SSA 寻优，10 个随机种子
python experiments.py models --experiments base7 image13 --models DNN KNN RF SVR XGB --optimizers SSA --seeds 10

# 随机搜索与 TPE 寻优对照
python experiments.py models --experiments base7 image13 --models DNN KNN RF SVR XGB --optimizers RandomSearch TPE --seeds 10

# 留一岩心交叉验证
python experiments.py models --experiments base7 image13 --protocols loco --models DNN --optimizers SSA --seeds 10

# 方向对照：三向共用常规参数、方向参数取三向平均、方向参数错配
python experiments.py models --experiments shared7 image13_iso image13_shuffle --models DNN --optimizers SSA --seeds 10

# 删除平均孔径、平均喉径后重新寻优和训练
python experiments.py collinearity --experiments base7 image13 --models DNN --optimizers SSA --seeds 10

# 只用孔隙度和 5 项方向参数（不需要提取孔隙网络），空间分块划分与留一岩心
python fast6.py blocked
python fast6.py loco

# 置换重要性，<job> 为 results/jobs/ 下对应任务的文件夹名
python experiments.py importance --job <job> --seeds 10 --repeats 10

# 汇总全部任务的指标到 results/metrics.csv
python experiments.py summary
```

每个任务按输入、代码和设置生成独立的任务号，结果存在 `results/jobs/<任务号>/`。

## 文件

| 文件 | 内容 |
|---|---|
| `config.py` | 路径、岩心名、体素大小、子样本尺寸、参数名、模型与寻优设置 |
| `make_subsamples.py` | 切 256³ 子样本 |
| `make_index.py` | 子样本编号与起点坐标索引 |
| `pnm_core.py` | SNOW 提取孔隙网络、常规参数、OpenPNM 三向流动求解 |
| `pnm_inscribed.py` | 按内切直径和锥柱模型计算三向渗透率，含负孔段长度的修正 |
| `porespy_compat.py` | PoreSpy 3.0.4 的 snow2 在 NumPy 1.26 下的兼容包装 |
| `directional_core.py` | 方向参数，迂曲度用 scikit-image 的 MCP_Geometric 计算测地距离 |
| `make_splits.py` | 数据划分 |
| `model_core.py` | 模型、搜索空间、数据组织和评价指标 |
| `ssa.py` | 麻雀搜索算法、随机搜索、TPE |
| `experiments.py` | 寻优、训练、评价、置换重要性 |
| `fast6.py` | 只用 6 项不需要提取孔隙网络的参数的 DNN |
