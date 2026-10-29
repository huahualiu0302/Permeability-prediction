# -*- coding: utf-8 -*-
"""
ssa.py：麻雀搜索算法（SSA）与两个对照优化器（随机搜索、TPE）

1. 每个超参数有独立的 (low, high, scale, dtype)，连续量支持对数尺度采样。
2. 发现者、跟随者、警戒者的更新按 SSA 原文公式实现。
3. 警戒者数量取 round(pop * SD_RATIO)。
4. 记录收敛曲线与评估次数，用于与随机搜索、TPE 做同预算对比。
"""
import numpy as np

ST = 0.8          # 安全阈值
PD_RATIO = 0.2    # 发现者比例
SD_RATIO = 0.2    # 警戒者比例


class Space:
    """超参数搜索空间。每维形如 (name, low, high, kind)。

    kind ∈ {'int', 'float', 'logfloat', 'cat'}：
      int      —— 整数，均匀采样
      float    —— 实数，均匀采样
      logfloat —— 实数，对数均匀采样（学习率、正则化系数必须用这个）
      cat      —— 类别，low 为候选列表，内部用索引编码
    """

    def __init__(self, dims):
        self.dims = dims
        self.dim = len(dims)

    def bounds(self):
        lo, hi = [], []
        for name, low, high, kind in self.dims:
            if kind == "cat":
                lo.append(0.0); hi.append(len(low) - 1e-9)
            elif kind == "logfloat":
                lo.append(np.log10(low)); hi.append(np.log10(high))
            else:
                lo.append(float(low)); hi.append(float(high))
        return np.array(lo), np.array(hi)

    def decode(self, x):
        """把内部连续向量翻译成超参数字典。"""
        out = {}
        for v, (name, low, high, kind) in zip(x, self.dims):
            if kind == "int":
                out[name] = int(np.clip(round(v), low, high))
            elif kind == "float":
                out[name] = float(np.clip(v, low, high))
            elif kind == "logfloat":
                out[name] = float(10 ** np.clip(v, np.log10(low), np.log10(high)))
            elif kind == "cat":
                out[name] = low[int(np.clip(np.floor(v), 0, len(low) - 1))]
        return out

    def sample(self, rng, n):
        lo, hi = self.bounds()
        return rng.uniform(lo, hi, size=(n, len(lo)))


def _clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def _ssa_budget(pop, iters):
    """SSA 的实际适应度评估次数：初始种群 + 每轮(发现者+跟随者=pop) + 警戒者 sdNum。
    随机搜索与 TPE 必须用同一预算，否则同预算对比不成立。"""
    return pop + iters * (pop + max(1, round(pop * SD_RATIO)))


def ssa_optimize(space: Space, fun, pop=20, iters=40, seed=0, verbose=True):
    """返回 (best_params, best_fitness, curve, n_eval)。fun 接收参数字典，返回越小越好的适应度。"""
    rng = np.random.default_rng(seed)
    lo, hi = space.bounds()
    dim = space.dim
    pNum = max(1, round(pop * PD_RATIO))
    sdNum = max(1, round(pop * SD_RATIO))

    X = space.sample(rng, pop)
    fit = np.array([fun(space.decode(x)) for x in X])
    n_eval = pop

    pX, pFit = X.copy(), fit.copy()
    gi = int(np.argmin(pFit))
    bestX, fMin = pX[gi].copy(), float(pFit[gi])
    curve = np.zeros(iters)

    for t in range(iters):
        order = np.argsort(pFit)
        worst = pX[order[-1]].copy()
        fmax = float(pFit[order[-1]])

        # ---- 发现者：式(1) ----
        R2 = rng.random()
        for rank in range(pNum):
            i = order[rank]
            if R2 < ST:
                alpha = rng.random() + 1e-12
                X[i] = pX[i] * np.exp(-(rank + 1) / (alpha * iters))
            else:
                X[i] = pX[i] + rng.normal()
            X[i] = _clip(X[i], lo, hi)
            fit[i] = fun(space.decode(X[i])); n_eval += 1

        bestXX = X[int(np.argmin(fit))].copy()

        # ---- 跟随者：式(2) ----
        for rank in range(pNum, pop):
            i = order[rank]
            if rank > pop / 2:
                X[i] = rng.normal() * np.exp((worst - pX[i]) / ((rank + 1) ** 2))
            else:
                A = np.where(rng.random(dim) < 0.5, -1.0, 1.0)
                X[i] = bestXX + np.mean(A * np.abs(pX[i] - bestXX))
            X[i] = _clip(X[i], lo, hi)
            fit[i] = fun(space.decode(X[i])); n_eval += 1

        # ---- 警戒者：式(3) ----
        for i in rng.permutation(pop)[:sdNum]:
            if pFit[i] > fMin:
                X[i] = bestX + rng.normal() * np.abs(pX[i] - bestX)
            else:
                lam = rng.uniform(-1, 1)
                X[i] = pX[i] + lam * (np.abs(pX[i] - worst) / (abs(pFit[i] - fmax) + 1e-50))
            X[i] = _clip(X[i], lo, hi)
            fit[i] = fun(space.decode(X[i])); n_eval += 1

        upd = fit < pFit
        pX[upd], pFit[upd] = X[upd], fit[upd]
        gi = int(np.argmin(pFit))
        if pFit[gi] < fMin:
            fMin, bestX = float(pFit[gi]), pX[gi].copy()
        curve[t] = fMin
        if verbose and (t + 1) % 10 == 0:
            print(f"    SSA iter {t+1:3d}  fMin={fMin:.5f}  n_eval={n_eval}")

    return space.decode(bestX), fMin, curve, n_eval


def random_search(space: Space, fun, n_eval=None, pop=20, iters=40, seed=0, verbose=False):
    """同预算随机搜索对照。预算按 SSA 的实际评估次数对齐。

    verbose 参数仅为与 ssa_optimize 保持签名一致，便于调用方统一传参。
    """
    rng = np.random.default_rng(seed)
    budget = n_eval if n_eval else _ssa_budget(pop, iters)
    X = space.sample(rng, budget)
    fits = np.array([fun(space.decode(x)) for x in X])
    curve = np.minimum.accumulate(fits)
    k = int(np.argmin(fits))
    return space.decode(X[k]), float(fits[k]), curve, budget


def tpe_search(space: Space, fun, n_eval=None, pop=20, iters=40, seed=0, verbose=False):
    """贝叶斯优化（TPE）对照，需要 optuna。未安装时退化为随机搜索并给出提示。"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("    未安装 optuna，TPE 对照退化为随机搜索：pip install optuna")
        return random_search(space, fun, n_eval, pop, iters, seed, verbose)

    budget = n_eval if n_eval else _ssa_budget(pop, iters)

    def objective(trial):
        params = {}
        for name, low, high, kind in space.dims:
            if kind == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            elif kind == "float":
                params[name] = trial.suggest_float(name, low, high)
            elif kind == "logfloat":
                params[name] = trial.suggest_float(name, low, high, log=True)
            elif kind == "cat":
                params[name] = trial.suggest_categorical(name, list(low))
        return fun(params)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    curve = np.minimum.accumulate([t.value for t in study.trials])
    return study.best_params, float(study.best_value), np.asarray(curve), budget


OPTIMIZERS = {"SSA": ssa_optimize, "RandomSearch": random_search, "TPE": tpe_search}