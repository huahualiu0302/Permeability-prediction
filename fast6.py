"""fast6.py：只用不需要提取孔隙网络的 6 项参数（孔隙度加除迂曲度外的 5 项方向参数）训练和评价 DNN。

流程与 experiments.py 的其他试验相同：训练集内按岩心分组三折交叉验证，SSA 种群 6、迭代 12、共 90 次评估，
寻优后以 10 个随机种子（42～51）重复训练。结果写入 results/jobs/<任务号>，中断后重跑跳过已完成的部分。
用法：python fast6.py [blocked|loco]
"""
import sys

import experiments as R

NET = ('pore_max', 'pore_mean', 'throat_max', 'throat_mean', 'mean_coordination')
DROP = NET + ('fractal_dimension', 'tau_geo_自身方向')


def main():
    protocol = sys.argv[1] if len(sys.argv) > 1 else 'blocked'
    d = R.data_for('image13', DROP)
    print('features:', d.feat_names, flush=True)
    for fold in d.folds_of(protocol):
        job, tr, te, inner = R.prepare_job(d, 'image13', DROP, protocol, fold, 'DNN', 'SSA', 90)
        print(protocol, fold, job.name, flush=True)
        t = R.tune(d, tr, inner, 'DNN', 'SSA', 90, job)
        print('tuned', t['params'], round(t['tune_sec'], 1), flush=True)
        R.evaluate_job(d, tr, te, 'DNN', t, job, range(42, 52))
    R.summarize()


if __name__ == '__main__':
    main()
