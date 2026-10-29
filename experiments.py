"""寻优、训练、评价和置换重要性。每个任务记录所用输入、代码和划分。"""
import os
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v,'2')
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import config as C
import model_core as M

BASE=Path(__file__).resolve().parent
RESULTS=BASE/'results'
EXPERIMENTS={
    'base7':dict(long=False,use_dir=False,use_perp=False),
    'shared7':dict(long=True,use_dir=False,use_perp=False),
    'image13':dict(long=True,use_dir=True,use_perp=False,dir_only=C.DIR_FEATURES_IMAGE),
    'image19':dict(long=True,use_dir=True,use_perp=True,dir_only=C.DIR_FEATURES_IMAGE),
    'image13_iso':dict(long=True,use_dir=True,use_perp=False,dir_only=C.DIR_FEATURES_IMAGE,dir_isomean=True),
    'image13_shuffle':dict(long=True,use_dir=True,use_perp=False,dir_only=C.DIR_FEATURES_IMAGE,dir_shuffle='derange'),
    'network17':dict(long=True,use_dir=True,use_perp=False,dir_only=C.DIR_FEATURES_IMAGE+C.DIR_FEATURES_NET),
    'network27':dict(long=True,use_dir=True,use_perp=True,dir_only=C.DIR_FEATURES_IMAGE+C.DIR_FEATURES_NET),
    'bundle8':dict(long=True,use_dir=True,use_perp=False,dir_only=['k_bundle']),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def versions():
    result={}
    for n in ['numpy','pandas','scikit-learn','scipy','xgboost','optuna','porespy','openpnm','h5py']:
        try: result[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: result[n]=None
    return result

def data_for(name,drop=()):
    d=M.Data(**EXPERIMENTS[name],verbose=False)
    keep=[i for i,n in enumerate(d.feat_names) if n not in drop]
    if set(drop)-set(d.feat_names): raise ValueError('Unknown dropped feature')
    d.feat_names=[d.feat_names[i] for i in keep]
    if d.long:
        d.Xax=[x[:,keep] for x in d.Xax]; d.X=d.Xax[0]
    else: d.X=d.X[:,keep]
    return d

def inner_folds(d,tr):
    # The outer test rows never enter this function; windows from one core stay together.
    groups=d.df.iloc[tr].core_id.to_numpy()
    n=min(3,len(np.unique(groups)))
    if n<2: raise ValueError('At least two training cores required for inner CV')
    return list(GroupKFold(n_splits=n).split(tr,groups=groups))

def matrices(d,rows):
    if d.long:
        x,y=d.stack(rows)
        return x,y[:,None]
    return d.X[rows],d.Y[rows]

def fit(d,tr,model,params,seed):
    x,y=matrices(d,tr)
    tx,sy=M.fit_pipeline(x,y)
    est=M.make_model(model,params,seed,multioutput=not d.long)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t=time.perf_counter()
        est.fit(tx(x),sy.transform(y).ravel() if d.long else sy.transform(y))
        elapsed=time.perf_counter()-t
    def predict_log(rows,arrays=None):
        if d.long:
            xx=d.Xax if arrays is None else arrays
            return np.column_stack([sy.inverse_transform(est.predict(tx(xx[a][rows])).reshape(-1,1)).ravel() for a in range(3)])
        xx=d.X if arrays is None else arrays
        return sy.inverse_transform(est.predict(tx(xx[rows])))
    return predict_log, elapsed, sorted(set(str(w.message) for w in caught))

def identity(d,experiment,drop,protocol,fold,model,optimizer,budget):
    return dict(schema=1,experiment=experiment,dropped=list(drop),features=d.feat_names,
        experiment_options=EXPERIMENTS[experiment],protocol=protocol,outer_fold=int(fold),
        model=model,optimizer=optimizer,budget=budget,seed=C.SEED,
        input_hashes={n:sha(BASE/'data'/n) for n in ['features_all.csv','splits.csv']},
        code_hashes={n:sha(BASE/n) for n in ['experiments.py','model_core.py','config.py','ssa.py']},
        versions=versions(),inner_group='core_id',imputation='training_fold_median',
        dnn_early_stopping=False)

def prepare_job(d,experiment,drop,protocol,fold,model,optimizer,budget):
    spec=identity(d,experiment,drop,protocol,fold,model,optimizer,budget)
    ident=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
    job=RESULTS/'jobs'/ident; job.mkdir(parents=True,exist_ok=True)
    write_json(job/'manifest.json',spec)
    tr,te,_=d.fold(protocol,fold)
    if set(tr)&set(te): raise ValueError('Train/test overlap in row ids')
    inner=inner_folds(d,tr)
    fold_record=dict(train_row_ids=tr.tolist(),test_row_ids=te.tolist(),inner=[
        dict(train_row_ids=tr[a].tolist(),validation_row_ids=tr[b].tolist()) for a,b in inner])
    write_json(job/'folds.json',fold_record)
    return job,tr,te,inner

def tune(d,tr,inner,model,optimizer,budget,job):
    fp=job/'tuning.json'
    if fp.exists(): return json.loads(fp.read_text(encoding='utf-8'))
    if optimizer=='TPE' and versions()['optuna'] is None:
        raise RuntimeError('optuna is missing; TPE must not silently become random search')
    errors=[]; fits=[]
    def objective(params):
        scores=[]
        for i,(a,b) in enumerate(inner):
            train=tr[a]; val=tr[b]
            # Fixed subsample shared by all candidate SVR configurations.
            if model=='SVR' and len(train)>M.SVR_TUNE_N:
                train=np.random.default_rng(C.SEED+i).choice(train,M.SVR_TUNE_N,replace=False)
            try:
                x,y=matrices(d,train); xv,yv=matrices(d,val)
                tx,sy=M.fit_pipeline(x,y)
                est=M.make_model(model,params,C.SEED,multioutput=not d.long)
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always')
                    est.fit(tx(x),sy.transform(y).ravel() if d.long else sy.transform(y))
                scores.append(float(mean_squared_error(sy.transform(yv).ravel() if d.long else sy.transform(yv),est.predict(tx(xv)))))
                fits.append(dict(inner_fold=i,warnings=sorted(set(str(w.message) for w in ws))))
            except Exception as e:
                errors.append(dict(params=params,inner_fold=i,error=repr(e)))
                return 1e9
        return float(np.mean(scores))
    t=time.perf_counter(); space=M.build_space(model)
    if optimizer=='SSA':
        if budget!=90: raise ValueError('SSA uses the fixed 6 population, 12 iterations, 90 evaluation protocol')
        params,score,curve,n=M.OPTIMIZERS[optimizer](space,objective,pop=6,iters=12,seed=C.SEED)
    else:
        params,score,curve,n=M.OPTIMIZERS[optimizer](space,objective,n_eval=budget,seed=C.SEED)
    if not np.isfinite(score) or score>=1e9:
        write_json(job/'tuning_failure.json',dict(errors=errors)); raise RuntimeError('All tuning candidates failed')
    rec=dict(params=params,inner_mse=float(score),n_eval=int(n),tune_sec=time.perf_counter()-t,
             convergence=list(map(float,curve)),failures=errors,fit_warnings=fits)
    write_json(fp,rec)
    return rec

def evaluate_job(d,tr,te,model,tuning,job,seeds):
    for seed in seeds:
        fp=job/f'eval_{seed}.json'
        if fp.exists(): continue
        predlog,fit_sec,ws=fit(d,tr,model,tuning['params'],seed)
        predlog(te[:min(32,len(te))])
        timings=[]
        for _ in range(5):
            start=time.perf_counter(); logp=predlog(te)
            timings.append((time.perf_counter()-start)/len(te))
        pred=10**logp
        if not np.isfinite(pred).all(): raise ValueError('Non-finite predictions')
        metrics=M.evaluate(d.Yraw[te],pred)
        for m in metrics:
            for key,val in m.items():
                if isinstance(val,float) and not np.isfinite(val): m[key]=None
        majority=int(np.bincount(d.Yraw[tr].argmax(axis=1),minlength=3).argmax())
        rec=dict(seed=int(seed),fit_sec=fit_sec,predict_sec_per_window=float(np.median(timings)),
                 timing_scope='cached feature matrix -> preprocessing -> three logK predictions; extraction excluded',
                 metrics=metrics,warnings=ws,n_train=len(tr),n_test=len(te),
                 majority_direction=C.TARGETS[majority],
                 majority_hit_rate=float(np.mean(d.Yraw[te].argmax(axis=1)==majority)))
        table=pd.DataFrame({'row_id':te,'core_id':d.df.iloc[te].core_id.to_numpy()})
        for j,k in enumerate(C.TARGETS): table['true_'+k]=d.Yraw[te,j];table['pred_'+k]=pred[:,j]
        table.to_csv(job/f'predictions_{seed}.csv',index=False,encoding='utf-8-sig')
        write_json(fp,rec)
        print(f'  seed={seed} completed, fit={fit_sec:.1f}s',flush=True)

def summarize():
    rows=[]
    for job in sorted((RESULTS/'jobs').glob('*')):
        if not (job/'manifest.json').exists(): continue
        spec=json.loads((job/'manifest.json').read_text(encoding='utf-8'))
        for p in job.glob('eval_*.json'):
            rec=json.loads(p.read_text(encoding='utf-8'))
            for metric in rec['metrics']:
                rows.append(dict(job_id=job.name,experiment=spec['experiment'],dropped=';'.join(spec['dropped']),
                    protocol=spec['protocol'],fold=spec['outer_fold'],model=spec['model'],optimizer=spec['optimizer'],
                    n_feature=len(spec['features']),seed=rec['seed'],fit_sec=rec['fit_sec'],
                    predict_sec_per_window=rec['predict_sec_per_window'],majority_hit_rate=rec['majority_hit_rate'],**metric))
    if rows:
        pd.DataFrame(rows).to_csv(RESULTS/'metrics.csv',index=False,encoding='utf-8-sig')
    print(f'Summary: {len(rows)} metric rows; distinct jobs are never silently averaged',flush=True)

def run_models(args):
    drops=[()] if args.command=='models' else [(),('pore_mean',),('throat_mean',),('pore_mean','throat_mean')]
    for experiment in args.experiments:
        for drop in drops:
            d=data_for(experiment,drop)
            for protocol in args.protocols:
                folds=d.folds_of(protocol) if args.folds is None else args.folds
                for fold in folds:
                    if fold not in d.folds_of(protocol): raise ValueError('Invalid outer fold')
                    for model in args.models:
                        for optimizer in args.optimizers:
                            print(f'{experiment} drop={drop} {protocol}/{fold} {model}/{optimizer}',flush=True)
                            job,tr,te,inner=prepare_job(d,experiment,drop,protocol,fold,model,optimizer,args.budget)
                            t=tune(d,tr,inner,model,optimizer,args.budget,job)
                            evaluate_job(d,tr,te,model,t,job,range(42,42+args.seeds))
                            summarize()

def importance(args):
    job=RESULTS/'jobs'/args.job
    spec=json.loads((job/'manifest.json').read_text(encoding='utf-8'))
    d=data_for(spec['experiment'],spec['dropped'])
    if identity(d,spec['experiment'],spec['dropped'],spec['protocol'],spec['outer_fold'],spec['model'],spec['optimizer'],spec['budget'])!=spec:
        raise RuntimeError('Code/input identity differs; select a current job')
    if spec['protocol']!='blocked': raise ValueError('This interpretation protocol uses the blocked holdout')
    tr,te,_=d.fold(spec['protocol'],spec['outer_fold'])
    tuning=json.loads((job/'tuning.json').read_text(encoding='utf-8'))
    groups={n:[n] for n in d.feat_names}
    groups['pore_mean+throat_mean']=[n for n in ['pore_mean','throat_mean'] if n in d.feat_names]
    groups['all_size_descriptors']=[n for n in d.feat_names if n.startswith(('pore_','throat_','chord_mean','corr_len'))]
    records=[]
    for seed in range(42,42+args.seeds):
        predict,_,_=fit(d,tr,spec['model'],tuning['params'],seed)
        base=predict(te); y=d.Y[te]
        for group,names in groups.items():
            if not names: continue
            cols=[d.feat_names.index(n) for n in names]
            rng=np.random.default_rng(seed)
            for repeat in range(args.repeats):
                perm=rng.permutation(len(te))
                if d.long:
                    arrays=[x[te].copy() for x in d.Xax]
                    for a in range(3): arrays[a][:,cols]=d.Xax[a][te[perm]][:,cols]
                else:
                    arrays=d.X[te].copy(); arrays[:,cols]=d.X[te[perm]][:,cols]
                p=predict(np.arange(len(te)),arrays)
                for a,target in enumerate(C.TARGETS):
                    records.append(dict(seed=seed,group=group,members=';'.join(names),repeat=repeat,target=target,
                        delta_R2_log=float(r2_score(y[:,a],base[:,a])-r2_score(y[:,a],p[:,a]))))
    pd.DataFrame(records).to_csv(job/'importance.csv',index=False,encoding='utf-8-sig')
    print(f'Importance saved for {spec["model"]}: {job}',flush=True)

def no_overlap(d,tr,te):
    total=0
    for core in d.df.core_id.unique():
        a=d.df.iloc[tr];b=d.df.iloc[te]
        x=a[a.core_id.eq(core)][['ox','oy','oz']].to_numpy()
        y=b[b.core_id.eq(core)][['ox','oy','oz']].to_numpy()
        for w in y: total+=int(np.all(np.abs(x-w)<C.WINDOW,axis=1).sum())
    return total

def preflight():
    manifest=json.loads((BASE/'input_manifest.json').read_text(encoding='utf-8'))
    for n,meta in manifest.items():
        if sha(BASE/'data'/n)!=meta['sha256']: raise ValueError(f'Input modified: {n}')
    d=data_for('image13');tr,te,_=d.fold('blocked',0)
    assert d.df[['core_id','ox','oy','oz']].duplicated().sum()==0
    assert np.isfinite(d.Yraw).all() and (d.Yraw>0).all()
    overlap=no_overlap(d,tr,te)
    if overlap: raise ValueError(f'Outer windows overlap: {overlap}')
    for a,b in inner_folds(d,tr):
        assert not set(d.df.iloc[tr[a]].core_id)&set(d.df.iloc[tr[b]].core_id)
    for fold in d.folds_of('loco'):
        lt,le,_=d.fold('loco',fold)
        assert not set(d.df.iloc[lt].core_id)&set(d.df.iloc[le].core_id)
        for a,b in inner_folds(d,lt):
            assert not set(lt[a])&set(le) and not set(lt[b])&set(le)
    counts={name:len(data_for(name).feat_names) for name in EXPERIMENTS}
    expected={'base7':7,'shared7':7,'image13':13,'image19':19,'image13_iso':13,'image13_shuffle':13,'network17':17,'network27':27,'bundle8':8}
    assert counts==expected,(counts,expected)
    report=dict(valid_samples=len(d.df),cores=d.df.core_id.nunique(),blocked_train=len(tr),blocked_test=len(te),
                outer_overlap_pairs=overlap,inner_group='core_id',feature_counts=counts,versions=versions())
    write_json(RESULTS/'preflight.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

def smoke():
    preflight()
    # Validation values cannot change medians; a test-only extreme remains extreme.
    x=np.array([[1.,10.],[3.,np.nan],[5.,30.]])
    tx,_=M.fit_pipeline(x,np.arange(3.)[:,None])
    np.testing.assert_allclose(tx([[3.,np.nan]]),tx([[3.,20.]]))
    try: M.fit_pipeline(np.array([[np.nan],[np.nan]]),np.ones((2,1)))
    except ValueError: pass
    else: raise AssertionError('All-missing training column was not rejected')
    d=data_for('image13');tr,te,_=d.fold('blocked',0)
    params=dict(h1=16,h2=16,h3=8,alpha=.01,learning_rate_init=.003,batch_size=128)
    # Smoke fitting is not a paper result and has a separate output directory.
    predict,elapsed,ws=fit(d,tr[:192],'DNN',params,42)
    p=predict(te[:24]); assert p.shape==(24,3) and np.isfinite(p).all()
    kdict=dict(n_neighbors=3,leaf_size=30,p=2,weights='distance')
    predictor,_,_=fit(data_for('base7'),tr[:192],'KNN',kdict,42)
    assert predictor(te[:24]).shape==(24,3)
    write_json(RESULTS/'smoke.json',dict(passed=True,scope='pipeline checks only; not scientific validation',dnn_fit_sec=elapsed,warnings=ws))
    print('SMOKE PASS',flush=True)

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['preflight','smoke','summary']: sub.add_parser(name)
    for name in ['models','collinearity']:
        p=sub.add_parser(name)
        p.add_argument('--experiments',nargs='+',choices=EXPERIMENTS,default=['base7','image13'])
        p.add_argument('--protocols',nargs='+',choices=['blocked','loco','random','random_sub'],default=['blocked'])
        p.add_argument('--models',nargs='+',choices=C.MODELS,default=['DNN'])
        p.add_argument('--optimizers',nargs='+',choices=C.OPTIMIZERS,default=['SSA'])
        p.add_argument('--budget',type=int,default=90)
        p.add_argument('--seeds',type=int,default=10)
        p.add_argument('--folds',nargs='+',type=int)
    p=sub.add_parser('importance');p.add_argument('--job',required=True)
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    if getattr(args,'seeds',1)<1 or getattr(args,'budget',1)<1: parser.error('Positive counts required')
    if args.command=='preflight': preflight()
    elif args.command=='smoke': smoke()
    elif args.command=='summary': summarize()
    elif args.command=='importance': importance(args)
    else: run_models(args)

if __name__=='__main__': main()
