#!/usr/bin/env python3
"""Sequential inference-only benchmarks, including the discarded-draw ablation."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified.cli import configuration,configure_runtime
from unified.data import ROOT,resolve_clip,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',default='test')
    p.add_argument('--presets',nargs='+',default=['quality','fast'],choices=['quality','fast'])
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--compare-conditionals',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'runs/unified_quality/benchmark.json')
    args=p.parse_args();configure_runtime()
    if args.repeat < 1:
        p.error('--repeat must be positive')
    import jax
    import genmatter_rt as rt
    from unified.models import Models
    from unified.tracker import track
    optimized=rt._last_independent_update
    modes=['optimized','all_draws'] if args.compare_conditionals else ['optimized']
    records=[]
    for preset in args.presets:
        config,ids=configuration(preset)
        models=Models(config)
        for mode in modes:
            if mode=='all_draws':
                rt._last_independent_update=lambda key,state,update,count:jax.lax.fori_loop(0,count,update,(key,state))
            else:
                rt._last_independent_update=optimized
            # Globals are captured at JIT compile; force the intended schedule.
            rt._MULTI_SWEEP_CACHE.clear()
            jax.clear_caches()
            for repeat in range(args.repeat):
                load_before=os.getloadavg()
                try:
                    gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                        '--format=csv,noheader'],text=True).split()
                    competing=[int(p) for p in gpu_pids if p.isdigit() and int(p)!=os.getpid()]
                except (OSError,subprocess.CalledProcessError):
                    competing=None  # optional telemetry must not prevent CUDA inference
                clip=resolve_clip(args.video,foreground_ids=ids.get(args.video))
                result=track(clip,config,args.out.parent/'unused',models=models,save_states=False)
                records.append(dict(preset=preset,mode=mode,repeat=repeat,
                    host_load_before=load_before,host_load_after=os.getloadavg(),
                    other_cuda_pids_at_start=competing,**result))
                write_json(args.out,dict(records=records,
                    method='sequential whole-clip runs, no rendering or state serialization; model/JIT initialization reported separately'))
                print(preset,mode,repeat,result['inference_fps'],flush=True)
        rt._last_independent_update=optimized


if __name__=='__main__':
    main()
