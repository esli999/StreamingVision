#!/usr/bin/env python3
"""Reproducible train-only selection and held-out DAVIS/custom validation."""
import argparse
import os
import sys
import traceback
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from unified.cli import configuration,configure_runtime
from unified.data import Capture,DEFAULT_DATA,ROOT,resolve_clip,write_json,fingerprint,implementation_fingerprint
from unified.metrics import evaluate_capture

SCREEN = ['bear','cat-girl','dog-agility','drift-turn','horsejump-low','mallard-fly']
CUSTOM = ['test','gray_jacket','purple_jacket','jello_trim','new_eagle_trim','snake_trim','two_blocks_centered','wine_swirl']


def candidates():
    quality,_ = configuration('quality')
    fast,_ = configuration('fast')
    cases = {'quality_start':quality,'fast_start':fast}
    variants = dict(one_sweep=dict(sweeps=1),four_sweeps=dict(sweeps=4),
        small_dense=dict(dino_h=504,dino_w=896),
        base=dict(dino_checkpoint='facebook/dinov2-base'),
        registers=dict(dino_checkpoint='facebook/dinov2-with-registers-base'),
        tight_features=dict(feature_variance=.4),loose_features=dict(feature_variance=1.2),
        strong_memory=dict(memory_weight=30.,persistence=8.),
        weak_memory=dict(memory_weight=10.,persistence=3.),
        more_foreground=dict(foreground_fraction=.78),
        adaptive_appearance=dict(appearance_update=.25),
        robust_motion=dict(huber_delta=2.),
        strong_geometry_prior=dict(mean_prior_count=3.,covariance_prior_count=8.),
        weak_geometry_prior=dict(mean_prior_count=.75,covariance_prior_count=2.))
    cases.update({k:{**quality,**v} for k,v in variants.items()})
    return cases


def aggregate(results):
    valid = [r for r in results if 'jf' in r]
    good = [r for r in results if not r.get('error')]
    timed = sum(r.get('timed_frames',0) for r in good)
    seconds = sum(r.get('inference_seconds',0) for r in good)
    return dict(clips=len(results),unique_clips=len({r['clip'] for r in results}),
                failures=[r['clip'] for r in results if r.get('error')],
                jf=float(np.mean([r['jf'] for r in valid])) if valid else None,
                j=float(np.mean([r['j'] for r in valid])) if valid else None,
                f=float(np.mean([r['f'] for r in valid])) if valid else None,
                inference_fps=timed/seconds if seconds else None,
                aggregation='macro average over clips (per-clip macro over objects); throughput is total completed frames / total time')


def run_case(name,config,clips,out,data_root,limit,seeds):
    from unified.models import Models
    from unified.tracker import track
    _,defaults = configuration()
    models = None
    results = []
    for seed in seeds:
        current = {**config,'seed':seed}
        for video in clips:
            folder = out/name/f'seed_{seed}'
            path = folder/f'{video}.capture'
            metric_path = folder/f'{video}.metrics.json'
            try:
                clip = resolve_clip(video,data_root,foreground_ids=defaults.get(video))
                if path.exists() and (path/'manifest.json').exists():
                    capture = Capture(path)
                    if capture.manifest['config'] != current:
                        raise ValueError('Existing capture has different config; use a new output directory')
                    if capture.manifest.get('inference_implementation_sha256') != implementation_fingerprint('inference'):
                        raise ValueError('Existing capture has different inference code; use a new output directory')
                    if (capture.manifest['source_sha256'] != fingerprint(clip.source)
                        or capture.manifest['initial_mask_sha256'] != fingerprint(clip.mask)):
                        raise ValueError('Existing capture has different input data; use a new output directory')
                else:
                    if models is None:
                        models = Models(current)
                    track(clip,current,path,limit,models)
                    capture = Capture(path)
                scores = evaluate_capture(capture,clip.annotations)
                scores.update(seed=seed,timed_frames=capture.manifest['timed_frames'],
                              inference_seconds=capture.manifest['inference_seconds'])
                write_json(metric_path,scores)
            except Exception as exc:
                traceback.print_exc()
                scores = dict(clip=video,seed=seed,error=f'{type(exc).__name__}: {exc}')
                write_json(metric_path,scores)
                print(scores,flush=True)
            results.append(scores)
            summary = dict(case=name,config=config,seeds=seeds,results=results,**aggregate(results))
            write_json(out/name/'summary.json',summary)
            print(f"{name}: {len(results)}/{len(clips)*len(seeds)} evaluated; J&F={summary['jf']}",flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['screen','finalists','validate'],required=True)
    p.add_argument('--cases',nargs='+')
    p.add_argument('--selected-config',type=Path)
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--out-dir',type=Path,default=ROOT/'runs/unified_quality/evaluation')
    p.add_argument('--max-frames',type=int,default=-1)
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    args=p.parse_args();configure_runtime()
    splits = {s:(args.data_root/f'ImageSets/2017/{s}.txt').read_text().split() for s in ('train','val')}
    assert not (set(splits['train']) & set(splits['val']))
    clips = SCREEN if args.mode=='screen' else splits['train'] if args.mode=='finalists' else splits['val']+CUSTOM
    if args.mode=='screen':
        assert set(clips) <= set(splits['train'])
    cases=candidates() if args.mode=='screen' else {name:configuration(name)[0] for name in ('quality','fast')}
    if args.selected_config:
        data=json.loads(args.selected_config.read_text())
        cases=data['configs']
    names=args.cases or list(cases)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.out_dir/'protocol.json',dict(mode=args.mode,clips=clips,seeds=args.seeds,max_frames=args.max_frames,
        train=splits['train'],val=splits['val'],selection='training J&F; fast within 0.01 of best quality, then maximize throughput',
        annotation_policy='Only frame-0 mask is passed to inference; later annotations read after capture completion'))
    summaries=[]
    for name in names:
        summaries.append(run_case(name,cases[name],clips,args.out_dir,args.data_root,args.max_frames,args.seeds))
        write_json(args.out_dir/'summary.json',summaries)
    if any(s['failures'] for s in summaries):
        raise SystemExit(1)


if __name__=='__main__':
    main()
