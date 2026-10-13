"""CLI entry point; help and offline rendering need no inference dependencies."""
import argparse
import os
from pathlib import Path
import yaml
from .data import ROOT, DEFAULT_DATA, Capture, resolve_clip, write_json


def configure_runtime():
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ.setdefault(key,'4')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.45')
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(ROOT/'.jax_cache/unified'))


def configuration(preset='quality',path=None):
    data = yaml.safe_load(Path(path or ROOT/'configs/unified.yaml').read_text())
    if not isinstance(data,dict) or not {'common','presets'} <= data.keys():
        raise ValueError('Unified configs require common and presets sections; see configs/unified.yaml')
    config = {**data['common'],**data['presets'][preset]}
    if config['sampling_stride'] not in (4,8):
        raise ValueError('sampling_stride must be 4 or 8')
    if any(config[k] <= 0 or config[k]%14 for k in ('dino_h','dino_w')):
        raise ValueError('DINO dimensions must be positive multiples of 14')
    if not 0 < config['foreground_fraction'] < 1 or config['sweeps'] < 1 or config['particles'] < 2:
        raise ValueError('Invalid foreground fraction, sweep count, or particle count')
    if any(config.get(k,1.) <= 0 for k in ('mean_prior_count','covariance_prior_count','particle_weight_prior')):
        raise ValueError('Geometry and mixture prior counts must be positive')
    defaults = data.get('custom_foreground_ids')
    if defaults is None:
        defaults = yaml.safe_load((ROOT/'configs/unified.yaml').read_text())['custom_foreground_ids']
    return config, defaults


def main(argv=None):
    parser = argparse.ArgumentParser(description='Causal GenMatter inference and reusable offline particle figures')
    parser.add_argument('--videos',nargs='+',default=['test'])
    parser.add_argument('--source',type=Path,help='Single MP4/MOV or image directory; requires an initial mask')
    parser.add_argument('--preset',choices=['quality','fast'],default='quality')
    parser.add_argument('--stage',choices=['all','track','render','benchmark'],default='all')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'runs/unified_quality/output')
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--capture',type=Path,help='Explicit saved capture for --stage render')
    parser.add_argument('--seed-mask',type=Path)
    parser.add_argument('--foreground-ids',type=int,nargs='+')
    parser.add_argument('--source-fps',type=float,default=30.)
    parser.add_argument('--max-frames',type=int,default=-1)
    parser.add_argument('--target-duration',type=float,default=0.,help='Legacy input-duration limit; 0 means complete source')
    parser.add_argument('--width',type=int,default=5120,help='Output width, aspect ratio 32:13')
    parser.add_argument('--representation',choices=['mesh','particles'],default='mesh',help='Offline readout of the same captured particle state')
    parser.add_argument('--raw-particles',action='store_true',help='Disable display geometry stabilization; offset views are checked against the raw geometry')
    args = parser.parse_args(argv)
    if args.capture:
        if args.stage != 'render':
            parser.error('--capture is only used with --stage render')
        args.videos = [Capture(args.capture).manifest['clip']]
    if args.source:
        args.videos = [str(args.source.resolve())]
    if args.seed_mask and len(args.videos) != 1:
        parser.error('--seed-mask applies to a single input')
    if args.source_fps <= 0 or args.width < 640 or args.width % 32:
        parser.error('--source-fps must be positive; --width must be >=640 and a multiple of 32')
    config, defaults = configuration(args.preset,args.config)
    if any(v in ('all-davis','all-custom','all') for v in args.videos):
        expanded = []
        for video in args.videos:
            if video in ('all-davis','all'):
                split = args.data_root/'ImageSets/2017/val.txt'
                if not split.is_file():
                    parser.error('all-davis requires the DAVIS corpus under --data-root')
                expanded.extend(split.read_text().split())
            if video in ('all-custom','all'):
                expanded.extend(defaults)
            if video not in ('all-davis','all-custom','all'):
                expanded.append(video)
        args.videos = list(dict.fromkeys(expanded))
    configure_runtime()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    models = None
    for video in args.videos:
        name = Path(video).stem
        capture_path = args.capture or args.out_dir/f'{name}_{args.preset}.capture'
        if args.stage in ('all','track','benchmark'):
            from .models import Models
            from .tracker import track
            ids = args.foreground_ids if args.seed_mask else (args.foreground_ids or defaults.get(name))
            clip = resolve_clip(video,args.data_root,args.seed_mask,ids,args.source_fps)
            if models is None:
                models = Models(config)
            limit = args.max_frames
            if limit < 0 and args.target_duration > 0:
                limit = max(1,round(args.target_duration*clip.fps))
            summary = track(clip,config,capture_path,limit,models,args.stage != 'benchmark')
            write_json(args.out_dir/f'{name}_{args.preset}_{args.stage}.json',summary)
        if args.stage in ('all','render'):
            from .render import render
            render(Capture(capture_path),args.out_dir/f'{name}_unified.mp4',args.width,args.representation,not args.raw_particles)


if __name__ == '__main__':
    main()
