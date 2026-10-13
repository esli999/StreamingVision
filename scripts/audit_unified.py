#!/usr/bin/env python3
"""Fully decode rendered MP4s and verify them against their inference captures."""
import argparse
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified.data import Capture,fingerprint,write_json
from unified.presentation import playback_fps


@lru_cache(maxsize=None)
def source_frames(path):
    source=Path(path)
    if source.is_dir():
        return sum(p.suffix.lower() in ('.jpg','.jpeg','.png') for p in source.iterdir() if p.is_file())
    result=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames','-of','json',str(source)],
        capture_output=True,text=True,check=True)
    if result.stderr.strip():
        raise ValueError(result.stderr.strip())
    return int(json.loads(result.stdout)['streams'][0]['nb_read_frames'])


def audit(video,allow_partial=False):
    report=json.loads(video.with_suffix('.render.json').read_text())
    capture=Capture(report['capture_path'])
    if fingerprint(capture.path/'manifest.json') != report['capture_manifest_sha256']:
        raise ValueError('Capture manifest changed after rendering')
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,duration,nb_frames',
        '-of','json',str(video)],capture_output=True,text=True,check=True)
    stream=json.loads(probe.stdout)['streams'][0]
    decoded=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-xerror',
        '-err_detect','explode','-threads','2','-i',str(video),'-map','0:v:0',
        '-filter_threads','1','-f','null','-progress','pipe:1','-nostats','-'],
        capture_output=True,text=True,check=True)
    if decoded.stderr.strip():
        raise ValueError(decoded.stderr.strip())
    frames=int([line.split('=',1)[1] for line in decoded.stdout.splitlines()
                if line.startswith('frame=')][-1])
    fps=float(Fraction(stream['avg_frame_rate']))
    expected=capture.manifest['frames']
    available=source_frames(capture.manifest['source'])
    if expected != available and not allow_partial:
        raise ValueError(f'Capture has {expected} frames but source contains {available}; use --allow-partial for an intentional debug clip')
    if frames != expected or int(stream['nb_frames']) != expected:
        raise ValueError(f'Decoded {frames} frames; capture contains {expected}')
    if (stream['width'],stream['height']) != (report['width'],report['height']):
        raise ValueError('Encoded dimensions differ from render report')
    if abs(fps-report['source_fps']) > .001:
        raise ValueError('Playback rate differs from source rate')
    timestamps=[]
    for path in sorted(capture.path.glob('frame_*.npz')):
        with np.load(path,allow_pickle=False) as frame:
            timestamps.append(float(frame['source_time']))
    decoded_rate=playback_fps(timestamps,capture.manifest['source_fps'])
    if abs(fps-decoded_rate)>.001:
        raise ValueError('Playback rate differs from decoded source timestamps')
    duration=float(stream['duration'])
    if abs(duration-expected/fps) > .01:
        raise ValueError('Encoded duration differs from the complete frame sequence')
    return dict(video=str(video.resolve()),frames=frames,width=stream['width'],
        height=stream['height'],playback_fps=fps,duration_seconds=duration,
        sha256=fingerprint(video),capture=str(capture.path.resolve()),
        source_frames=available,complete_source=expected==available,
        complete_decode=True,capture_manifest_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders',nargs='+',type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--allow-partial',action='store_true',help='Permit intentional --max-frames debug captures')
    args=p.parse_args()
    videos=sorted({v for folder in args.folders for v in folder.glob('*_unified.mp4')})
    if not videos:
        p.error('No *_unified.mp4 videos found')
    results=[]
    for video in videos:
        try:
            result=audit(video,args.allow_partial)
            print(f"Verified {video.name}: {result['frames']} frames",flush=True)
        except Exception as exc:
            result=dict(video=str(video),error=f'{type(exc).__name__}: {exc}')
            print(result,flush=True)
        results.append(result)
        write_json(args.out,dict(videos=len(results),passed=sum('error' not in r for r in results),
            method='Full FFmpeg decode with errors fatal; dimensions, frame count, rate, duration and capture identity checked',
            results=results))
    if any('error' in r for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
