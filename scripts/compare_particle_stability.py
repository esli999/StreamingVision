#!/usr/bin/env python3
"""Render raw/flow-stabilized particles with identical cameras, RGB and opacity."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from unified.data import Capture,write_json,fingerprint,implementation_fingerprint
from unified.presentation import Presentation
from unified.render import scaled_camera,paste_fit,text
from unified.surface import particle_surfaces,raster_surfaces


def compare(path,out):
    capture=Capture(path);p=Presentation(capture);name=capture.manifest['clip']
    video=out/f'{name}_stability.mp4'
    process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1600x1000','-r',str(p.fps),'-i','-','-an','-c:v','libx264','-crf','17',
        '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    changes=[[],[]];previous=[None,None]
    sheet=None;pages=[]
    try:
        for i in range(capture.manifest['frames']):
            raw=capture.read(i)
            image=Image.new('RGB',(1600,1000),'white');draw=ImageDraw.Draw(image)
            if i%12==0:
                sheet=Image.new('RGB',(1800,1200),'white');sd=ImageDraw.Draw(sheet)
            for k,enabled in enumerate([False,True]):
                f=p.frame(raw,i,stabilized=enabled)
                geometry=particle_surfaces(f)
                scene=raster_surfaces(geometry,scaled_camera(f['scene_camera'],760/p.scene_camera.width),semantic=False)
                detail=raster_surfaces([s for s in geometry if s.object_id>0],
                                      scaled_camera(p.detail_camera,720/p.detail_camera.width),semantic=False)
                paste_fit(image,scene,(k*800+20,50,760,450))
                paste_fit(image,detail,(k*800+40,530,720,420))
                text(draw,(k*800+25,15),'Flow-stabilized display' if enabled else 'Raw particle geometry',25,bold=True)
                text(draw,(k*800+25,505),'Object detail · same particles',19)
                if previous[k] is not None:
                    visible=np.any(detail<235,axis=2)|np.any(previous[k]<235,axis=2)
                    changes[k].append(float(np.abs(detail.astype(float)-previous[k])[visible].mean()) if visible.any() else 0.)
                previous[k]=detail
            text(draw,(25,968),f'{name} · {i+1}/{capture.manifest["frames"]} · identical cameras / RGB / visibility',19)
            process.stdin.write(image.tobytes())
            # Consecutive object-detail pairs reveal shape/position reversals.
            pair=image.crop((0,530,1600,950)).resize((600,158),Image.Resampling.LANCZOS)
            context=image.crop((0,50,1600,500)).resize((600,169),Image.Resampling.LANCZOS)
            x=(i%3)*600;y=((i%12)//3)*300
            text(sd,(x+10,y+10),f'{name} · frame {i+1} · raw / stabilized',20)
            sheet.paste(context.resize((600,119),Image.Resampling.LANCZOS),(x,y+35))
            sheet.paste(pair.resize((600,140),Image.Resampling.LANCZOS),(x,y+157))
            if (i+1)%12==0 or i+1==capture.manifest['frames']:
                page=out/f'{name}_consecutive_{i//12:02d}.jpg';sheet.save(page,quality=94);pages.append(page.name)
            if (i+1)%20==0: print(f'compare {name}: {i+1}/{capture.manifest["frames"]}',flush=True)
        process.stdin.close()
        if process.wait()!=0: raise RuntimeError('FFmpeg comparison encode failed')
    except BaseException:
        process.kill();process.wait();raise
    report=dict(clip=name,video=str(video),frames=capture.manifest['frames'],source_fps=p.fps,
        capture_manifest_sha256=fingerprint(path/'manifest.json'),renderer_implementation_sha256=implementation_fingerprint(),
        geometry=p.diagnostics['geometry_stability'],sheets=pages,
        pixel_delta={key:dict(mean=float(np.mean(values)) if values else None,p95=float(np.percentile(values,95)) if values else None)
            for key,values in zip(['raw','stabilized'],changes)},
        note='Only displayed means/covariances differ; cameras, RGB and visibility match. Pixel changes include real motion, not tracking accuracy.')
    write_json(out/f'{name}_stability.json',report)
    print(f'Saved {video}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures',nargs='+',type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    results=[compare(path,args.out_dir) for path in args.captures]
    write_json(args.out_dir/'stability_comparison.json',results)


if __name__=='__main__': main()
