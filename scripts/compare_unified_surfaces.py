#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,time,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2
from PIL import Image,ImageDraw
from unified.data import Capture,write_json,fingerprint,implementation_fingerprint
from unified.presentation import Presentation
from unified.geometry import orbit_camera
from unified.render import splat_particles,text
from unified.surface import mesh_surfaces,raster_surfaces

def main():
    parser=argparse.ArgumentParser(description='Compare particle ellipsoids and a density mesh on identical captured states and cameras.')
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--clips',nargs='+',default=['test','blackswan','bear','cat-girl','drift-turn','horsejump-low'])
    args=parser.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for name in args.clips:
     c=Capture(args.captures/f'{name}.capture');p=Presentation(c)
     spacing=p.surface_spacing
     process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x400','-r',str(p.fps),'-i','-','-an','-c:v','libx264','-crf','18','-threads','2','-pix_fmt','yuv420p',str(out/f'{name}_comparison.mp4')],stdin=subprocess.PIPE)
     sheet=Image.new('RGB',(1440,170*((len(p.means)+3)//4)),'white');d=ImageDraw.Draw(sheet)
     last=None;diffs=[[],[]];triangles=[];times=[]
     for i in range(len(p.means)):
      f=p.frame(c.read(i),i);cam=orbit_camera(p.targets[i],14,-9,p.radius*2.65,500.,640,350)
      start=time.perf_counter();surfaces=mesh_surfaces(f,spacing);mesh=raster_surfaces(surfaces,cam,near=p.radius*.25)
      times.append(time.perf_counter()-start);triangles.append(sum(len(s.faces) for s in surfaces))
      cloud=splat_particles(f,cam,near=p.radius*.25,background_alpha=0)
      ims=[cloud,mesh]
      if last is not None:
       region=np.any(np.stack([*last,*ims])<230,axis=(0,3))
       for k in range(2):diffs[k].append(float(np.abs(ims[k].astype(float)-last[k])[region].mean()))
      last=ims
      video=Image.new('RGB',(1280,400),'white');vd=ImageDraw.Draw(video)
      for k,im in enumerate(ims):video.paste(Image.fromarray(im),(640*k,45));text(vd,(640*k+15,10),f'{name} · frame {i+1} · '+['ellipsoids','density mesh'][k],20)
      process.stdin.write(video.tobytes())
      x=(i%4)*360;y=(i//4)*170
      pair=np.concatenate(ims,axis=1);tile=Image.fromarray(pair);tile.thumbnail((360,135));sheet.paste(tile,(x,y+25));text(d,(x+4,y+3),str(i+1),18)
     process.stdin.close();assert process.wait()==0
     sheet.save(out/f'{name}_consecutive.jpg',quality=94)
     reports.append(dict(clip=name,frames=len(p.means),spacing=spacing,
            capture_path=str(c.path.resolve()),capture_manifest_sha256=fingerprint(c.path/'manifest.json'),
            renderer_implementation_sha256=implementation_fingerprint(),playback_fps=p.fps,
            mean_triangle_count=float(np.mean(triangles)),mesh_seconds_per_frame=float(np.mean(times)),raw_pixel_delta=dict(ellipsoid_mean=float(np.mean(diffs[0])),mesh_mean=float(np.mean(diffs[1])),ellipsoid_p95=float(np.percentile(diffs[0],95)),mesh_p95=float(np.percentile(diffs[1],95))),note='Same geometry and camera; raw foreground pixel differences include true motion and are not a tracking or perceptual jitter score.'))
     write_json(out/'comparison.json',reports);print(reports[-1],flush=True)


if __name__=='__main__':
    main()
