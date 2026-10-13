#!/usr/bin/env python3
"""Build a local, inspectable MP4 gallery and review sheets from saved states."""
import argparse
import html
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from unified.data import Capture,write_json,fingerprint,implementation_fingerprint
from unified.render import compose,render,text,figure_boxes,CANVAS_SIZE
from unified.presentation import Presentation

AXES=['foreground_readability','shading','motion_stability','composition','novel_view_clarity','text_readability']


def review_signature(capture):
    return dict(capture_manifest_sha256=fingerprint(capture.path/'manifest.json'),
                renderer_implementation_sha256=implementation_fingerprint())


def current_video(capture,path,width):
    report=path.with_suffix('.render.json')
    if not path.is_file() or not report.is_file():
        return False
    saved=json.loads(report.read_text())
    return (all(saved.get(k)==v for k,v in review_signature(capture).items())
            and saved.get('width')==width and saved.get('representation')=='mesh'
            and saved.get('geometry_filter_enabled') is True)


def review_sheet(capture,path):
    signature=dict(**review_signature(capture),review_tool_sha256=fingerprint(Path(__file__)))
    video=path.parent/f"{capture.manifest['clip']}_unified.mp4"
    encoded=current_video(capture,video,CANVAS_SIZE[0])
    if encoded: signature['encoded_video_sha256']=fingerprint(video)
    stamp=path.with_suffix('.review.json')
    if path.is_file() and path.with_name(path.stem+'_details.jpg').is_file() and stamp.is_file():
        saved=json.loads(stamp.read_text())
        if all(saved.get(k)==v for k,v in signature.items()):
            return saved['indices']
    presentation=None if encoded else Presentation(capture)
    ground_available=(json.loads(video.with_suffix('.render.json').read_text())['ground_surface_shown']
                      if encoded else presentation.ground_available)
    def frame_image(i):
        if encoded:
            decoder=cv2.VideoCapture(str(video))
            try:
                decoder.set(cv2.CAP_PROP_POS_FRAMES,i)
                ok,bgr=decoder.read()
                if not ok: raise RuntimeError(f'Cannot decode review frame {i}: {video}')
                return Image.fromarray(bgr[...,::-1])
            finally: decoder.release()
        return compose(presentation.frame(capture.read(i),i),presentation.manifest,
                       presentation.targets[i],presentation.radius,i)
    n=capture.manifest['frames']
    # Include the largest observed particle-motion transition as well as evenly
    # spaced frames, so a calm opening cannot conceal later unstable states.
    jumps=[];previous=None
    for i in range(n):
        frame=capture.read(i)
        if previous is not None:
            use=(frame['object_ids']>0)&(frame.get('support',frame['weights']*2925)>2)
            jump=float(np.percentile(np.linalg.norm(frame['means'][use]-previous['means'][use],axis=1),95)) if use.any() else 0.
            jumps.append((jump,i))
        previous=frame
    indices=set(np.linspace(0,n-1,min(n,6)).round().astype(int).tolist())
    for _,i in sorted(jumps,reverse=True)[:2]:
        indices.update([max(i-1,0),i])
    indices=sorted(indices)
    overview_height=round(1040*CANVAS_SIZE[1]/CANVAS_SIZE[0])
    row_height=overview_height+40
    sheet=Image.new('RGB',(2080,row_height*((len(indices)+1)//2)),(238,242,246))
    draw=ImageDraw.Draw(sheet)
    detail_indices=sorted(set(np.linspace(0,n-1,min(n,5)).round().astype(int).tolist()
                              + [max(jumps)[1] if jumps else 0]))
    boxes=figure_boxes(ground_available)
    keys=['input','scene','surface']
    if ground_available: keys.append('ground')
    keys+=['segmentation','observed','novel_left','novel']
    crops=[boxes[k] for k in keys]
    widths=[600 if k=='scene' else 320 for k in keys]
    details=Image.new('RGB',(sum(widths),454*len(detail_indices)),(238,242,246))
    detail_draw=ImageDraw.Draw(details)
    caption=('input / particles / deformable object / '+('background / ' if ground_available else '')+
             'segmentation / initial rendering / pose 1 rendering / pose 2 rendering')
    for j,i in enumerate(indices):
        original=frame_image(i)
        image=original.resize((1040,overview_height),Image.Resampling.LANCZOS)
        x,y=(j%2)*1040,(j//2)*row_height
        sheet.paste(image,(x,y))
        text(draw,(x+10,y+overview_height+5),f'Frame {i+1} / {n}',18)
        if i in detail_indices:
            row=detail_indices.index(i);x=0;y=row*454
            for box,width in zip(crops,widths):
                tile=original.crop(box);tile.thumbnail((width,420),Image.Resampling.LANCZOS)
                details.paste(tile,(x+(width-tile.width)//2,y+(420-tile.height)//2));x+=width
            text(detail_draw,(10,y+424),f"{capture.manifest['clip']} · frame {i+1}/{n} · {caption}",20)
    sheet.save(path,quality=91)
    # A detail candidate can be absent from the broader overview's rounded grid.
    for i in set(detail_indices)-set(indices):
        row=detail_indices.index(i);x=0;y=row*454
        original=frame_image(i)
        for box,width in zip(crops,widths):
            tile=original.crop(box);tile.thumbnail((width,420),Image.Resampling.LANCZOS)
            details.paste(tile,(x+(width-tile.width)//2,y+(420-tile.height)//2));x+=width
        text(detail_draw,(10,y+424),f"{capture.manifest['clip']} · frame {i+1}/{n} · {caption}",20)
    details.save(path.with_name(path.stem+'_details.jpg'),quality=94)
    write_json(stamp,dict(**signature,indices=indices))
    return indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--render',action='store_true')
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--reviews',type=Path,help='Completed review JSON to display in the gallery')
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    reviews={r['clip']:r for r in json.loads(args.reviews.read_text())} if args.reviews else {}
    entries=[]
    for folder in sorted(args.captures.glob('*.capture')):
        c=Capture(folder);name=c.manifest['clip']
        video=args.out_dir/f'{name}_unified.mp4'
        if args.render and not current_video(c,video,args.width):
            render(c,video,args.width)
        indices=review_sheet(c,args.out_dir/f'{name}_review.jpg')
        metric_path=folder.with_name(name+'.metrics.json')
        metrics=json.loads(metric_path.read_text()) if metric_path.exists() else {}
        entries.append(dict(clip=name,video=video.name,review_sheet=f'{name}_review.jpg',
                            detail_sheet=f'{name}_review_details.jpg',
                            frames=c.manifest['frames'],reviewed_frame_candidates=indices,metrics=metrics,
                            scores={k:None for k in AXES},notes='',review_status='pending',featured=False))
        if name in reviews:
            review=reviews[name]
            scores=review['scores']
            if set(scores) != set(AXES) or any(not isinstance(v,(int,float)) or not 1 <= v <= 5 for v in scores.values()):
                raise ValueError(f'{name}: each aesthetic dimension requires an actual 1–5 score')
            featured=bool(review.get('featured',False))
            if featured and (np.mean(list(scores.values())) < 4 or min(scores.values()) < 3):
                raise ValueError(f'{name}: featured clips require mean >=4 and every dimension >=3')
            entries[-1].update(scores=scores,notes=review.get('notes',''),
                               review_status=review.get('review_status','reviewed'),featured=featured)
        print(f'Review sheet: {name}',flush=True)
    write_json(args.out_dir/'review_template.json',entries)
    cards=[]
    for e in sorted(entries,key=lambda e:(not e['featured'],e['clip'])):
        name=html.escape(e['clip']);video=html.escape(e['video']);sheet=html.escape(e['review_sheet'])
        jf=e['metrics'].get('jf')
        metric=f'J&F {jf*100:.1f}' if jf is not None else 'Custom clip · no held-out annotation score'
        if e['featured']:
            metric='Featured · '+metric
        review=''
        if e['review_status'] != 'pending':
            scores=e['scores']
            mean=float(np.mean(list(scores.values())))
            dimensions=' · '.join(f'{k.replace("_"," ")}: {v:g}/5' for k,v in scores.items())
            review=f'<p>Aesthetic review: {mean:.2f}/5</p><p>{html.escape(dimensions)}</p><p>{html.escape(e["notes"])}</p>'
        available=(args.out_dir/e['video']).is_file()
        media=(f'<video controls preload="metadata" poster="{name}_unified_00000.jpg" src="{video}"></video>'
               if available else f'<img loading="lazy" alt="{name} review sheet" src="{sheet}">')
        download=f' · <a href="{video}">Download MP4</a>' if available else ''
        cards.append(f'<article data-name="{name.lower()}"><h2>{name}</h2><p>{e["frames"]} frames · {metric}</p>'
                     +media+review+
                     f'<p><a href="{sheet}">Figure overview</a> · <a href="{html.escape(e["detail_sheet"])}">Inspect detail sheet</a>'
                     +download+'</p></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>StreamingVision validation gallery</title><style>
body{font:16px system-ui;margin:36px;background:#f6f8fb;color:#223044}h1{font-size:36px}
input{font:inherit;padding:12px;border:1px solid #bbc9d5;border-radius:8px;margin:12px 0 24px;width:min(480px,90%)}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px}
article{background:white;padding:20px;border:1px solid #dfe5ec;border-radius:14px}video,img{width:100%}a{color:#156ea1}
@media(max-width:600px){main{display:block}article{margin-bottom:20px}body{margin:16px}}
</style><h1>StreamingVision validation gallery</h1><p>Complete captured sequences. Seed-frame masks only; later labels are inferred.
Novel views reproject observed relative-depth geometry. Inference speed and playback rate are labeled separately.</p>
<label for="filter">Find a clip</label><br><input id="filter" type="search" placeholder="Clip name" autocomplete="off">
<main>'''+''.join(cards)+'''</main><script>
document.getElementById('filter').addEventListener('input',function(){
  const query=this.value.toLowerCase().trim();
  document.querySelectorAll('article').forEach(card=>card.hidden=!card.dataset.name.includes(query));
});
</script></html>'''
    (args.out_dir/'index.html').write_text(page)


if __name__=='__main__':
    main()
