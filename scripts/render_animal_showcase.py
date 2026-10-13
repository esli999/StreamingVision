#!/usr/bin/env python3
"""Render the four visually reviewed animal sequences from immutable captures."""
import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified.data import Capture, ROOT, write_json
from unified.render import render
from scripts.review_unified import current_video, review_sheet


CHOICES = [
    ('blackswan', 'validation', 'A distinct neck and body make the representation easy to recognize.'),
    ('bear', 'training', 'A readable single-animal body and consistent silhouette over uneven ground.'),
    ('elephant', 'training', 'A single animal with a coherent particle body and readable figure–ground separation.'),
    ('dog', 'validation', 'Walking and turning expose object motion and the shared scene representation.'),
]


def main():
    base = ROOT / 'runs/unified_quality'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-captures', type=Path,
                        default=base / 'validation_final/quality/seed_42')
    parser.add_argument('--training-captures', type=Path,
                        default=base / 'train_occlusion_final/quality/seed_42')
    parser.add_argument('--out-dir', type=Path, default=base / 'animal_percepts')
    parser.add_argument('--width', type=int, default=5120)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cards, entries = [], []
    for name, split, reason in CHOICES:
        folder = getattr(args, f'{split}_captures')
        capture = Capture(folder / f'{name}.capture')
        video = args.out_dir / f'{name}_unified.mp4'
        if not current_video(capture, video, args.width):
            render(capture, video, args.width)
        review_sheet(capture, args.out_dir / f'{name}_review.jpg')
        metric_path = folder / f'{name}.metrics.json'
        metrics = json.loads(metric_path.read_text()) if metric_path.exists() else {}
        jf = metrics.get('jf')
        comparison = f'stability_comparison/{name}_stability.mp4'
        comparison_link = (f' · <a href="{comparison}">Raw / stabilized particles</a>'
                           if (args.out_dir / comparison).is_file() else '')
        cards.append(f'''<article><h2>{html.escape(name.title())}</h2>
<video controls loop playsinline preload="metadata" poster="{name}_unified_00000.jpg"
 src="{video.name}"></video>
<label>Playback <select aria-label="{name} playback speed"><option value="1">1×</option>
<option value="0.5">0.5× for inspection</option><option value="0.25">0.25×</option></select></label>
<p><a href="{video.name}">Open MP4</a> · <a href="{name}_unified_{capture.manifest['frames']//2:05d}.png">Paper figure PNG</a> · <a href="{name}_review_details.jpg">Inspect multiple views</a>{comparison_link}</p>
</article>''')
        entries.append(dict(clip=name, split=split, reason=reason, video=video.name,
                            frames=capture.manifest['frames'], j_and_f=jf,
                            render_report=f'{name}_unified.render.json'))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GenMatter · Four animal scenes</title><style>
body{font:17px/1.55 system-ui,sans-serif;background:white;color:#253448;margin:32px auto;padding:0 24px;max-width:1440px}
h1{font-size:40px;line-height:1.15}h2{margin-bottom:4px}article{margin:48px 0}
video{width:100%;display:block;margin:18px 0}a{color:#176887}select{font:inherit;padding:5px;margin-left:8px}
@media(max-width:600px){body{padding:0 12px}article{padding:12px}h1{font-size:30px}}
</style><h1>GenMatter</h1>'''
    page += ''.join(cards) + '''<script>
document.querySelectorAll('article').forEach(card=>{
  const video=card.querySelector('video');
  card.querySelector('select').addEventListener('change',event=>video.playbackRate=Number(event.target.value));
  video.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause();}));
});
</script></html>'''
    (args.out_dir / 'index.html').write_text(page)
    write_json(args.out_dir / 'selection.json', entries)
    print(args.out_dir / 'index.html')


if __name__ == '__main__':
    main()
