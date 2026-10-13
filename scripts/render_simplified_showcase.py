#!/usr/bin/env python3
"""Recompose the approved movies into the separate screenshot-style slide layout.

The existing particle animation, camera poses, RGB projections and measured FPS
are retained. Only panel placement and the visible set of readouts change.
"""
import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified.data import ROOT, fingerprint, write_json
from unified.render import (
    CANVAS_SIZE as SOURCE_SIZE, CONTENT_BOXES, DEFAULT_COLOR, FONT_ROOT, INK, VIEW_COLORS,
    WHITE, arrow, text,
)


CANVAS_SIZE = (4096, 2080)
CLIPS = ('blackswan', 'bear', 'elephant', 'dog', 'test')
BOXES = {
    'input': (160, 432, 1184, 1008),
    'scene': (1600, 200, 4040, 1200),
    # Match the left-to-right order of the actual frustums in the particle view.
    'novel': (1096, 1512, 1992, 2016),
    'observed': (2120, 1512, 3016, 2016),
    'novel_left': (3144, 1512, 4040, 2016),
}


def compose(source, fps, yaws):
    if source.size != SOURCE_SIZE:
        raise ValueError(f'Expected an approved {SOURCE_SIZE} figure, got {source.size}')
    image = Image.new('RGB', CANVAS_SIZE, WHITE)
    draw = ImageDraw.Draw(image)
    for key, box in BOXES.items():
        tile = source.crop(CONTENT_BOXES[key])
        size = (box[2] - box[0], box[3] - box[1])
        if tile.size != size:
            tile = tile.resize(size, Image.Resampling.LANCZOS)
        image.paste(tile, box[:2])
    text(draw, (160, 328), 'Input video', 72, INK, True)
    text(draw, (1600, 104), '3D particle-based matter representation', 72, INK, True)
    text(draw, (1224, 452), 'Real-time', 62, INK, True)
    text(draw, (1224, 532), 'inference', 62, INK, True)
    text(draw, (1260, 612), f'{fps:.1f} FPS' if np.isfinite(fps) else '— FPS', 56, INK, True)
    arrow(draw, (1220, 742), (1570, 742), (91, 111, 128), 44)

    heading = '2D renderings from the 3D particle model'
    # Center the heading over the three camera-colored views.
    font = ImageFont.truetype(str(FONT_ROOT / 'DejaVuSans-Bold.ttf'), 72)
    x = (BOXES['novel'][0] + BOXES['novel_left'][2] - draw.textlength(heading, font=font)) / 2
    text(draw, (x, 1224), heading, 72, INK, True)
    for key, label, tint, yaw in (
        ('novel', 'Reposed camera 2', VIEW_COLORS[1], yaws[1]),
        ('observed', 'Initial camera pose', DEFAULT_COLOR, None),
        ('novel_left', 'Reposed camera 1', VIEW_COLORS[0], yaws[0]),
    ):
        x, y, right, bottom = BOXES[key]
        text(draw, (x, 1436 if yaw is None else 1376), label, 62, tint, True)
        if yaw is not None:
            offset = f'{abs(yaw):g}° ' + ('right' if yaw < 0 else 'left')
            text(draw, (x, 1454), offset, 42, tint, True)
        draw.rectangle((x - 4, y - 4, right + 4, bottom + 4), outline=tint, width=7)
    return image


def source_video(name):
    folder = 'percept_debug' if name == 'test' else 'animal_percepts'
    return ROOT / 'runs/unified_quality' / folder / f'{name}_unified.mp4'


def read_metadata(video):
    report = json.loads(video.with_suffix('.render.json').read_text())
    with np.load(video.with_suffix('.cameras.npz')) as state:
        rates = state['rollout_fps'].copy()
    measured = rates[np.isfinite(rates)]
    rates[~np.isfinite(rates)] = measured[0] if len(measured) else np.nan
    if len(rates) != report['frames']:
        raise ValueError(f'FPS trace and frame count disagree: {video}')
    return report, rates


def save_still(video, output, report, rates, index):
    # The source lossless still avoids a second video compression in paper figures.
    path = video.with_name(f'{video.stem}_{index:05d}.png')
    with Image.open(path) as source:
        image = compose(source, rates[index], report['novel_view_degrees'])
    image.save(output.with_name(f'{output.stem}_{index:05d}.png'))
    image.save(output.with_name(f'{output.stem}_{index:05d}.jpg'), quality=95)


def render(video, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing movie: {output}')
    report, rates = read_metadata(video)
    source_sha = fingerprint(video)
    temporary = output.with_name(output.stem + '.partial.mp4')
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}',
        '-r', str(report['source_fps']), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'medium', '-crf', '16', '-threads', '4', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(temporary),
    ]
    cap = cv2.VideoCapture(str(video))
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if count >= report['frames']:
                raise ValueError(f'Unexpected extra source frame: {video}')
            source = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frame = compose(source, rates[count], report['novel_view_degrees'])
            process.stdin.write(frame.tobytes())
            count += 1
            if count % 20 == 0:
                print(f'{output.stem}: {count}/{report["frames"]}', flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f'Encoding failed: {output}')
        if count != report['frames']:
            raise ValueError(f'Incomplete source decode: {count}/{report["frames"]}')
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
    temporary.replace(output)
    for index in sorted({0, count // 2, count - 1}):
        save_still(video, output, report, rates, index)
    write_json(output.with_suffix('.layout.json'), dict(
        source_video=str(video), source_video_sha256=source_sha,
        source_render_report=str(video.with_suffix('.render.json')),
        layout_script_sha256=fingerprint(Path(__file__)),
        width=CANVAS_SIZE[0], height=CANVAS_SIZE[1], frames=count,
        playback_fps=report['source_fps'], panel_boxes=BOXES,
        camera_order=['Pose 2', 'Initial', 'Pose 1'],
        novel_view_degrees=report['novel_view_degrees'],
        provenance='Layout-only recomposition of the approved movie; particle animation, '
                   'frustums and RGB projections retained; no inference or camera refitting.',
        displayed_inference_timing=report['displayed_inference_timing'],
    ))
    print(f'Saved {output}', flush=True)


def gallery(out_dir, names):
    cards = []
    for name in names:
        source = source_video(name)
        report, _ = read_metadata(source)
        stem = f'{name}_simplified'
        title = 'Test video' if name == 'test' else name.title()
        cards.append(f'''<article><h2>{html.escape(title)}</h2>
<video controls loop playsinline preload="metadata" poster="{stem}_00000.jpg"
 src="{stem}.mp4"></video>
<p><a href="{stem}.mp4">MP4</a> ·
<a href="{stem}_{report['frames']//2:05d}.png">Still figure</a></p></article>''')
    page = '''<!doctype html><html lang="en">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simplified particle-model movies</title><style>
body{font:18px/1.5 system-ui,sans-serif;color:#1e2836;margin:32px auto;padding:0 24px;max-width:1800px}
article{margin:40px 0}h2{margin-bottom:12px}video{display:block;width:100%}a{color:#256676}
</style>''' + ''.join(cards) + '''<script>
document.querySelectorAll('video').forEach(video=>video.addEventListener('play',()=>{
 document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause();});
}));</script></html>'''
    with (out_dir / 'index.html').open('x') as stream:
        stream.write(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', nargs='+', choices=CLIPS, default=list(CLIPS))
    parser.add_argument('--out-dir', type=Path,
                        default=ROOT / 'runs/unified_quality/screenshot_slide_layout')
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    previews = set()
    for name in args.clips:
        source = source_video(name)
        if args.out_dir.resolve() == source.parent.resolve():
            parser.error('The simplified layout must use a separate output folder.')
        if (args.out_dir / f'{name}_simplified.mp4').exists():
            parser.error('Output movie already exists; choose a new --out-dir.')
        report, _ = read_metadata(source)
        previews.update(f'{name}_simplified_{report["frames"]//2:05d}.{suffix}'
                        for suffix in ('png', 'jpg'))
    if args.out_dir.exists() and {p.name for p in args.out_dir.iterdir()} - previews:
        parser.error('Output folder contains an earlier export; choose a new --out-dir.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.clips:
        source = source_video(name)
        output = args.out_dir / f'{name}_simplified.mp4'
        if args.preview_only:
            report, rates = read_metadata(source)
            save_still(source, output, report, rates, report['frames'] // 2)
        else:
            render(source, output)
    if not args.preview_only:
        gallery(args.out_dir, args.clips)
        print(args.out_dir / 'index.html', flush=True)


if __name__ == '__main__':
    main()
