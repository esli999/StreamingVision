#!/usr/bin/env python3
"""Render four separate slide movies of particle-derived mid-level vision.

Reuse the approved display states and camera tracks. Render only the initial
frustum, segmentation, object surface and background surface; no novel views.
"""
import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_simplified_showcase import read_metadata, source_video
from unified.data import Capture, ROOT, fingerprint, write_json
from unified.geometry import Camera
from unified.novel import layered_readouts, shaded_surface
from unified.presentation import camera_coordinates
from unified.projection import project_readouts
from unified.render import (
    CONTENT_BOXES, DEFAULT_COLOR, FONT_ROOT, INK, WHITE, arrow, paste_fit,
    scaled_camera, scene_cameras, text,
)
from unified.surface import particle_surfaces, raster_surfaces


CLIPS = ('blackswan', 'bear', 'elephant', 'dog')
CANVAS_SIZE = (4096, 2304)
BOXES = {
    'input': (96, 352, 1248, 1000),
    'scene': (1600, 160, 4040, 1160),
    'segmentation': (64, 1552, 1344, 2272),
    'surface': (1408, 1552, 2688, 2272),
    'ground': (2752, 1552, 4032, 2272),
}


class SavedDisplay:
    """Replay the reviewed state without refitting cameras or rerunning inference."""

    def __init__(self, name):
        self.video = source_video(name)
        self.report, self.rates = read_metadata(self.video)
        self.capture = Capture(self.report['capture_path'])
        if fingerprint(self.capture.path / 'manifest.json') != self.report['capture_manifest_sha256']:
            raise ValueError(f'Capture differs from the approved export: {name}')
        with np.load(self.video.with_suffix('.cameras.npz')) as state:
            self.state = {key: state[key].copy() for key in state.files}
        if not self.state['ground_available']:
            raise ValueError(f'This showcase requires the approved background surface: {name}')
        if len(self.state['display_means']) != self.report['frames']:
            raise ValueError(f'Incomplete display track: {name}')
        self.colors = []
        for index in range(self.report['frames']):
            raw = self.capture.read(index)
            color = raw['colors'].copy()
            if index:
                unseen = raw.get('support', raw['weights'] * 2925) < .5
                color[unseen] = self.colors[-1][unseen]
            self.colors.append(color)

    def camera(self, prefix):
        state = self.state
        return Camera(state[prefix + '_position'], state[prefix + '_rotation'],
                      float(state[prefix + '_focal']), *state[prefix + '_size'])

    def frame(self, index):
        frame = camera_coordinates(self.capture.read(index))
        state = self.state
        scene = self.camera('scene')
        frame.update(means=state['display_means'][index], covs=state['display_covs'][index],
                     colors=self.colors[index], display_alpha=state['alpha'][index],
                     source_camera=self.camera('default'), detail_camera=self.camera('detail'),
                     ground_camera=self.camera('ground'), novel_target=state['novel_target'],
                     scene_camera=Camera(state['scene_positions'][index], scene.rotation,
                                         scene.focal, scene.width, scene.height))
        return frame


def compose(display, index, *, verify_scene=False):
    frame = display.frame(index)
    image = Image.new('RGB', CANVAS_SIZE, WHITE)
    draw = ImageDraw.Draw(image)

    def place(array, key):
        x, y, right, bottom = BOXES[key]
        paste_fit(image, array, (x, y, right - x, bottom - y))

    place(frame['bgr'][..., ::-1], 'input')
    cloud = raster_surfaces(particle_surfaces(frame, highlight=True, context_opacity=.90),
                            frame['scene_camera'], semantic=False)
    if verify_scene:
        # Confirm the saved-state replay matches an approved lossless particle
        # panel exactly before removing its two unneeded novel-camera glyphs.
        state = display.state
        cameras = [Camera(p, r, f, 640, 360) for p, r, f in zip(
            state['novel_positions'], state['novel_rotations'], state['novel_focals'])]
        original = scene_cameras(cloud, frame['scene_camera'], cameras,
                                 frame['novel_target'], default_camera=frame['source_camera'])
        still = display.video.with_name(f'{display.video.stem}_{index:05d}.png')
        with Image.open(still) as source:
            np.testing.assert_array_equal(original, np.asarray(source.crop(CONTENT_BOXES['scene'])))
    place(scene_cameras(cloud, frame['scene_camera'], [], frame['novel_target'],
                        default_camera=frame['source_camera']), 'scene')

    # Query the same particle-conditioned percepts at slide-native resolution.
    readouts = layered_readouts(frame)
    _, segmentation, _, _ = project_readouts(readouts.observed,
                                             scaled_camera(frame['source_camera'], 2.))
    place(segmentation, 'segmentation')
    place(raster_surfaces([shaded_surface(readouts.figure)],
                         scaled_camera(frame['detail_camera'], 1.25), semantic=False), 'surface')
    place(raster_surfaces([shaded_surface(readouts.ground_observed)],
                         scaled_camera(frame['ground_camera'], 1.25), semantic=False), 'ground')

    text(draw, (96, 246), 'Input video', 80, INK, True)
    text(draw, (1600, 62), '3D particle-based matter representation', 80, INK, True)
    text(draw, (1272, 368), 'Real-time', 60, INK, True)
    text(draw, (1272, 448), 'inference', 60, INK, True)
    fps = display.rates[index]
    speed = f'{fps:.1f} FPS' if np.isfinite(fps) else '— FPS'
    text(draw, (1300, 528), speed, 56, INK, True)
    arrow(draw, (1272, 676), (1570, 676), (91, 111, 128), 44)
    arrow(draw, (2820, 1182), (2820, 1270), (91, 111, 128), 36)

    heading = 'Inferred mid-level visual representations from initial camera view'
    font = ImageFont.truetype(str(FONT_ROOT / 'DejaVuSans-Bold.ttf'), 76)
    x = (CANVAS_SIZE[0] - draw.textlength(heading, font=font)) / 2
    text(draw, (x, 1288), heading, 76, DEFAULT_COLOR, True)
    for key, label in (
        ('segmentation', '2D segmentation'),
        ('surface', 'Deformable object · 2.5D surface'),
        ('ground', 'Background · 2.5D surface'),
    ):
        x, y, right, bottom = BOXES[key]
        text(draw, (x, 1440), label, 66, INK, True)
        draw.rectangle((x - 3, y - 3, right + 3, bottom + 3), outline=DEFAULT_COLOR, width=5)
    return image


def save_still(image, output, index):
    image.save(output.with_name(f'{output.stem}_{index:05d}.png'))
    image.save(output.with_name(f'{output.stem}_{index:05d}.jpg'), quality=95)


def render(name, out_dir, preview_only=False):
    display = SavedDisplay(name)
    output = out_dir / f'{name}_midlevel.mp4'
    count = display.report['frames']
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}')
    if preview_only:
        save_still(compose(display, count // 2, verify_scene=True), output, count // 2)
        print(f'Preview ready: {name}; saved particle scene matches the approved export.', flush=True)
        return
    temporary = output.with_name(output.stem + '.partial.mp4')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}',
               '-r', str(display.report['source_fps']), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'medium', '-crf', '16', '-threads', '4', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(count):
            image = compose(display, index)
            if index in {0, count // 2, count - 1}:
                save_still(image, output, index)
            process.stdin.write(image.tobytes())
            if (index + 1) % 20 == 0:
                print(f'{name}: {index + 1}/{count}', flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f'Encoding failed: {output}')
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(output)
    write_json(output.with_suffix('.render.json'), dict(
        video=str(output), width=CANVAS_SIZE[0], height=CANVAS_SIZE[1], frames=count,
        source_fps=display.report['source_fps'], capture_path=str(display.capture.path),
        capture_manifest_sha256=display.report['capture_manifest_sha256'],
        source_camera_state=str(display.video.with_suffix('.cameras.npz')),
        source_camera_state_sha256=fingerprint(display.video.with_suffix('.cameras.npz')),
        renderer_script_sha256=fingerprint(Path(__file__)), panel_boxes=BOXES,
        displayed_inference_timing=display.report['displayed_inference_timing'],
        geometry='Approved stabilized particle states, held appearance and camera tracks replayed without refitting.',
        frustum='Initial camera only; same pose, field of view and glyph as the detailed figure.',
        readouts='Particle-conditioned 2D segmentation, deformable-object 2.5D surface and background 2.5D surface; initial-camera support.',
        surface_view='Approved fixed oblique display cameras reveal surface depth; no novel-view synthesis panels.',
    ))
    print(f'Saved {output}', flush=True)


def gallery(out_dir, clips):
    cards = []
    for name in clips:
        report = json.loads((out_dir / f'{name}_midlevel.render.json').read_text())
        stem = f'{name}_midlevel'
        cards.append(f'''<article><h2>{html.escape(name.title())}</h2>
<video controls loop playsinline preload="metadata" poster="{stem}_00000.jpg" src="{stem}.mp4"></video>
<p><a href="{stem}.mp4">MP4</a> · <a href="{stem}_{report['frames']//2:05d}.png">Still figure</a></p></article>''')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mid-level vision from GenMatter</title><style>
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
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'runs/unified_quality/midlevel_slide_layout')
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    allowed = set()
    for name in args.clips:
        report, _ = read_metadata(source_video(name))
        allowed.update(f'{name}_midlevel_{report["frames"]//2:05d}.{suffix}' for suffix in ('png', 'jpg'))
    if args.out_dir.exists() and {p.name for p in args.out_dir.iterdir()} - allowed:
        parser.error('Output folder contains an earlier export; choose a fresh --out-dir.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.clips:
        render(name, args.out_dir, args.preview_only)
    if not args.preview_only:
        gallery(args.out_dir, args.clips)
        print(args.out_dir / 'index.html', flush=True)


if __name__ == '__main__':
    main()
