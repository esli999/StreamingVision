"""Input identity, masks and atomic, non-pickle inference captures."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image
from . import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'runs/unified_quality/data/DAVIS'


def fingerprint(path):
    p = Path(path)
    h = hashlib.sha256()
    files = sorted(p.glob('*')) if p.is_dir() else [p]
    for f in files:
        if not f.is_file():
            continue
        if p.is_dir():
            h.update(f.name.encode()+b'\0'+str(f.stat().st_size).encode()+b'\0')
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
    return h.hexdigest()


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(p)


def labels_from_mask(path):
    # PIL preserves DAVIS palette indices; cv2 would convert them into colors.
    with Image.open(path) as image:
        m = np.asarray(image)
    if m.ndim == 3:
        # Encode colors directly so an object disappearing cannot renumber the
        # remaining identities. Reserve the low IDs (including DAVIS void 255)
        # for indexed masks. Black/transparent pixels are background.
        rgb = m[...,:3].astype(np.int32)
        packed = (rgb[...,0]<<16)|(rgb[...,1]<<8)|rgb[...,2]
        foreground = packed != 0
        if m.shape[2] == 4:
            foreground &= m[...,3] > 0
        m = np.where(foreground,packed|(1<<24),0)
    return m.astype(np.int32)


def select_foreground(mask, ids=None):
    ids = sorted(int(x) for x in (np.unique(mask) if ids is None else ids) if int(x) > 0)
    if not ids or any(not np.any(mask == x) for x in ids):
        raise ValueError('The initial mask must contain every selected foreground ID')
    out = np.zeros_like(mask, np.int32)
    for i, label in enumerate(ids, 1):
        out[mask == label] = i
    return out, ids


def implementation_fingerprint(kind='all'):
    digest = hashlib.sha256()
    if kind=='inference':
        files=[ROOT/'unified'/f'{name}.py' for name in
               ('data','geometry','inference','models','seed','tracker')]
    elif kind=='all':
        files=sorted((ROOT/'unified').glob('*.py'))
    else:
        raise ValueError(f'Unknown implementation fingerprint kind: {kind}')
    files=files+[ROOT/'genmatter_rt.py',ROOT/'streaming_dino.py']
    for file in files:
        digest.update(str(file.relative_to(ROOT)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


@dataclass
class Clip:
    name: str
    source: Path
    mask: Path
    provenance: str
    fps: float = 30.
    annotations: Path | None = None
    foreground_ids: list | None = None


def resolve_clip(name, data_root=DEFAULT_DATA, seed_mask=None, foreground_ids=None, source_fps=30.):
    data_root = Path(data_root)
    direct = Path(name)
    label = direct.resolve().stem if direct.exists() else str(name)
    source_override = None
    if direct.exists() and (direct.is_file() or direct.is_dir()):
        source = direct.resolve()
        if seed_mask:
            return Clip(label, source, Path(seed_mask).resolve(), 'user initial mask', source_fps,
                        foreground_ids=foreground_ids)
        if source == (ROOT/'assets/test.mp4').resolve():
            source_override = source
            name, label = 'test', 'test'
        else:
            raise ValueError('Raw input requires --seed-mask and optionally --foreground-ids')
    for base in (data_root,):
        src, ann = base/'JPEGImages/480p'/str(name), base/'Annotations/480p'/str(name)
        if src.is_dir() and (ann/'00000.png').is_file():
            mask = Path(seed_mask) if seed_mask else ann/'00000.png'
            provenance = 'user initial mask' if seed_mask else 'DAVIS frame-0 annotation'
            return Clip(str(name), src, mask, provenance, source_fps, ann,
                        foreground_ids)
    old = ROOT/'assets/tapvid_davis_30_videos_processed'
    src, ann = old/'tapvid_davis_rgb_frames'/str(name), old/'tapvid_davis_segmasks'/str(name)
    if src.is_dir() and (ann/'00000.png').is_file():
        mask = Path(seed_mask) if seed_mask else ann/'00000.png'
        provenance = 'user initial mask' if seed_mask else 'DAVIS frame-0 annotation'
        return Clip(str(name), src, mask, provenance, source_fps, ann,
                    foreground_ids)
    base = ROOT/'assets/custom_videos'/str(name)
    sources = sorted(base.glob('source.*'))
    mask = Path(seed_mask) if seed_mask else base/'pseudo_gt_sam/segmasks'/str(name)/'00000.png'
    if (sources or source_override) and mask.is_file():
        provenance = 'user initial mask' if seed_mask else 'SAM frame-0 instances'
        return Clip(str(name), source_override or sources[0], mask, provenance, source_fps, None, foreground_ids)
    raise FileNotFoundError(f'Missing source or initial mask for {name}')


def iter_frames(source, max_frames=-1, *, with_time=False, fps=30.):
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg'))
        for i, p in enumerate(files):
            if max_frames >= 0 and i >= max_frames:
                break
            frame = cv2.imread(str(p))
            if frame is None:
                raise ValueError(f'Cannot decode {p}')
            yield (frame,i/fps) if with_time else frame
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {source}')
        try:
            i = 0
            last_time = -1.
            while max_frames < 0 or i < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
                if not np.isfinite(timestamp) or timestamp <= last_time:
                    timestamp = max(0.,last_time+1/fps)
                yield (frame,timestamp) if with_time else frame
                last_time = timestamp
                i += 1
        finally:
            cap.release()


def fit_frame(frame, shape=(360,640), nearest=False):
    h, w = shape
    ih, iw = frame.shape[:2]
    scale = min(w/iw, h/ih)
    nw, nh = round(iw*scale), round(ih*scale)
    x, y = (w-nw)//2, (h-nh)//2
    out = np.zeros((h,w)+frame.shape[2:], frame.dtype)
    out[y:y+nh,x:x+nw] = cv2.resize(frame, (nw,nh), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    valid = np.zeros((h,w), bool)
    valid[y:y+nh,x:x+nw] = True
    return out, valid


class Capture:
    def __init__(self, path, manifest=None):
        self.path = Path(path)
        if manifest is None:
            self.manifest = json.loads((self.path/'manifest.json').read_text())
            if self.manifest.get('schema') != SCHEMA_VERSION or not self.manifest.get('complete'):
                raise ValueError('Incomplete or incompatible capture; rerun --stage track')
            expected = {f'frame_{i:05d}.npz' for i in range(self.manifest['frames'])}
            actual = {p.name for p in self.path.glob('frame_*.npz')}
            if actual != expected:
                raise ValueError('Capture frame count does not match its manifest')
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            if (self.path/'manifest.json').exists():
                raise FileExistsError(f'Capture already exists: {self.path}; choose a new output directory')
            self.manifest = {**manifest, 'schema':SCHEMA_VERSION, 'complete':False}
            write_json(self.path/'manifest.json', self.manifest)

    def write(self, index, **arrays):
        target = self.path/f'frame_{index:05d}.npz'
        tmp = self.path/f'frame_{index:05d}.tmp'
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(target)

    def finish(self, **summary):
        self.manifest.update(summary)
        self.manifest['complete'] = True
        write_json(self.path/'manifest.json', self.manifest)

    def read(self, index):
        with np.load(self.path/f'frame_{index:05d}.npz', allow_pickle=False) as data:
            return {k:data[k] for k in data.files}


def rate(durations):
    a = np.asarray(durations, float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(len(a)/a.sum()) if len(a) else None
