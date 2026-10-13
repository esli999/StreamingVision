"""Lazy feature extraction. Importing the renderer never initializes CUDA."""
import argparse
import json
import sys
import time
import cv2
import numpy as np
from .data import ROOT, fingerprint


def load_flow_model(device='cuda'):
    """Load the explicit SEA-RAFT weights, including from a fully offline cache."""
    sys.path.insert(0, str(ROOT/'third_party/SEA-RAFT/core'))
    from raft import RAFT
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_model
    args = argparse.Namespace(**json.loads((ROOT/'third_party/SEA-RAFT/config/eval/spring-M.json').read_text()))
    args.iters, args.scale = 4, 0
    # This upstream repository has weights but no config.json. Loading the
    # weights directly avoids an unnecessary config fetch that fails offline.
    flow = RAFT(args)
    weights = hf_hub_download('MemorySlices/Tartan-C-T-TSKH-spring540x960-M','model.safetensors')
    missing,unexpected = load_model(flow,weights,strict=False)
    # Upstream weights include four obsolete BatchNorm bookkeeping buffers.
    unexpected = {k for k in unexpected if not k.endswith('.num_batches_tracked')}
    if missing or unexpected:
        raise RuntimeError(f'SEA-RAFT checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    return flow.to(device).eval(),fingerprint(weights)


class Models:
    def __init__(self, config):
        started = time.perf_counter()
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from streaming_dino import load_dino
        torch.set_num_threads(4)
        self.torch, self.config = torch, config
        self.device = 'cuda'
        self.depth_processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
        self.depth = AutoModelForDepthEstimation.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf', torch_dtype=torch.float16).to(self.device).eval()
        self.flow,flow_hash = load_flow_model(self.device)
        self.dino = load_dino(self.device, checkpoint=config['dino_checkpoint'])
        self.versions = dict(depth_revision=getattr(self.depth.config,'_commit_hash',None),
                             dino_revision=getattr(self.dino.config,'_commit_hash',None),
                             flow_weights_sha256=flow_hash)
        self.previous = None
        self.basis = self.mean = self.std = None
        self.load_seconds = time.perf_counter()-started
        self.clips_tracked = 0

    def reset(self):
        self.previous = self.basis = self.mean = self.std = None

    def extract(self, bgr):
        from streaming_dino import dino_patches
        torch = self.torch
        start = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        with torch.inference_mode():
            inputs = self.depth_processor(images=rgb, return_tensors='pt').to(self.device, torch.float16)
            depth = self.depth(**inputs).predicted_depth[:,None]
            depth = torch.nn.functional.interpolate(depth.float(), (h,w), mode='bilinear', align_corners=False)[0,0].cpu().numpy()
            current = torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().to(self.device)
            # Bidirectional flow is essential: current depth is paired with the
            # PREVIOUS depth at its backward-warped coordinates, not at x,y.
            if self.previous is None:
                forward = backward = np.zeros((2,h,w), np.float32)
            else:
                pad = (0, (-w)%8, 0, (-h)%8)
                old = torch.nn.functional.pad(self.previous, pad, mode='replicate')
                new = torch.nn.functional.pad(current, pad, mode='replicate')
                forward = self.flow(old, new, iters=4, test_mode=True)['flow'][-1][0,:,:h,:w].cpu().numpy()
                backward = self.flow(new, old, iters=4, test_mode=True)['flow'][-1][0,:,:h,:w].cpu().numpy()
            self.previous = current
            patches, grid = dino_patches(self.dino, rgb, dino_h=self.config['dino_h'], dino_w=self.config['dino_w'])
        if self.basis is None:
            self.mean = patches.mean(0)
            centered = patches-self.mean
            _, vectors = np.linalg.eigh(centered.T@centered/(len(centered)-1))
            self.basis = vectors[:,-32:].astype(np.float32)
            self.std = (centered@self.basis).std(0)+.01
        # Project before interpolation: 32 channels instead of 384/768. This
        # commutes with linear interpolation and preserves the frozen basis.
        reduced = ((patches-self.mean)@self.basis/self.std).reshape(*grid,32)
        stride = self.config['sampling_stride']
        features = cv2.resize(reduced, (640//stride,360//stride), interpolation=cv2.INTER_LINEAR)
        torch.cuda.synchronize()
        return depth, forward, backward, features, time.perf_counter()-start
