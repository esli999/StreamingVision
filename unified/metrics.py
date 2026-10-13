"""Post-inference evaluation. This module alone reads later DAVIS annotations."""
import numpy as np
import cv2
from .data import labels_from_mask
from .geometry import warp_previous


def boundary(mask):
    """One-pixel segmentation boundary, matching DAVIS's neighbor differences."""
    m = np.asarray(mask,bool)
    east,south,se = np.zeros_like(m),np.zeros_like(m),np.zeros_like(m)
    east[:,:-1] = m[:,1:]
    south[:-1] = m[1:]
    se[:-1,:-1] = m[1:,1:]
    b = (m^east)|(m^south)|(m^se)
    b[-1,:] = m[-1,:]^east[-1,:]
    b[:,-1] = m[:,-1]^south[:,-1]
    b[-1,-1] = False
    return b


def j_and_f(prediction,truth,valid=None):
    valid = np.ones_like(truth,bool) if valid is None else valid
    p,t = np.asarray(prediction,bool)&valid,np.asarray(truth,bool)&valid
    union = np.sum(p|t)
    j = float(np.sum(p&t)/union) if union else 1.
    pb,tb = boundary(p),boundary(t)
    # Suppress artificial boundaries created by void pixels.
    valid_boundary = cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    pb &= valid_boundary; tb &= valid_boundary
    radius = max(1,int(np.ceil(.008*np.linalg.norm(p.shape))))
    yy,xx = np.mgrid[-radius:radius+1,-radius:radius+1]
    kernel = (xx*xx+yy*yy <= radius*radius).astype(np.uint8)
    pd = cv2.dilate(pb.astype(np.uint8),kernel).astype(bool)
    td = cv2.dilate(tb.astype(np.uint8),kernel).astype(bool)
    npred,ntrue = int(pb.sum()),int(tb.sum())
    if not npred and not ntrue:
        f = 1.
    elif not npred or not ntrue:
        f = 0.
    else:
        precision,recall = float((pb&td).sum()/npred),float((tb&pd).sum()/ntrue)
        f = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return j,f


def native_labels(prediction,shape):
    h,w = prediction.shape
    nh,nw = shape
    scale = min(w/nw,h/nh)
    cw,ch = round(nw*scale),round(nh*scale)
    x,y = (w-cw)//2,(h-ch)//2
    return cv2.resize(prediction[y:y+ch,x:x+cw],(nw,nh),interpolation=cv2.INTER_NEAREST)


def evaluate_capture(capture,annotations=None):
    ids = capture.manifest['foreground_ids']
    scores = {int(k):[] for k in ids}
    temporal,centroid_jumps,coverage = [],[],[]
    previous = None
    frame_scores = []
    for i in range(capture.manifest['frames']):
        frame = capture.read(i)
        labels = frame['labels']
        coverage.append(float((labels>0).mean()))
        if previous is not None:
            if 'backward_grid' in frame:
                old,valid = warp_previous(previous['labels'][4::8,4::8],frame['backward_grid'].astype(np.float32),nearest=True)
                valid &= frame['reliable_grid']
                current = labels[4::8,4::8]
                use = valid & ((old>0)|(current>0))
                if use.any():
                    temporal.append(float(np.mean(old[use] != current[use])))
            supported = (frame['object_ids']>0)&(frame['weights']*2925>3)&(previous['weights']*2925>3)
            if supported.any():
                centroid_jumps.extend(np.linalg.norm(frame['means'][supported]-previous['means'][supported],axis=1).tolist())
        if annotations is not None and i > 0:
            path = annotations/f'{i:05d}.png'
            if not path.is_file():
                raise FileNotFoundError(f'Missing evaluation annotation: {path}')
            truth = labels_from_mask(path)
            prediction = native_labels(labels,truth.shape)
            values = []
            for predicted_id,original_id in enumerate(ids,1):
                j,f = j_and_f(prediction == predicted_id,truth == original_id,truth != 255)
                scores[int(original_id)].append((j,f)); values.append((j,f))
            frame_scores.append(dict(frame=i,j=float(np.mean([x[0] for x in values])),f=float(np.mean([x[1] for x in values]))))
        previous = frame
    object_scores = {str(k):dict(j=float(np.mean(v,axis=0)[0]),f=float(np.mean(v,axis=0)[1]),frames=len(v))
                     for k,v in scores.items() if v}
    result = dict(clip=capture.manifest['clip'],frames=capture.manifest['frames'],objects=object_scores,
                  seed_frame_excluded=True,identity_matching='fixed frame-0 IDs; no per-frame relabeling',
                  resolution='original annotation resolution',per_frame=frame_scores,
                  temporal_disagreement=float(np.mean(temporal)) if temporal else None,
                  particle_motion_p95=float(np.percentile(centroid_jumps,95)) if centroid_jumps else None,
                  foreground_fraction_min=min(coverage),foreground_fraction_max=max(coverage),
                  inference_fps=capture.manifest.get('inference_fps'))
    if object_scores:
        result['j'] = float(np.mean([x['j'] for x in object_scores.values()]))
        result['f'] = float(np.mean([x['f'] for x in object_scores.values()]))
        result['jf'] = (result['j']+result['f'])/2
    return result
