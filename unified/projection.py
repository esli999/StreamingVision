"""Project observed posterior readouts on particle-conditioned relative geometry.

These are conditioned reprojections, not unconditional novel-scene synthesis.
RGB is the saved posterior expectation and membership is its inferred label map.
The depth of each readout sample is the Gaussian-mixture conditional expectation
along its source ray, restricted to particles of that inferred object. No input
RGB texture, later annotation, hole filling, or unseen surface is substituted.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from numba import njit
from .surface import _triangles
from .geometry import Camera, look_at

_VIEW_CACHE={}


@dataclass
class ReadoutSurface:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    foreground: np.ndarray


@njit(cache=True)
def _conditional_depth(rays, labels, means, inverse, log_prior, ids):
    out=np.empty(len(rays),np.float64)
    for i in range(len(rays)):
        best=-1e30; total=0.; depth=0.
        for j in range(len(means)):
            if ids[j]!=labels[i] or log_prior[j]<-1e20:
                continue
            a=0.;b=0.;c=0.
            for k in range(3):
                for l in range(3):
                    a+=rays[i,k]*inverse[j,k,l]*rays[i,l]
                    b+=rays[i,k]*inverse[j,k,l]*means[j,l]
                    c+=means[j,k]*inverse[j,k,l]*means[j,l]
            z=b/a
            if z<=.5:
                continue
            score=log_prior[j]-.5*(c-b*b/a)-.5*np.log(a)
            new=max(best,score)
            old_scale=np.exp(best-new);weight=np.exp(score-new)
            total=total*old_scale+weight
            depth=depth*old_scale+weight*z
            best=new
        out[i]=depth/total if total>0 else np.nan
    return out


def readout_surface(frame, stride=1):
    """Lift actual inferred 2D readouts; pixel cells do not bridge object edges."""
    labels=np.asarray(frame['labels'])
    h,w=labels.shape
    if (h,w)!=(360,640) or h%stride or w%stride:
        raise ValueError('Readout surface requires a 640x360 capture and a dividing stride')
    yy,xx=np.mgrid[stride//2:h:stride,stride//2:w:stride]
    ids=labels[yy,xx].ravel()
    # Odd strides need the half-pixel offset: raster samples are pixel centers,
    # not integer pixel corners. Native resolution preserves thin silhouettes.
    offset=stride*.5-stride//2
    rays=np.stack([(xx+offset-w/2)/500.,(yy+offset-h/2)/500.,np.ones_like(xx)],-1).reshape(-1,3)
    covariance=np.asarray(frame['covs'],float)+np.eye(3)*1e-8
    alpha=frame.get('display_alpha',np.ones(len(covariance)))
    prior=np.log(np.maximum(frame['weights'],1e-12))-.5*np.linalg.slogdet(covariance)[1]
    prior+=np.log(np.maximum(alpha,1e-12))
    prior=np.where(alpha>.005,prior,-1e30)
    depth=_conditional_depth(rays,ids,np.asarray(frame['means'],float),
                             np.linalg.inv(covariance),prior,np.asarray(frame['object_ids']))
    # Shared corners join neighboring cells of the SAME inferred object. This
    # prevents the pinholes produced by independent front-facing pixel quads.
    # Semantic discontinuities stay disconnected; they are not stretched shut.
    shape=xx.shape
    corner_depth=np.zeros((len(ids),4))
    rgb=frame['reconstruction'][yy,xx,::-1].astype(float)
    corner_rgb=np.zeros((len(ids),4,3))
    def corner_sum(a):
        a=np.pad(a,((1,1),(1,1))+((0,0),)*(a.ndim-2))
        return a[:-1,:-1]+a[1:,:-1]+a[1:,1:]+a[:-1,1:]
    for label in np.unique(ids):
        selected=(ids==label)&np.isfinite(depth)
        weight=selected.reshape(shape).astype(float)
        count=corner_sum(weight)
        z=corner_sum(np.where(selected,depth,0).reshape(shape))/np.maximum(count,1)
        c=corner_sum(rgb*weight[...,None])/np.maximum(count[...,None],1)
        values=np.stack([z[:-1,:-1],z[:-1,1:],z[1:,1:],z[1:,:-1]],-1).reshape(-1,4)
        colors=np.stack([c[:-1,:-1],c[:-1,1:],c[1:,1:],c[1:,:-1]],-2).reshape(-1,4,3)
        corner_depth[selected]=values[selected]
        corner_rgb[selected]=colors[selected]
    offsets=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*stride
    corners=rays[:,None,:]+np.pad(offsets/500.,((0,0),(0,1)))[None]
    vertices=(corners*corner_depth[:,:,None]).reshape(-1,3).astype(np.float32)
    faces=(np.array([[0,1,2],[0,2,3]])[None]+4*np.arange(len(ids))[:,None,None]).reshape(-1,3).astype(np.int32)
    valid=np.isfinite(depth)
    faces=faces[np.repeat(valid,2)]
    return ReadoutSurface(vertices,faces,corner_rgb.reshape(-1,3).astype(np.float32),
                          np.repeat(ids>0,4).astype(np.float32))


def project_readouts(surface,camera,antialias=2):
    """One z-buffer and one set of samples supplies both RGB and segmentation."""
    from .render import color_for
    h,w=camera.height*antialias,camera.width*antialias
    uv,z=camera.project(surface.vertices)
    depth=np.full((h,w),np.inf,np.float32)
    color=np.zeros((h,w,3),np.float32);coverage=np.zeros((h,w),np.float32)
    opacity=np.ones(len(surface.vertices),np.float32)
    _triangles((uv*antialias).astype(np.float32),z.astype(np.float32),surface.faces,
               surface.colors,opacity,depth,color,coverage)
    # Rasterize IDs through identical geometry/depth. Values are constant on
    # each cell, so interpolation cannot mix foreground and background IDs.
    semantic=np.zeros_like(color);semantic_depth=np.full_like(depth,np.inf)
    _triangles((uv*antialias).astype(np.float32),z.astype(np.float32),surface.faces,
               np.repeat(surface.foreground[:,None],3,axis=1),opacity,
               semantic_depth,semantic,np.zeros_like(coverage))
    color[coverage==0]=255
    size=(camera.width,camera.height)
    rgb=cv2.resize(color,size,interpolation=cv2.INTER_AREA).clip(0,255).astype(np.uint8)
    mask=cv2.resize(semantic[:,:,0],size,interpolation=cv2.INTER_AREA)>.5
    visible=cv2.resize(coverage,size,interpolation=cv2.INTER_AREA)
    segmentation=np.where(mask[...,None],color_for(1),np.array([211,219,225],np.uint8)).astype(np.uint8)
    segmentation[visible<.5]=255
    return rgb,segmentation,mask,visible


def choose_view(capture,presentation):
    """Reject disoccluding viewpoints over the COMPLETE clip, then freeze one.

    A forward displacement reveals much less unobserved matter than a side
    orbit. Candidates reduce that displacement, rather than filling holes or
    switching cameras when a difficult frame arrives. A failed clip has no
    novel-view claim: the observed projection is used instead.
    """
    key=(str(getattr(capture,'path','')),presentation.stabilized)
    if key[0] and key in _VIEW_CACHE:
        return _VIEW_CACHE[key]
    target=presentation.novel_target
    fractions=[.04,.02,.01,.005]
    candidates=[look_at(target*f,target,500.,640,360) for f in fractions]
    worst=np.zeros(len(candidates));largest=np.zeros(len(candidates),int)
    evaluated=np.zeros(len(candidates),int)
    clipped=np.zeros(len(candidates));active=np.ones(len(candidates),bool)
    for i in range(capture.manifest['frames']):
        frame=presentation.frame(capture.read(i),i)
        surface=readout_surface(frame)
        source_uv,_=presentation.source_camera.project(surface.vertices)
        source_inside=np.all((source_uv>=[2,2])&(source_uv<=[638,358]),axis=1)
        fg=surface.foreground.astype(bool)&source_inside
        for k,cam in enumerate(candidates):
            if not active[k]:
                continue
            evaluated[k]+=1
            small=Camera(cam.position,cam.rotation,450.,576,324)
            _,_,_,coverage=project_readouts(surface,small)
            missing=coverage<.5
            worst[k]=max(worst[k],float(missing.mean()))
            _,_,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),8)
            component=int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
            largest[k]=max(largest[k],component)
            uv,z=cam.project(surface.vertices[fg])
            loss=float(np.mean((z<=.5)|np.any((uv<[0,0])|(uv>[640,360]),axis=1))) if fg.any() else 0.
            clipped[k]=max(clipped[k],loss)
            active[k]=(worst[k]<=.00075 and largest[k]<=24 and clipped[k]<=.01)
        if (i+1)%20==0:
            print(f"view selection {presentation.manifest.get('clip','clip')}: {i+1}/{capture.manifest['frames']}",flush=True)
        if not active.any():
            break
    accepted=np.flatnonzero(active)
    selected=int(accepted[0]) if len(accepted) else None
    report=dict(policy='one fixed forward view passing coverage and clipping gates on every frame; no filling',
                forward_fractions=fractions,passed=active.tolist(),
                maximum_missing_fraction=worst.tolist(),maximum_missing_component_pixels=largest.tolist(),
                maximum_additional_foreground_clipping=clipped.tolist(),
                evaluated_frames=evaluated.tolist(),
                rejected_candidate_policy='stop at first failure; maxima describe evaluated frames only',
                selected_fraction=fractions[selected] if selected is not None else 0.,
                novel_view_available=selected is not None,checked_frames=i+1,
                resolution=[576,324],maximum_missing_fraction_limit=.00075,
                maximum_component_pixels_limit=24,maximum_foreground_clipping_limit=.01)
    result=(candidates[selected] if selected is not None else presentation.source_camera,report)
    if key[0]: _VIEW_CACHE[key]=result
    return result
