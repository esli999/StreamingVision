"""Offset-pose rendering of layered, particle-conditioned 2.5D matter.

The figure carries observed posterior appearance on its inferred surface. Ground
depth AND color are queried from background Gaussians, including behind the
figure. This is a model prediction there, not observed texture or image inpainting.
Both layers are frozen before querying a camera and share a real depth buffer.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from numba import njit
from .geometry import Camera
from .projection import ReadoutSurface, readout_surface, project_readouts
from .surface import Surface


@njit(cache=True)
def _ground_queries(rays, means, inverse, prior, colors):
    depth=np.empty(len(rays)); rgb=np.empty((len(rays),3)); support=np.empty(len(rays))
    for i in range(len(rays)):
        best=-1e30; total=0.; zsum=0.; color=np.zeros(3); nearest=1e30
        for j in range(len(means)):
            a=0.; b=0.; c=0.
            for k in range(3):
                for l in range(3):
                    a+=rays[i,k]*inverse[j,k,l]*rays[i,l]
                    b+=rays[i,k]*inverse[j,k,l]*means[j,l]
                    c+=means[j,k]*inverse[j,k,l]*means[j,l]
            z=b/a
            if z<=.5: continue
            distance=max(0.,c-b*b/a); nearest=min(nearest,distance)
            score=prior[j]-.5*distance-.5*np.log(a); new=max(best,score)
            old=np.exp(best-new); weight=np.exp(score-new)
            total=total*old+weight; zsum=zsum*old+weight*z
            color=color*old+weight*colors[j]; best=new
        depth[i]=zsum/total if total>0 else np.nan
        rgb[i]=color/max(total,1e-30); support[i]=nearest
    return depth,rgb,support


@dataclass
class LayeredReadouts:
    observed: ReadoutSurface
    figure: ReadoutSurface
    ground: ReadoutSurface
    scene: ReadoutSurface
    ground_observed: ReadoutSurface
    ground_supported_fraction: float
    ground_occlusion_adjusted_fraction: float


def subset(surface, faces):
    """Compact a mesh without changing any retained vertices or attributes."""
    used,inverse=np.unique(faces,return_inverse=True)
    return ReadoutSurface(surface.vertices[used],inverse.reshape(-1,3).astype(np.int32),
                          surface.colors[used],surface.foreground[used])


def constrain_ground_occlusion(depth, xx, yy, observed, labels):
    """Respect observed figure-before-ground ordering in the hidden layer.

    This is a geometric inequality for the readout, not a new depth observation.
    The particle-conditioned figure supplies the bound. Two source pixels of
    support cover the background grid's cells at silhouette boundaries. Colors,
    particle states, and background depths away from this support stay intact.
    """
    pixel_depth=observed.vertices[:,2].reshape(360,640,4).max(-1)
    lower=np.where(labels>0,pixel_depth+.04,0).astype(np.float32)
    lower=cv2.dilate(lower,np.ones((5,5),np.uint8))
    limit=cv2.remap(lower,xx.astype(np.float32),yy.astype(np.float32),
                   cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()
    return np.maximum(depth,limit)


def layered_readouts(frame):
    observed=readout_surface(frame)
    figure=subset(observed,observed.faces[observed.foreground[observed.faces[:,0]]>0])
    # Overscan queries the SAME Gaussian model beyond the source image.
    # Gaussians have full support: report the fraction within 3.5 sigma rather
    # than turning their low-density conditional predictions into image holes.
    # No layer exists when the model has no supported background particles.
    # Keep native support dense, with a wider, coarser query domain outside it.
    # This covers stronger offset poses without spending most samples offscreen.
    xs=np.r_[np.arange(-768,0,8),np.arange(0,640,2),np.arange(640,1409,8)]
    ys=np.r_[np.arange(-768,0,8),np.arange(0,360,2),np.arange(360,1129,8)]
    xx,yy=np.meshgrid(xs,ys)
    rays=np.stack([(xx-320)/500.,(yy-180)/500.,np.ones_like(xx)],-1).reshape(-1,3)
    alpha=frame.get('display_alpha',np.ones(len(frame['means'])))
    use=(frame['object_ids']==0)&(alpha>.005)
    covariance=np.asarray(frame['covs'][use],float)+np.eye(3)*1e-8
    prior=np.log(np.maximum(frame['weights'][use],1e-12))-.5*np.linalg.slogdet(covariance)[1]+np.log(alpha[use])
    depth,rgb,support=_ground_queries(rays,np.asarray(frame['means'][use],float),
        np.linalg.inv(covariance),prior,frame['colors'][use,::-1].astype(float))
    constrained=constrain_ground_occlusion(depth,xx,yy,observed,frame['labels'])
    adjusted=constrained>depth+1e-6
    depth=constrained
    valid=np.isfinite(depth)
    vertices=(rays*np.nan_to_num(depth,nan=0.)[:,None]).astype(np.float32)
    grid=np.arange(len(rays)).reshape(xx.shape)
    a=grid[:-1,:-1].ravel(); b=grid[:-1,1:].ravel()
    c=grid[1:,1:].ravel(); d=grid[1:,:-1].ravel()
    faces=np.stack([np.stack([a,b,c],1),np.stack([a,c,d],1)],1).reshape(-1,3)
    faces=faces[valid[faces].all(1)].astype(np.int32)
    ground=ReadoutSurface(vertices,faces,rgb.astype(np.float32),np.zeros(len(rays),np.float32))
    in_source=(xx.ravel()>=0)&(xx.ravel()<=640)&(yy.ravel()>=0)&(yy.ravel()<=360)
    ground_observed=subset(ground,faces[in_source[faces].all(1)])
    scene=ReadoutSurface(np.concatenate([ground.vertices,figure.vertices]),
        np.concatenate([ground.faces,figure.faces+len(ground.vertices)]),
        np.concatenate([ground.colors,figure.colors]),np.concatenate([ground.foreground,figure.foreground]))
    return LayeredReadouts(observed,figure,ground,scene,ground_observed,
                          float((valid[in_source]&(support[in_source]<=3.5**2)).mean()),
                          float(adjusted[in_source].mean()))


def shaded_surface(surface):
    """Natural-color 2.5D sheet, with shared vertex normals and no identity tint."""
    vertices=surface.vertices; faces=surface.faces
    # Pixel cells duplicate corners. Weld equal coordinates for smooth normals
    # without joining disconnected figure/ground boundaries or changing shape.
    _,inverse=np.unique(vertices,axis=0,return_inverse=True)
    normals=np.zeros((int(inverse.max())+1,3),np.float32) if len(inverse) else np.empty((0,3),np.float32)
    tri=vertices[faces]
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    # Orient the observed front toward the source camera.
    n=np.where((np.einsum('ij,ij->i',n,tri.mean(1))>0)[:,None],-n,n)
    for k in range(3): np.add.at(normals,inverse[faces[:,k]],n)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    return Surface(vertices,faces,normals[inverse],surface.colors,0,
                   np.ones(len(vertices),np.float32),False)


def offset_cameras(target, angles, focal=450.):
    from .render import views
    _,cameras=views(dict(world_translation=np.zeros(3),world_rotation=np.eye(3)),target,angles)
    return [Camera(c.position,c.rotation,focal,640,360) for c in cameras]


def coverage_statistics(coverage):
    missing=coverage<.5
    _,_,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
    return float(missing.mean()),largest


_SELECTION_CACHE={}


def ground_fit_points(vertices):
    """Exact support points for the fixed surface viewer, without empty box corners."""
    from .geometry import orbit_camera
    rotation=orbit_camera(np.zeros(3),20.,14.,1.,1100.,1024,576).rotation
    local=vertices@rotation.T
    indices=[np.argmin(local[:,2])]
    for axis,size in [(0,1024),(1,576)]:
        limit=size*.45/1100.
        indices.extend([np.argmax(local[:,axis]-limit*local[:,2]),
                        np.argmin(local[:,axis]+limit*local[:,2])])
    return vertices[np.unique(indices)]


def choose_offset_views(capture,presentation):
    """Two opposite, translated AND rotated poses; one selection for the clip."""
    key=(str(getattr(capture,'path','')),presentation.stabilized)
    if key[0] and key in _SELECTION_CACHE: return _SELECTION_CACHE[key]
    angles=[-20.,-18.,-16.,16.,20.,24.,28.]
    cameras=offset_cameras(presentation.novel_target,angles,focal=410.)
    active=np.ones(len(angles),bool); worst=np.zeros(len(angles)); gaps=np.zeros(len(angles),int)
    clipped=np.zeros(len(angles)); evaluated=np.zeros(len(angles),int)
    visible=np.ones(len(angles))
    ground_min=1.; bounds=[]; ground_bounds=[]
    def fit_points(vertices):
        lo,hi=vertices.min(0),vertices.max(0)
        corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        return np.concatenate([vertices[::64],corners])
    for i in range(capture.manifest['frames']):
        frame=presentation.frame(capture.read(i),i)
        readouts=layered_readouts(frame)
        ground_min=min(ground_min,readouts.ground_supported_fraction)
        if len(readouts.figure.vertices): bounds.append(fit_points(readouts.figure.vertices))
        if len(readouts.ground_observed.vertices): ground_bounds.append(ground_fit_points(readouts.ground_observed.vertices))
        for k,camera in enumerate(cameras):
            if not active[k]: continue
            evaluated[k]+=1
            small=Camera(camera.position,camera.rotation,camera.focal*.875,560,315)
            _,_,scene_mask,coverage=project_readouts(readouts.scene,small)
            _,_,figure_mask,_=project_readouts(readouts.figure,small)
            # Complete RGB coverage can conceal a bad view: a coarse inferred
            # ground fold may cover most of the animal. Keep the shared depth
            # buffer intact and reject that pose instead of painting over it.
            retained=float(scene_mask.sum()/max(figure_mask.sum(),1))
            visible[k]=min(visible[k],retained)
            missing,gap=coverage_statistics(coverage)
            worst[k]=max(worst[k],missing); gaps[k]=max(gaps[k],gap)
            uv,z=camera.project(readouts.figure.vertices)
            loss=float(np.mean((z<=.5)|np.any((uv<[4,4])|(uv>[636,356]),axis=1))) if len(z) else 0.
            clipped[k]=max(clipped[k],loss)
            active[k]=missing<=.001 and gap<=96 and loss==0. and retained>=.92
        if (i+1)%20==0: print(f"offset checks {presentation.manifest.get('clip','clip')}: {i+1}/{capture.manifest['frames']}",flush=True)
    left=[k for k,a in enumerate(angles) if a<0 and active[k]]
    right=[k for k,a in enumerate(angles) if a>0 and active[k]]
    selected=([left[0],right[-1]] if left and right else [])
    report=dict(policy='fixed opposite orbit poses, validated on the complete clip',candidate_degrees=angles,
        passed=active.tolist(),maximum_missing_fraction=worst.tolist(),maximum_missing_component_pixels=gaps.tolist(),
        maximum_foreground_clipping=clipped.tolist(),evaluated_frames=evaluated.tolist(),
        minimum_visible_figure_fraction=visible.tolist(),minimum_visible_figure_fraction_limit=.92,
        selected_degrees=[angles[k] for k in selected],resolution=[560,315],
        maximum_missing_fraction_limit=.001,maximum_component_pixels_limit=96,
        foreground_border_margin_pixels=4,maximum_foreground_clipping_limit=0.,
        ground_supported_fraction_min=ground_min,
        ground_semantics='background-Gaussian conditional depth and RGB, with observed figure-before-ground ordering; occluded regions are constrained predictions',
        rejected_candidate_policy='stop that candidate at first failure; selected candidates pass every frame')
    result=([cameras[k] for k in selected],report,
            dict(figure=np.concatenate(bounds) if bounds else np.empty((0,3)),
                 ground=np.concatenate(ground_bounds) if ground_bounds else np.empty((0,3))))
    if key[0]: _SELECTION_CACHE[key]=result
    return result
