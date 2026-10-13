"""Derived Gaussian-density surfaces and CPU triangle rasterization.

The mesh is an illustration of the captured posterior, not inferred topology or
new geometry. Coordinates, voxel spacing and density level stay fixed over time.
Object fields are extracted separately so differently labelled matter cannot fuse.
"""
from dataclasses import dataclass
from functools import lru_cache
import cv2
import numpy as np
from numba import njit
from skimage.measure import marching_cubes


@dataclass
class Surface:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    object_id: int
    opacity: np.ndarray
    highlight: bool = False


@lru_cache(maxsize=1)
def _unit_sphere():
    """Fixed tessellation: particle orientation cannot change its connectivity."""
    rings, sectors = 16, 24
    vertices = [[0., 0., 1.]]
    for j in range(1, rings):
        phi = np.pi*j/rings
        for k in range(sectors):
            theta = 2*np.pi*k/sectors
            vertices.append([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
    vertices.append([0., 0., -1.])
    faces = []
    for k in range(sectors):
        nxt = (k+1) % sectors
        faces.append([0, 1+k, 1+nxt])
        for j in range(rings-2):
            a, b = 1+j*sectors+k, 1+j*sectors+nxt
            faces.extend([[a, a+sectors, b], [b, a+sectors, b+sectors]])
        faces.append([len(vertices)-1, 1+(rings-2)*sectors+nxt, 1+(rings-2)*sectors+k])
    return np.asarray(vertices), np.asarray(faces, np.int32)


def particle_surfaces(frame, sigma=1.8, *, highlight=True, context_opacity=.58):
    """Rasterizable covariance ellipsoids for every observed scene particle.

    The symmetric positive square root is continuous across eigenvector sign
    flips and eigenvalue crossings. Geometry and normals use the full covariance;
    opacity never rescales a region. No points or connecting topology are added.
    """
    sphere, faces = _unit_sphere()
    result = []
    alpha = frame.get('display_alpha', np.ones(len(frame['means'])))
    for label in np.unique(frame['object_ids']):
        selected = (frame['object_ids'] == label) & (alpha > .005)
        means, covs = frame['means'][selected], frame['covs'][selected]
        if not len(means):
            continue
        values, vectors = np.linalg.eigh(covs)
        root = np.einsum('nij,nj,nkj->nik', vectors, np.sqrt(np.maximum(values, 1e-8))*sigma, vectors)
        vertices = np.einsum('vi,nji->nvj', sphere, root)+means[:, None]
        normals = np.einsum('vi,nij->nvj', sphere, np.linalg.inv(root))
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        colors = frame['colors'][selected, ::-1].astype(float)
        opacity = alpha[selected].copy()
        if label == 0:
            opacity *= context_opacity
        n = len(sphere)
        triangles = (faces[None]+np.arange(len(means))[:, None, None]*n).reshape(-1, 3)
        result.append(Surface(vertices.reshape(-1, 3).astype(np.float32), triangles,
            normals.reshape(-1, 3).astype(np.float32), np.repeat(colors, n, axis=0).astype(np.float32),
            int(label), np.repeat(opacity, n).astype(np.float32),bool(highlight and label>0)))
    return result


def mesh_surfaces(frame, spacing, level=.24):
    """Extract a smooth union of the actual covariance ellipsoids on a fixed lattice."""
    if not np.isfinite(spacing) or spacing<=0 or not 0<level<1:
        raise ValueError('Mesh spacing must be positive and density level between zero and one')
    result=[]
    for label in np.unique(frame['object_ids']):
        if label == 0:
            continue
        alpha=frame.get('display_alpha',np.ones(len(frame['means'])))
        selected=(frame['object_ids']==label)&(alpha>.005)
        means=frame['means'][selected].astype(np.float64)
        covs=frame['covs'][selected].astype(np.float64)
        colors=frame['colors'][selected,::-1].astype(np.float64)
        alpha=alpha[selected]
        if not len(means):
            continue
        # Observation confidence controls transparency, not the physical size of
        # a region. Multiplying density by alpha made a stationary ellipsoid
        # shrink and split as support faded. Keep its field unchanged until it
        # is nearly transparent, then retire it smoothly to avoid ghost bridges.
        geometry_weight=np.clip(alpha/.20,0,1)
        geometry_weight=geometry_weight**2*(3-2*geometry_weight)
        extent=3.5*np.sqrt(np.maximum(np.diagonal(covs,axis1=1,axis2=2),1e-8))
        lo=np.floor((means-extent).min(0)/spacing).astype(int)-1
        hi=np.ceil((means+extent).max(0)/spacing).astype(int)+1
        # Fixed lattice tiles bound memory without refitting the grid or clipping
        # distant components. Adjacent tiles share their boundary sample plane.
        tile=96
        lower=np.floor_divide(lo,tile)*tile
        inverse=np.linalg.inv(covs+np.eye(3)*1e-8)
        for x in range(lower[0],hi[0],tile):
            for y in range(lower[1],hi[1],tile):
                for z in range(lower[2],hi[2],tile):
                    origin=np.array([x,y,z])*spacing
                    end=origin+tile*spacing
                    intersects=np.all(means+extent>=origin,1)&np.all(means-extent<=end,1)
                    if not intersects.any():
                        continue
                    field=np.zeros((tile+1,)*3,np.float32)
                    for j in np.flatnonzero(intersects):
                        a=np.maximum(np.floor((means[j]-extent[j]-origin)/spacing).astype(int),0)
                        b=np.minimum(np.ceil((means[j]+extent[j]-origin)/spacing).astype(int)+1,tile+1)
                        if np.any(b<=a):
                            continue
                        grid=np.stack(np.meshgrid(*(np.arange(a[k],b[k])*spacing+origin[k]-means[j,k] for k in range(3)),indexing='ij'),-1)
                        distance=np.einsum('...i,ij,...j->...',grid,inverse[j],grid,optimize=True)
                        density=geometry_weight[j]*np.exp(-.5*distance)
                        patch=field[a[0]:b[0],a[1]:b[1],a[2]:b[2]]
                        patch[:]=1-(1-patch)*(1-density)
                    if field.max()<=level or field.min()>=level:
                        continue
                    vertices,faces,normals,_=marching_cubes(field,level=level,spacing=(spacing,)*3,allow_degenerate=False)
                    vertices+=origin
                    # Appearance is a continuous local blend, never a nearest-ID
                    # assignment that would switch color at a Voronoi boundary.
                    rgb=np.empty((len(vertices),3),np.float32)
                    normals=np.empty_like(vertices)
                    opacity=np.empty(len(vertices),np.float32)
                    for start in range(0,len(vertices),2048):
                        delta=vertices[start:start+2048,None]-means[None]
                        gaussian=np.exp(-.5*np.einsum('vni,nij,vnj->vn',delta,inverse,delta,optimize=True))
                        weights=gaussian*geometry_weight
                        opacity[start:start+2048]=(weights@alpha)/np.maximum(weights.sum(1),1e-12)
                        rgb[start:start+2048]=(weights@colors)/np.maximum(weights.sum(1,keepdims=True),1e-12)
                        gradient=np.einsum('vn,nij,vnj->vi',weights/np.maximum(1-weights,1e-6),inverse,delta,optimize=True)
                        normals[start:start+2048]=gradient/np.maximum(np.linalg.norm(gradient,axis=1,keepdims=True),1e-12)
                    result.append(Surface(vertices.astype(np.float32),faces,normals,rgb,int(label),opacity))
    return result


@njit(cache=True)
def _triangles(uv,z,faces,rgb,opacity,depth,canvas,coverage):
    h,w=depth.shape
    for face in faces:
        a,b,c=face
        if min(z[a],z[b],z[c])<=0:
            continue
        x0,y0=uv[a];x1,y1=uv[b];x2,y2=uv[c]
        area=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(area)<1e-8:
            continue
        left=max(0,int(np.floor(min(x0,x1,x2))));right=min(w-1,int(np.ceil(max(x0,x1,x2))))
        top=max(0,int(np.floor(min(y0,y1,y2))));bottom=min(h-1,int(np.ceil(max(y0,y1,y2))))
        for y in range(top,bottom+1):
            for x in range(left,right+1):
                u=((y1-y2)*(x+.5-x2)+(x2-x1)*(y+.5-y2))/area
                v=((y2-y0)*(x+.5-x2)+(x0-x2)*(y+.5-y2))/area
                q=1-u-v
                if min(u,v,q)<-1e-6:
                    continue
                iz=u/z[a]+v/z[b]+q/z[c]
                d=1/iz
                if d>=depth[y,x]:
                    continue
                u=u/z[a]/iz;v=v/z[b]/iz;q=q/z[c]/iz
                alpha=u*opacity[a]+v*opacity[b]+q*opacity[c]
                if alpha<=.001:
                    continue
                depth[y,x]=d
                coverage[y,x]=alpha
                for k in range(3):
                    canvas[y,x,k]=u*rgb[a,k]+v*rgb[b,k]+q*rgb[c,k]


def raster_surfaces(surfaces,camera,*,semantic=True,near=.05,antialias=2,background=None):
    """Perspective-correct depth-buffered triangles, shaded with smooth normals."""
    from .render import color_for
    h,w=camera.height*antialias,camera.width*antialias
    background=(np.full((h,w,3),255.,np.float32) if background is None else
                cv2.resize(background,(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32))
    light=np.array([-.35,-.50,-.79]);light/=np.linalg.norm(light)
    layers=[]
    # Keep each object's nearest surface, then composite objects in per-pixel
    # depth order. A fading foreground object must reveal the object behind it,
    # rather than replace it with the white backdrop.
    for label in sorted({s.object_id for s in surfaces}):
        depth=np.full((h,w),np.inf,np.float32)
        color=np.zeros((h,w,3),np.float32);coverage=np.zeros((h,w),np.float32)
        for surface in surfaces:
            if surface.object_id!=label:
                continue
            uv,z=camera.project(surface.vertices)
            normal=surface.normals@camera.rotation.T
            diffuse=np.clip(normal@light,0,1)
            colors=surface.colors
            # Keep measured appearance in each particle's interior. A bold
            # object-colored rim adds identity on top, not in place of its RGB.
            # Background has the same appearance/shading model, without tint.
            colors=np.clip(colors*(.68+.32*diffuse[:,None])+6*diffuse[:,None]**16,0,255)
            if (semantic or surface.highlight) and surface.object_id>0:
                view=(camera.position-surface.vertices)@camera.rotation.T
                view/=np.maximum(np.linalg.norm(view,axis=1,keepdims=True),1e-8)
                facing=np.abs(np.einsum('ij,ij->i',normal,view))
                rim=.18+.82*(1-np.clip(facing,0,1))**1.15
                colors=colors*(1-rim[:,None])+color_for(surface.object_id)*rim[:,None]
            alpha=np.clip((z-near)/max(near*.5,.025),0,1)
            alpha=alpha*alpha*(3-2*alpha)*.97*surface.opacity
            _triangles((uv*antialias).astype(np.float32),z.astype(np.float32),surface.faces,
                       colors.astype(np.float32),alpha.astype(np.float32),depth,color,coverage)
        highlighted=label>0 and (semantic or any(s.highlight for s in surfaces if s.object_id==label))
        if highlighted:
            # A continuous alpha outline emphasizes the object without painting
            # over its appearance. No thresholded silhouette can blink on/off.
            r=max(1,round(1.5*antialias))
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
            expanded=cv2.dilate(coverage,kernel)
            outline=np.maximum(expanded-coverage,0.)
            combined=coverage+outline
            color=(color*coverage[...,None]+color_for(label)*outline[...,None])/np.maximum(combined[...,None],1e-8)
            coverage=combined
            depth=np.where(outline>0,cv2.erode(depth,kernel),depth)
        layers.append((depth,color,coverage))
    canvas=background.copy()
    if layers:
        order=np.argsort(np.stack([a[0] for a in layers]),axis=0)[::-1]
        colors=np.stack([a[1] for a in layers]);alpha=np.stack([a[2] for a in layers])
        labels=sorted({s.object_id for s in surfaces})
        if 0 in labels:
            highlighted=[k for k,label in enumerate(labels) if label>0 and
                (semantic or any(s.highlight for s in surfaces if s.object_id==label))]
            if highlighted:
                # Keep contextual RGB while making material in front of the
                # highlighted object more transparent, like a cutaway view.
                mask=np.max(alpha[highlighted],axis=0)
                alpha[labels.index(0)]*=1-.65*mask
        yy,xx=np.indices((h,w))
        for rank in order:
            a=alpha[rank,yy,xx,None]
            canvas=canvas*(1-a)+colors[rank,yy,xx]*a
    image=canvas.clip(0,255).astype(np.uint8)
    if antialias>1:
        image=cv2.resize(image,(camera.width,camera.height),interpolation=cv2.INTER_AREA)
    return image
