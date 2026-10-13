"""A derived mesh must preserve identities and avoid lattice/depth-order pops."""
import numpy as np
from unified.geometry import Camera
from unified.surface import mesh_surfaces,raster_surfaces,particle_surfaces


def state():
    return dict(means=np.array([[-.12,0,2.9],[.12,0,3.0]]),
                covs=np.repeat(np.diag([.03,.02,.01])[None],2,0),
                colors=np.array([[20,70,200],[200,70,20]],np.uint8),
                object_ids=np.array([1,2]),display_alpha=np.ones(2))


def test_mesh_keeps_object_fields_separate_and_raster_depth_is_order_independent():
    f=state();s=mesh_surfaces(f,.04)
    assert {m.object_id for m in s}=={1,2}
    assert all(len(m.faces)>0 and np.isfinite(m.normals).all() for m in s)
    cam=Camera(np.zeros(3),np.eye(3),150,180,120)
    a=raster_surfaces(s,cam);b=raster_surfaces(s[::-1],cam)
    np.testing.assert_array_equal(a,b)
    assert a.min()<100


def test_subvoxel_motion_is_continuous_at_a_tile_boundary():
    f=state();f['means'][:,0]=[-.00001,.12]
    cam=Camera(np.zeros(3),np.eye(3),150,180,120)
    a=raster_surfaces(mesh_surfaces(f,.04),cam)
    f['means'][0,0]+=.00002
    b=raster_surfaces(mesh_surfaces(f,.04),cam)
    assert np.abs(a.astype(float)-b).mean()<.02
    # Flat normals at a grid boundary would introduce visible lighting seams.
    s=mesh_surfaces(f,.04)
    for m in s:
        np.testing.assert_allclose(np.linalg.norm(m.normals,axis=1),1,atol=1e-5)


def test_unsupported_matter_cannot_create_an_opaque_surface():
    f=state();f['display_alpha'][:]=0
    assert mesh_surfaces(f,.04)==[]
    cam=Camera(np.zeros(3),np.eye(3),150,180,120)
    assert np.all(raster_surfaces([],cam)==255)


def test_mesh_confidence_fades_opacity_and_reveals_the_object_behind():
    f=state();f['means'][:]=[[0,0,2.7],[0,0,3.1]]
    cam=Camera(np.zeros(3),np.eye(3),150,180,120)
    meshes=mesh_surfaces(f,.04)
    back=[s for s in meshes if s.object_id==2]
    behind=raster_surfaces(back,cam)
    opaque=raster_surfaces(meshes,cam)
    for surface in meshes:
        if surface.object_id==1:
            surface.opacity[:]=.01
    fading=raster_surfaces(meshes,cam)
    assert np.abs(fading.astype(float)-behind).mean()<np.abs(opaque.astype(float)-behind).mean()*.1
    f['display_alpha'][:]=.5
    for surface in mesh_surfaces(f,.04):
        np.testing.assert_allclose(surface.opacity,.5,atol=1e-6)


def test_visibility_changes_do_not_deform_or_recolor_stationary_matter():
    f=state();f['object_ids'][:]=1
    original=mesh_surfaces(f,.04)
    # Different observation counts on overlapping regions used to change the
    # silhouette, normals and colors despite identical captured geometry.
    f['display_alpha'][:]=[.25,.7]
    fading=mesh_surfaces(f,.04)
    assert len(original)==len(fading)
    for a,b in zip(original,fading):
        np.testing.assert_array_equal(a.faces,b.faces)
        np.testing.assert_array_equal(a.vertices,b.vertices)
        np.testing.assert_allclose(a.normals,b.normals)
        np.testing.assert_allclose(a.colors,b.colors)
        assert b.opacity.max()<a.opacity.min()
    # Retirement below 20% opacity remains finite and smoothly invisible.
    cam=Camera(np.zeros(3),np.eye(3),150,180,120)
    f['display_alpha'][:]=.01
    almost_gone=raster_surfaces(mesh_surfaces(f,.04),cam)
    assert almost_gone.min()>=252


def test_particle_mesh_preserves_covariance_and_opacity_cannot_deform_it():
    from scipy.spatial.transform import Rotation
    f=state();f['object_ids'][0]=0
    rotation=Rotation.from_rotvec([.2,.6,-.3]).as_matrix()
    f['covs']=np.einsum('ij,njk,lk->nil',rotation,f['covs'],rotation)
    original=particle_surfaces(f)
    assert {s.object_id for s in original}=={0,2}
    for surface in original:
        k=int(np.flatnonzero(f['object_ids']==surface.object_id)[0])
        delta=surface.vertices-f['means'][k]
        precision=np.linalg.inv(f['covs'][k])
        mahal=np.einsum('ni,ij,nj->n',delta,precision,delta)
        np.testing.assert_allclose(mahal,1.8**2,atol=1e-5)
        normal=delta@precision
        normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        np.testing.assert_allclose(surface.normals,normal,atol=3e-6)
    f['display_alpha'][:]=.25
    faded=particle_surfaces(f)
    for a,b in zip(original,faded):
        np.testing.assert_array_equal(a.vertices,b.vertices)
        np.testing.assert_array_equal(a.normals,b.normals)
        np.testing.assert_array_equal(a.colors,b.colors)
        np.testing.assert_allclose(b.opacity,a.opacity*.25)


def test_nearly_equal_covariance_axes_do_not_flip_the_particle_tessellation():
    f=state();f['covs'][:]=np.eye(3)*.02
    f['covs'][:,0,0]-=1e-8
    a=particle_surfaces(f)
    f['covs'][:,0,0]+=2e-8
    b=particle_surfaces(f)
    for left,right in zip(a,b):
        np.testing.assert_array_equal(left.faces,right.faces)
        assert np.abs(left.vertices-right.vertices).max()<1e-6


def test_object_highlight_preserves_both_object_and_background_rgb():
    f=state();f['object_ids'][0]=0
    plain=particle_surfaces(f,highlight=False)
    highlighted=particle_surfaces(f,highlight=True)
    for a,b in zip(plain,highlighted):
        k=int(np.flatnonzero(f['object_ids']==a.object_id)[0])
        expected=np.broadcast_to(f['colors'][k,::-1],a.colors.shape)
        np.testing.assert_array_equal(a.colors,expected)
        np.testing.assert_array_equal(b.colors,expected)
        np.testing.assert_array_equal(a.vertices,b.vertices)
        assert b.highlight==(b.object_id>0)
