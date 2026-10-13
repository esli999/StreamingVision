"""Regressions for render-induced jitter, support flicker and depth-order pops."""
import numpy as np
from scipy.spatial.transform import Rotation
from unified.geometry import Camera
from unified.presentation import camera_coordinates,visibility_trace,Presentation,playback_fps
from unified.render import splat_particles


def particle_frame():
    return dict(means=np.array([[-.15,0.,3.],[.15,0.,3.],[0,.2,3.2]]),
                covs=np.repeat(np.diag([.035,.025,.018])[None],3,0),
                object_ids=np.array([1,1,2]),support=np.array([3.,5.,4.]),
                weights=np.ones(3)/3,colors=np.array([[30,90,210],[190,70,35],[60,190,80]],np.uint8),
                world_rotation=np.eye(3),world_translation=np.zeros(3))


def rebase(frame,r,t):
    return dict(frame,means=frame['means']@r.T+t,
                covs=np.einsum('ij,njk,lk->nil',r,frame['covs'],r),
                world_rotation=r,world_translation=t)


def test_camera_rebasing_preserves_projection_extent_and_distances():
    original=particle_frame()
    r=Rotation.from_rotvec([.4,-.8,.1]).as_matrix();t=np.array([1.,2.,-.3])
    moved=rebase(original,r,t);local=camera_coordinates(moved)
    source=Camera(t,r.T,500,640,360);neutral=Camera(np.zeros(3),np.eye(3),500,640,360)
    np.testing.assert_allclose(source.project(moved['means'])[0],neutral.project(local['means'])[0])
    np.testing.assert_allclose(source.footprints(moved['means'],moved['covs']),neutral.footprints(local['means'],local['covs']))
    np.testing.assert_allclose(local['means'],original['means'],atol=1e-12)
    np.testing.assert_allclose(local['covs'],original['covs'],atol=1e-12)


def test_arbitrary_per_frame_world_pose_cannot_jitter_the_display_camera():
    class Saved:
        manifest={'frames':30,'source_fps':30.}
        def __init__(self,jitter): self.jitter=jitter
        def read(self,i):
            f=particle_frame()
            if self.jitter:
                r=Rotation.from_rotvec(np.random.default_rng(i).normal(0,.2,3)).as_matrix()
                return rebase(f,r,np.array([i*.1,(-1)**i*.4,.1*i]))
            return f
    a,b=Presentation(Saved(False)),Presentation(Saved(True))
    np.testing.assert_allclose(a.targets,b.targets,atol=1e-12)
    np.testing.assert_allclose(a.radius,b.radius,atol=1e-12)
    np.testing.assert_allclose(np.diff(b.targets,n=2,axis=0),0,atol=1e-12)


def test_support_gaps_hold_identity_but_persistent_failure_fades():
    s=np.full((100,2),3.);s[:,1]=0;s[10:13,0]=0;s[20:,0]=0
    alpha,age=visibility_trace(s,30.)
    assert np.all(alpha[:,1]==0)  # no evidence cannot create matter
    assert np.all(alpha[10:13,0]>.95)  # brief occlusion does not delete matter
    assert alpha[-1,0]<.002  # no indefinitely floating ghosts
    assert np.abs(np.diff(alpha,axis=0)).max()<.3
    assert age[13,0]==0 and age[-1,0]>2.
    alpha60,_=visibility_trace(np.repeat(s,2,axis=0),60.)
    np.testing.assert_allclose(alpha60[1::2],alpha,atol=.04)


def test_depth_compositing_is_order_independent_and_continuous_at_crossing():
    camera=Camera(np.zeros(3),np.eye(3),120.,160,100)
    f=particle_frame();f['means'][:2]=[0,0,3];f['display_alpha']=np.ones(3)
    a=splat_particles(f,camera)
    reverse={k:(v[::-1] if isinstance(v,np.ndarray) and k not in ('world_rotation','world_translation') else v) for k,v in f.items()}
    b=splat_particles(reverse,camera)
    assert np.abs(a.astype(float)-b).max()<=1
    f['means'][0,2]-=1e-5
    left=splat_particles(f,camera)
    f['means'][0,2]+=2e-5
    right=splat_particles(f,camera)
    assert np.abs(left.astype(float)-right).max()<=1


def test_occluded_particle_retains_its_last_observed_color():
    class Saved:
        manifest={'frames':4,'source_fps':30.}
        def read(self,i):
            f=particle_frame()
            if i:
                f['support'][0]=0
                f['colors'][0]=[0,0,0]
            return f
    p=Presentation(Saved())
    np.testing.assert_array_equal(p.colors[:,0],np.tile(particle_frame()['colors'][0],(4,1)))


def test_decoded_timestamps_override_stale_trimmed_video_metadata():
    assert abs(playback_fps(np.arange(36)/30,24.977)-30)<1e-9
    assert playback_fps([0.],24.)==24.
    assert playback_fps([0.,0.,0.],24.)==24.


def test_near_camera_unsupported_geometry_cannot_fill_the_whole_image():
    f=particle_frame();f['means'][:,2]=.1;f['display_alpha']=np.ones(3)
    camera=Camera(np.zeros(3),np.eye(3),120.,160,100)
    assert np.all(splat_particles(f,camera,near=.5)==255)
    f['means'][:,2]=.50001
    # Crossing the near plane fades in continuously, even for a large footprint.
    assert splat_particles(f,camera,near=.5).min()>=254


def test_receding_subject_stays_in_front_of_the_fixed_distance_viewer():
    from unified.geometry import orbit_camera
    class Saved:
        manifest={'frames':60,'source_fps':30.}
        def read(self,i):
            f=particle_frame();f['means'][:,2]+=i*.07-1.5
            return f
    p=Presentation(Saved())
    for i in range(60):
        camera=orbit_camera(p.targets[i],14,-9,p.radius*2.65,790,850,562)
        uv,z=camera.project(p.means[i])
        assert np.all(z>p.radius*.25)
        assert np.all((uv>[0,0])&(uv<[850,562]))


def test_slide_camera_follows_smoothly_and_keeps_the_foreground_legible_despite_distant_context():
    from unified.surface import particle_surfaces
    class Saved:
        manifest={'frames':12,'source_fps':30.}
        def read(self,i):
            f=particle_frame()
            for key in ('means','covs','object_ids','support','weights','colors'):
                f[key]=np.concatenate([f[key],f[key][-1:]])
            f['object_ids'][3]=0  # far background is drawn, but need not fit
            f['means'][3]=[3.,-1.5,6.]
            f['means'][:3]+=[i*.07,0.,i*.04]
            return f
    capture=Saved();p=Presentation(capture)
    for i in range(12):
        frame=p.frame(capture.read(i),i)
        viewer=frame['scene_camera']
        np.testing.assert_array_equal(viewer.rotation,p.scene_camera.rotation)
        assert viewer.focal==p.scene_camera.focal
        assert frame['novel_camera'] is p.novel_camera
        for surface in particle_surfaces(frame):
            if surface.object_id==0:
                continue  # distant context may be cropped; it cannot shrink the animal
            uv,z=viewer.project(surface.vertices)
            assert np.all(z>.05)
            assert np.all(uv>[0,0]) and np.all(uv<[p.scene_camera.width,p.scene_camera.height])
    assert not np.allclose(p.scene_positions[0],p.scene_positions[-1])
    # The followed object stays near the center as it moves and recedes.
    for i in range(12):
        uv,_=p.frame(capture.read(i),i)['scene_camera'].project(p.scene_targets[i:i+1])
        np.testing.assert_allclose(uv[0],[1220,500],atol=1e-8)
    class RemoteContext(Saved):
        def read(self,i):
            f=super().read(i)
            f['means'][3]=[300.,-150.,600.]
            return f
    distant=Presentation(RemoteContext())
    np.testing.assert_array_equal(p.scene_camera.position,distant.scene_camera.position)
    np.testing.assert_array_equal(p.scene_camera.rotation,distant.scene_camera.rotation)
    assert p.scene_camera.focal==distant.scene_camera.focal
