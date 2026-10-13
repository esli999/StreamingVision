import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified.data import Capture, fit_frame, labels_from_mask, rate, select_foreground, resolve_clip, iter_frames
from unified.geometry import (Camera, orbit_camera, warp_previous, measured_motion,
                              correspondence_confidence, align_depth, rigid_alignment,sharpen_object_depth)
from unified.metrics import j_and_f, native_labels
from unified.tracker import sample_indices, predict_particles


def test_cameras_share_extrinsics_and_perspective_covariance():
    target = np.array([.2,.1,3.])
    camera = orbit_camera(target,15,0,3,500,640,360)
    uv,z = camera.project(target[None])
    np.testing.assert_allclose(uv,[[320,180]],atol=1e-7)
    np.testing.assert_allclose(z,[3.])
    origin = Camera(np.zeros(3),np.eye(3),500,640,360)
    footprints = origin.footprints(np.array([[0,0,2.],[0,0,4.]]),np.tile(np.eye(3)*.01,(2,1,1)))
    np.testing.assert_allclose(footprints[0]-.5*np.eye(2),4*(footprints[1]-.5*np.eye(2)))


def test_motion_uses_corresponding_pixels_and_rejects_occlusion():
    h,w=20,30
    yy,xx=np.mgrid[:h,:w]
    previous=np.stack([xx,yy,xx*.1+1],-1).astype(np.float32)
    backward=np.zeros((2,h,w),np.float32);backward[0]=-2
    forward=-backward
    current,_=warp_previous(previous,backward)
    current[...,1]+=.3
    valid=correspondence_confidence(forward,backward)
    assert not valid[:,:2].any()
    v=measured_motion(current,previous,backward,valid)
    np.testing.assert_allclose(v[valid],np.tile([0,.3,0],(valid.sum(),1)),atol=1e-6)
    forward[:,7:12,7:12]=20
    assert not correspondence_confidence(forward,backward)[8:11,10:13].any()


def test_depth_alignment_follows_flow_and_ignores_moving_foreground():
    yy,xx=np.mgrid[:80,:120]
    old=(xx*.1+yy*.05).astype(np.float32)
    flow=np.zeros((2,80,120),np.float32);flow[0]=-4
    reference,valid=warp_previous(old,flow)
    raw=(reference-2)/1.2
    background=np.ones_like(valid);background[25:55,30:80]=False
    raw[~background]+=30
    aligned,bounds=align_depth(raw,old,flow,valid,background,(0,20))
    np.testing.assert_allclose(aligned[valid&background],reference[valid&background],atol=1e-4)
    assert bounds==(0,20)


def test_background_rigid_alignment_handles_outliers():
    rng=np.random.default_rng(2)
    p=rng.normal(size=(30,40,3))*.2
    angle=.03
    r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    t=np.array([.01,.02,-.03]);q=p@r.T+t
    q[:3]+=1
    got_r,got_t=rigid_alignment(p,q,np.ones((30,40),bool))
    np.testing.assert_allclose(got_r,r,atol=.004)
    np.testing.assert_allclose(got_t,t,atol=.004)


def test_rigid_alignment_is_exact_for_asymmetric_scenes():
    rng=np.random.default_rng(9)
    p=rng.exponential(.2,(30,40,3))
    angle=.1
    r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    t=np.array([.01,.02,-.03]);q=p@r.T+t
    got_r,got_t=rigid_alignment(p,q,np.ones((30,40),bool))
    np.testing.assert_allclose(got_r,r,atol=1e-7)
    np.testing.assert_allclose(got_t,t,atol=1e-7)


def test_mask_identity_and_letterbox_roundtrip(tmp_path):
    m=np.zeros((40,30),np.uint16);m[10:30,8:20]=513
    path=tmp_path/'mask.png';Image.fromarray(m).save(path)
    selected,ids=select_foreground(labels_from_mask(path),[513])
    assert ids==[513] and selected.max()==1
    fitted,valid=fit_frame(selected,nearest=True)
    np.testing.assert_array_equal(native_labels(fitted,m.shape),selected)
    assert not fitted[~valid].any()


def test_metrics_penalize_swaps_missing_objects_and_boundaries():
    a=np.zeros((100,100),bool);a[10:60,10:60]=True
    assert j_and_f(a,a)==(1.,1.)
    assert j_and_f(np.zeros_like(a),a)==(0.,0.)
    b=np.roll(a,8,axis=1)
    j,f=j_and_f(b,a)
    assert 0<j<.8 and 0<f<.8
    assert j_and_f(~a,a)[0]==0


def test_timing_is_completed_frames_over_total_time():
    assert rate([.1,.9])==2.
    assert rate([]) is None
    assert rate([float('nan'),0,-1]) is None


def test_capture_detects_incomplete_and_missing_states(tmp_path):
    c=Capture(tmp_path/'capture',dict(clip='synthetic'))
    c.write(0,means=np.ones((3,3)))
    with pytest.raises(ValueError,match='Incomplete'):
        Capture(c.path)
    c.finish(frames=1)
    np.testing.assert_array_equal(Capture(c.path).read(0)['means'],np.ones((3,3)))
    (c.path/'frame_00000.npz').unlink()
    with pytest.raises(ValueError,match='frame count'):
        Capture(c.path)


def test_offline_renderer_does_not_load_inference_models():
    import subprocess
    command="import sys; import unified.render; assert not ({'torch','jax','genmatter_rt','render_demo'} & set(sys.modules))"
    subprocess.run([sys.executable,'-c',command],check=True,cwd=Path(__file__).resolve().parents[1])


def test_sampling_preserves_small_foreground_and_exact_batch_size():
    labels=np.zeros((90,160),np.uint16);labels[20:30,20:30]=1
    ix=sample_indices(labels)
    assert len(ix)==2925 and len(np.unique(ix))==2925
    assert np.sum(labels.reshape(-1)[ix]>0)==100
    np.testing.assert_array_equal(ix,sample_indices(labels))


def test_particle_prediction_uses_measured_motion_not_free_running_velocity():
    h,w=360,640
    yy,xx=np.mgrid[:h,:w]
    old=np.stack([(xx-320)/500,(yy-180)/500,np.ones((h,w))],-1).astype(np.float32)
    new=old.copy();new[...,2]+=.1
    flow=np.zeros((2,h,w),np.float32);flow[0]=10
    labels=np.ones((h,w),np.int32)
    means=np.array([[0,0,1]],np.float32)
    pred,motion=predict_particles(means,np.array([1]),old,new,labels,labels,flow,
        np.ones((h,w),bool),np.eye(3),np.zeros(3),np.zeros_like(old))
    np.testing.assert_allclose(pred,[[.02,0,1.1]],atol=1e-6)
    np.testing.assert_allclose(motion,[[.02,0,.1]],atol=1e-6)


def test_dino_register_tokens_are_never_interpreted_as_image_patches():
    import torch
    from types import SimpleNamespace
    from streaming_dino import dino_patches
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.ones(1))
            self.config=SimpleNamespace(num_register_tokens=4)
        def forward(self,**kwargs):
            return SimpleNamespace(last_hidden_state=torch.arange(9*3).reshape(1,9,3).float())
    patches,shape=dino_patches(Model(),np.zeros((28,28,3),np.uint8),device='cpu',dino_h=28,dino_w=28)
    assert shape==(2,2)
    np.testing.assert_array_equal(patches,np.arange(15,27).reshape(4,3))


def test_davis_eight_object_palette_has_no_identity_collisions():
    from unified.render import color_for
    assert len(np.unique(color_for(np.arange(9)),axis=0))==9


def test_novel_views_orbit_the_actual_input_camera_for_off_center_objects():
    from unified.render import views
    frame=dict(world_translation=np.array([.2,0,.1]),world_rotation=np.eye(3))
    target=np.array([1.,0.,3.])
    source,novel=views(frame,target)
    baseline=source.position-target
    for camera in novel:
        offset=camera.position-target
        np.testing.assert_allclose(np.linalg.norm(offset),np.linalg.norm(baseline))
        angle=np.arccos(np.dot(offset,baseline)/np.linalg.norm(offset)/np.linalg.norm(baseline))
        np.testing.assert_allclose(np.rad2deg(angle),15.,atol=1e-6)
        np.testing.assert_allclose(camera.project(target[None])[0],[[320,180]],atol=1e-6)


def test_raw_frame_directory_and_explicit_seed_override(tmp_path,monkeypatch):
    frames=tmp_path/'frames';frames.mkdir()
    cv2.imwrite(str(frames/'00000.jpg'),np.zeros((20,30,3),np.uint8))
    mask=tmp_path/'initial.png';Image.fromarray(np.ones((20,30),np.uint8)).save(mask)
    monkeypatch.chdir(tmp_path)
    clip=resolve_clip('frames',seed_mask=mask)
    assert clip.name=='frames' and clip.mask==mask
    frame,timestamp=next(iter_frames(clip.source,with_time=True,fps=24.))
    assert frame.shape==(20,30,3) and timestamp==0.
    data=tmp_path/'DAVIS';src=data/'JPEGImages/480p/clip';ann=data/'Annotations/480p/clip'
    src.mkdir(parents=True);ann.mkdir(parents=True)
    Image.fromarray(np.ones((20,30),np.uint8)).save(ann/'00000.png')
    named=resolve_clip('clip',data_root=data,seed_mask=mask)
    assert named.mask==mask and named.provenance=='user initial mask'


def test_object_depth_boundary_repair_preserves_interior_and_background():
    depth=np.ones((80,100),np.float32)
    labels=np.zeros((80,100),np.int32);labels[20:60,30:70]=1
    depth[23:57,33:67]=5.
    fixed=sharpen_object_depth(depth,labels)
    np.testing.assert_array_equal(fixed[labels==0],depth[labels==0])
    np.testing.assert_array_equal(fixed[23:57,33:67],depth[23:57,33:67])
    np.testing.assert_allclose(fixed[labels==1],5.)
    # Smooth, plausible foreground depth must remain untouched.
    depth[20:60,30:70]=5.
    np.testing.assert_array_equal(sharpen_object_depth(depth,labels),depth)


def test_rgb_mask_ids_survive_another_object_disappearing(tmp_path):
    first=np.zeros((10,20,3),np.uint8)
    first[:,:10]=[255,0,0];first[:,10:]=[0,0,255]
    later=first.copy();later[:,10:]=0
    a,b=tmp_path/'first.png',tmp_path/'later.png'
    Image.fromarray(first).save(a);Image.fromarray(later).save(b)
    x,y=labels_from_mask(a),labels_from_mask(b)
    assert x[0,0]==y[0,0] and x[0,10]!=255 and y[0,10]==0


def test_camera_fit_holds_foreground_framing_when_the_subject_exits():
    from unified.presentation import Presentation
    class Saved:
        manifest={'frames':3,'source_fps':30.}
        def read(self,i):
            return dict(means=np.array([[-.2,0,3],[.2,0,3],[0,.2,3],[20,20,30]]),
                        covs=np.repeat(np.eye(3)[None]*.01,4,axis=0),colors=np.zeros((4,3),np.uint8),
                        object_ids=np.array([1,1,1,0]),weights=np.ones(4)/4,
                        support=np.array([4,4,4,100]) if i==0 else np.array([0,0,0,100]))
    p=Presentation(Saved())
    assert p.radius < .5
    np.testing.assert_allclose(p.targets,np.tile(p.targets[0],(3,1)))


def test_explicit_bundled_source_is_preserved_when_resolving_its_initial_mask(tmp_path,monkeypatch):
    import unified.data as data
    monkeypatch.setattr(data,'ROOT',tmp_path)
    source=tmp_path/'assets/test.mp4';source.parent.mkdir();source.touch()
    other=tmp_path/'assets/custom_videos/test/source.mp4';other.parent.mkdir(parents=True);other.touch()
    mask=other.parent/'pseudo_gt_sam/segmasks/test/00000.png';mask.parent.mkdir(parents=True)
    Image.fromarray(np.ones((10,10),np.uint8)).save(mask)
    clip=resolve_clip(str(source),data_root=tmp_path/'missing')
    assert clip.source==source and clip.mask==mask
    other.unlink()
    assert resolve_clip(str(source),data_root=tmp_path/'missing').source==source
