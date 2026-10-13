"""Motion direction, timing provenance and perspective frustum regressions."""
import numpy as np
from unified.motion import motion_vectors, rollout_rates, scene_motion
from unified.render import frustum_points,scene_cameras,VIEW_COLORS,DEFAULT_COLOR
from unified.geometry import orbit_camera,Camera


def test_current_anchored_motion_points_from_previous_to_current_pixel():
    # Backward correspondence says the previous pixel was six pixels left.
    frame=dict(backward_grid=np.broadcast_to(np.array([-.75,0])[:,None,None],(2,45,80)).copy(),
               reliable_grid=np.ones((45,80),bool),bgr=np.zeros((360,640,3),np.uint8))
    vectors,valid=motion_vectors(frame)
    np.testing.assert_array_equal(vectors,np.broadcast_to([6.,0.],vectors.shape))
    assert valid.all()
    frame['reliable_grid'][10,20]=False
    vectors,valid=motion_vectors(frame)
    np.testing.assert_array_equal(vectors[10,20],[0.,0.])
    assert not valid[10,20]
    image=scene_motion(frame,6.)
    assert image.shape==(360,640,3) and image.dtype==np.uint8


def test_rollout_fps_uses_completed_frames_over_engine_time_and_is_causal():
    times=np.array([10.,5.,2.,.1,.2,.4,.1,.1])
    rates=rollout_rates(times,warmup=3,window=3)
    assert np.isnan(rates[:3]).all()
    np.testing.assert_allclose(rates[3:],[10.,2/.3,3/.7,3/.7,3/.6])
    np.testing.assert_array_equal(rollout_rates(times[:6],3,3),rates[:6])
    assert np.isnan(rollout_rates([1.,1.,1.,np.nan])[3])


def test_frustum_has_volume_and_projects_to_the_cameras_actual_image_corners():
    target=np.array([.3,.2,3.])
    camera=orbit_camera(target,-23,17,3.,500.,640,360)
    points=frustum_points(camera,target)
    uv,z=camera.project(points[1:])
    expected=np.array([[0,0],[640,0],[640,360],[0,360]])
    np.testing.assert_allclose(uv,np.tile(expected,(2,1)),atol=1e-9)
    assert np.linalg.matrix_rank(points[1:]-camera.position)==3
    assert np.all(z[:4]>0) and np.all(z[4:]>z[:4])


def test_overlapping_frustum_faces_cannot_erase_another_views_badge():
    target=np.array([0.,0.,3.])
    viewer=orbit_camera(target,24,16,6.,650.,1320,792)
    cameras=[Camera(np.array([0.,0.,z]),np.eye(3),500.,640,360) for z in (0.,.9)]
    default=Camera(np.array([0.,0.,.45]),np.eye(3),500.,640,360)
    image=scene_cameras(np.full((792,1320,3),255,np.uint8),viewer,cameras,target,default_camera=default)
    for color in VIEW_COLORS[:2]+[DEFAULT_COLOR]:
        # Each opaque circular badge must survive the translucent face pass.
        assert np.count_nonzero(np.all(image==color,axis=-1))>300
