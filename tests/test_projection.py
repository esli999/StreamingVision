"""Conditioned readouts are co-registered 3D projections, not pasted input RGB."""
import numpy as np
from unified.geometry import Camera
from unified.projection import readout_surface,project_readouts,choose_view


def frame():
    labels=np.zeros((360,640),np.uint16);labels[101:261,241:401]=1
    labels[79:101,321]=1  # a one-pixel appendage cannot vanish in a coarse grid
    reconstruction=np.full((360,640,3),[40,80,120],np.uint8)
    reconstruction[labels>0]=[20,150,230]
    return dict(labels=labels,reconstruction=reconstruction,bgr=np.zeros_like(reconstruction),
        means=np.array([[0.,0.,4.],[0.,0.,2.]]),
        covs=np.array([np.diag([4.,4.,.05]),np.diag([.1,.1,.01])]),
        weights=np.array([.5,.5]),object_ids=np.array([0,1]),display_alpha=np.ones(2))


def test_observed_camera_recovers_posterior_rgb_and_binary_membership():
    f=frame();surface=readout_surface(f)
    camera=Camera(np.zeros(3),np.eye(3),500.,640,360)
    rgb,seg,mask,coverage=project_readouts(surface,camera)
    np.testing.assert_array_equal(mask,f['labels']>0)
    np.testing.assert_allclose(coverage,1)
    np.testing.assert_array_equal(rgb[180,320],f['reconstruction'][180,320,::-1])
    np.testing.assert_array_equal(rgb[20,20],f['reconstruction'][20,20,::-1])
    assert len(np.unique(seg.reshape(-1,3),axis=0))==2
    assert rgb.max()>0  # raw input RGB is zero and must not be used as a texture


def test_nearby_projection_changes_with_3d_depth_and_rgb_matches_membership():
    f=frame();camera=Camera(np.array([.12,0.,.04]),np.eye(3),500.,640,360)
    rgb,_,near_mask,coverage=project_readouts(readout_surface(f),camera)
    foreground_color=f['reconstruction'][180,320,::-1]
    np.testing.assert_array_equal(rgb[near_mask& (coverage==1)][100],foreground_color)
    f['means'][1,2]=3.
    _,_,far_mask,_=project_readouts(readout_surface(f),camera)
    assert np.count_nonzero(near_mask!=far_mask)>100
    # Moving matter farther away changes its parallax without relabeling the
    # original observed projection; this cannot be achieved by copying a mask.
    source=Camera(np.zeros(3),np.eye(3),500.,640,360)
    np.testing.assert_array_equal(project_readouts(readout_surface(f),source)[2],f['labels']>0)


def test_no_geometry_is_created_for_an_unrepresented_object():
    f=frame();f['display_alpha'][1]=0
    surface=readout_surface(f)
    camera=Camera(np.zeros(3),np.eye(3),500.,640,360)
    _,seg,mask,coverage=project_readouts(surface,camera)
    assert not mask.any()
    assert coverage[180,320]==0
    np.testing.assert_array_equal(seg[180,320],[255,255,255])


def test_selection_rejects_a_view_if_a_late_frame_is_incomplete():
    class Saved:
        manifest={'frames':2}
        def read(self,i):
            f=frame()
            if i: f['display_alpha'][:]=0
            return f
    class Display:
        stabilized=True
        manifest={'clip':'synthetic'}
        novel_target=np.array([0.,0.,2.])
        source_camera=Camera(np.zeros(3),np.eye(3),500.,640,360)
        def frame(self,f,i): return f
    camera,report=choose_view(Saved(),Display())
    assert camera is Display.source_camera
    assert not report['novel_view_available']
    assert report['checked_frames']==2
    assert not any(report['passed'])
