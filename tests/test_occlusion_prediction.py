"""A particle behind another surface must not follow that surface's optical flow."""
import numpy as np
from unified.tracker import predict_particles


def test_same_object_occluder_does_not_drag_a_particle_toward_the_camera():
    h,w=360,640
    yy,xx=np.mgrid[:h,:w]
    old=np.stack([(xx-320)/500,(yy-180)/500,np.ones((h,w))],-1).astype(np.float32)
    new=old+np.array([0,0,-.4],np.float32)
    labels=np.ones((h,w),np.int32);flow=np.zeros((2,h,w),np.float32)
    velocity=np.zeros((h,w,3),np.float32);velocity[...,0]=.03
    # Both visible and hidden surfaces belong to the same object identity.
    means=np.array([[0.,0.,3.],[0.,0.,1.]],np.float32)
    covs=np.repeat((np.eye(3)*.002)[None],2,0)
    predicted,shift=predict_particles(means,np.ones(2,int),old,new,labels,labels,
        flow,np.ones((h,w),bool),np.eye(3),np.zeros(3),velocity,covs)
    np.testing.assert_allclose(shift[0],[0,0,0],atol=1e-6)
    np.testing.assert_allclose(shift[1],[0,0,-.4],atol=1e-6)
    assert predicted[0,2]==3.


def test_uncertain_surface_depth_preserves_reliable_image_motion():
    h,w=360,640;yy,xx=np.mgrid[:h,:w]
    old=np.stack([(xx-320)/500,(yy-180)/500,np.ones((h,w))],-1).astype(np.float32)
    labels=np.ones((h,w),np.int32);flow=np.zeros((2,h,w),np.float32);flow[0]=20
    means=np.array([[0.,0.,3.]],np.float32)
    predicted,_=predict_particles(means,np.ones(1,int),old,old,labels,labels,
        flow,np.ones((h,w),bool),np.eye(3),np.zeros(3),np.zeros_like(old),
        np.eye(3)[None]*.002)
    np.testing.assert_allclose(predicted,[[.12,0,3]],atol=1e-6)


def test_planar_transport_uses_the_current_camera_pose():
    from scipy.spatial.transform import Rotation
    from unified.geometry import Camera
    h,w=360,640;yy,xx=np.mgrid[:h,:w]
    old=np.stack([(xx-320)/500,(yy-180)/500,np.ones((h,w))],-1).astype(np.float32)
    labels=np.ones((h,w),np.int32);flow=np.zeros((2,h,w),np.float32);flow[0]=20
    means=np.array([[0.,0.,3.]],np.float32)
    rotation=Rotation.from_rotvec([0,.05,0]).as_matrix();translation=np.array([.03,0,0])
    predicted,_=predict_particles(means,np.ones(1,int),old,old,labels,labels,
        flow,np.ones((h,w),bool),np.eye(3),np.zeros(3),np.zeros_like(old),
        np.eye(3)[None]*.002,rotation,translation)
    uv,_=Camera(translation,rotation.T,500,640,360).project(predicted)
    np.testing.assert_allclose(uv,[[340,180]],atol=1e-4)


def test_bilinear_image_border_cannot_create_camera_origin_motion():
    h,w=360,640;old=np.zeros((h,w,3),np.float32);old[...,2]=2
    labels=np.ones((h,w),np.int32);flow=np.zeros((2,h,w),np.float32)
    flow[0]=.8
    means=np.array([[(638.5-320)*2/500,0,2]],np.float32)
    _,shift=predict_particles(means,np.ones(1,int),old,old,labels,labels,flow,
        np.ones((h,w),bool),np.eye(3),np.zeros(3),np.zeros_like(old))
    np.testing.assert_allclose(shift,0,atol=1e-6)
