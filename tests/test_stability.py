"""Particle display jitter must decrease without freezing real image motion."""
import numpy as np
from unified.stability import stabilize, project, advect


def sequence(n=40, speed=0., noise=0.):
    ids=np.array([0,0,0,1,1,1])
    uv=np.tile(np.array([[100,130],[130,130],[160,130],[300,220],[330,220],[360,220]])[None],(n,1,1)).astype(float)
    uv[:,:,0]+=np.arange(n)[:,None]*speed
    clean=uv.copy()
    rng=np.random.default_rng(12)
    uv[1:]+=rng.normal(0,noise,uv[1:].shape)
    z=np.full((n,len(ids)),3.)
    means=np.concatenate([(uv-[320,180])*z[...,None]/500,z[...,None]],axis=-1)
    covs=np.tile(np.diag([.02,.01,.005]),(n,len(ids),1,1))
    support=np.full((n,len(ids)),4.)
    frame=dict(backward_grid=np.broadcast_to(np.array([-speed/8,0])[:,None,None],(2,45,80)).copy(),
               reliable_grid=np.ones((45,80),bool))
    return means,covs,support,ids,lambda i:frame,clean


def test_flow_transport_does_not_lag_fast_true_motion():
    m,c,s,ids,reader,clean=sequence(speed=5.)
    output,shapes,valid=stabilize(m,c,s,ids,reader,30.)
    np.testing.assert_allclose(project(output),clean,atol=1e-5)
    np.testing.assert_allclose(shapes,c,atol=1e-12)
    assert valid[1:].all()


def test_noisy_centers_and_shapes_stabilize_while_the_object_keeps_moving():
    m,c,s,ids,reader,clean=sequence(speed=3.,noise=4.)
    c[1::2]*=1.5;c[2::2]*=.7
    output,shapes,_=stabilize(m,c,s,ids,reader,30.)
    raw_error=np.linalg.norm(project(m)-clean,axis=-1).mean()
    display_error=np.linalg.norm(project(output)-clean,axis=-1).mean()
    assert display_error<raw_error*.5
    assert np.linalg.norm(np.diff(shapes,axis=0),axis=(-1,-2)).mean()<np.linalg.norm(np.diff(c,axis=0),axis=(-1,-2)).mean()*.3
    assert np.linalg.eigvalsh(shapes).min()>0
    np.testing.assert_allclose((project(output)[-1]-project(output)[0])[:,0],39*3.,atol=4.)


def test_filter_is_causal_and_does_not_mutate_captured_geometry():
    m,c,s,ids,reader,_=sequence(noise=3.)
    before=m.copy();before_cov=c.copy()
    full=stabilize(m,c,s,ids,reader,30.)
    short=stabilize(m[:20],c[:20],s[:20],ids,reader,30.)
    for a,b in zip(full,short): np.testing.assert_array_equal(a[:20],b)
    np.testing.assert_array_equal(m,before)
    np.testing.assert_array_equal(c,before_cov)


def test_flow_from_an_occluder_or_unsupported_region_is_rejected():
    uv=np.array([[100.,100.],[200.,200.],[300.,200.]])
    frame=dict(backward_grid=np.ones((2,45,80)),reliable_grid=np.ones((45,80),bool),
               labels=np.zeros((360,640),int),support=np.array([0.,4.,4.]))
    _,valid=advect(uv,frame,np.array([0,1,0]))
    np.testing.assert_array_equal(valid,[False,False,True])


def test_shared_depth_jitter_is_damped_without_stopping_sustained_depth_motion():
    m,c,s,ids,reader,clean=sequence(n=80)
    depth=3*np.exp(np.arange(80)*.003)
    depth[10::2]*=1.025;depth[11::2]/=1.025
    m[:,:,2]=depth[:,None]
    m[:,:,:2]=(clean-[320,180])*depth[:,None,None]/500
    output,_,_=stabilize(m,c,s,ids,reader,30.)
    raw_acc=np.abs(np.diff(m[:,:,2],n=2,axis=0))[15:].mean()
    smooth_acc=np.abs(np.diff(output[:,:,2],n=2,axis=0))[15:].mean()
    assert smooth_acc<raw_acc*.3
    # Smooth image flow and relative-depth trend both remain, rather than a
    # frozen 3D object with a merely cleaner projected position.
    np.testing.assert_allclose(project(output),clean,atol=1e-5)
    np.testing.assert_allclose(output[-10:,:,2].mean(1),3*np.exp(np.arange(70,80)*.003),rtol=.01)
