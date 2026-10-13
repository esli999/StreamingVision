"""Real GenMatter conditionals: optimization preserves the final sample and key."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jnp
import genmatter_rt as rt
from unified.seed import balanced_seed


def make_state():
    rng=np.random.default_rng(12)
    positions=rng.normal(size=(48,3)).astype(np.float32)*.2+[0,0,2]
    positions=positions.astype(np.float32)
    velocities=rng.normal(size=(48,3)).astype(np.float32)*.01
    features=rng.normal(size=(48,32)).astype(np.float32)
    labels=np.repeat([0,1],24)
    choice,_=balanced_seed(positions,velocities,labels,4)
    state,_=rt.init_state(positions,velocities,features,jax.random.key(4),seed_choice_map=choice,num_warmup_sweeps=0)
    return state


def test_skipping_unused_draws_preserves_actual_conditionals():
    state=make_state()
    # Exercise empty, low-count and fully supported blobs in the same test.
    assignments=jnp.asarray([0]*30+[1]*16+[2]*2)
    state=state.replace({'datapoints_state':{'blob_assignments':assignments}})
    for conditional,count in [(rt.gibbs_blob_vel_means,15),(rt.gibbs_blob_vel_covs,15),(rt.gibbs_blob_features_dino,3)]:
        def update(_,carry):
            key,st=carry
            key,draw_key=jax.random.split(key)
            return key,conditional(draw_key,st)
        baseline=jax.jit(lambda k,s:jax.lax.fori_loop(0,count,update,(k,s)))
        optimized=jax.jit(lambda k,s:rt._last_independent_update(k,s,update,count))
        ka,a=baseline(jax.random.key(7),state)
        kb,b=optimized(jax.random.key(7),state)
        np.testing.assert_array_equal(jax.random.key_data(ka),jax.random.key_data(kb))
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=1e-6,atol=1e-7)


def test_depth_outlier_cannot_stretch_a_particle_covariance():
    from unified.inference import refine_geometry
    state=make_state()
    x=np.zeros((48,3),np.float32)+[0,0,2]
    x[-1]=[5,5,7]
    labels=jnp.zeros(48,jnp.int32)
    state=state.replace({'datapoints_state':{'datapoint_positions':jnp.asarray(x,dtype=jnp.float32)}})
    means=jnp.tile(jnp.array([0.,0.,2.]),(4,1))
    covs=jnp.tile(jnp.eye(3)*.001,(4,1,1))
    result=refine_geometry(state,labels,means,covs,jnp.arange(4),covariance_anchor=covs)
    got=np.asarray(result.blobs_state.blob_covs)
    assert np.linalg.eigvalsh(got).max() < .0011
    assert np.linalg.norm(np.asarray(result.blobs_state.blob_means)[0]-[0,0,2]) < .01


def test_soft_support_prevents_map_winner_from_starving_other_particles():
    from unified.inference import refine_geometry,appearance_estimate
    state=make_state()
    x=jnp.tile(jnp.array([.02,0.,2.]),(48,1))
    state=state.replace({'datapoints_state':{'datapoint_positions':x}})
    means=jnp.tile(jnp.array([0.,0.,2.]),(4,1))
    covs=jnp.tile(jnp.eye(3)*.001,(4,1,1))
    result=refine_geometry(state,jnp.zeros(48,jnp.int32),means,covs,jnp.arange(4),
        covariance_anchor=covs,responsibilities=jnp.full((48,4),.25))
    assert np.all(np.asarray(result.blobs_state.blob_means)[:,0]>.01)
    np.testing.assert_allclose(result.blobs_state.blob_weights,np.full(4,.25),atol=1e-6)
    colors,rgb=appearance_estimate(jnp.array([[.75,.25],[.25,.75]]),
        jnp.array([[255.,0.,0.],[0.,0.,255.]]),jnp.zeros((2,3)),1.)
    np.testing.assert_allclose(colors,[[191.25,0,63.75],[63.75,0,191.25]])
    np.testing.assert_allclose(rgb,[[159.375,0,95.625],[95.625,0,159.375]])
