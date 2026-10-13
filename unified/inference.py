"""Object-conditioned GenMatter posterior and robust sufficient statistics.

The object memory is a likelihood term within blob assignment inference. A
render never substitutes a mask for these assignments. GenMatter's hierarchical
Gibbs sweep still updates motion/geometry; this conditional posterior removes
sampling noise from the displayed state and prevents semantic identity drift.
"""
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.cluster import KMeans


class AppearanceMemory:
    def __init__(self, features, labels, prototypes=12):
        self.centers, self.ids = [], []
        for label in np.unique(labels):
            x = features[labels == label]
            k = min(prototypes*(2 if label == 0 else 1), len(x))
            centers = KMeans(n_clusters=k, n_init=1, random_state=42).fit(x).cluster_centers_
            self.centers.extend(centers); self.ids.extend([label]*k)
        self.centers = np.asarray(self.centers, np.float32)
        self.ids = np.asarray(self.ids, np.int32)
        self.n_objects = int(labels.max())+1

    def logits(self, features):
        # Squared distances without an N × prototypes × channels temporary.
        d = (np.sum(features*features,axis=1)[:,None]
             + np.sum(self.centers*self.centers,axis=1)[None]
             - 2*features@self.centers.T)/features.shape[1]
        score = np.stack([-d[:,self.ids == k].min(1) for k in range(self.n_objects)], 1)
        return (score-score.max(1,keepdims=True)).astype(np.float32)


@jax.jit
def posterior(state, positions, features, object_evidence, previous_labels,
              reliable, blob_ids, memory_weight, persistence, feature_weight):
    """Conditional MAP blob labels, retaining a probability for every object."""
    bs = state.blobs_state
    cov = bs.blob_covs + jnp.eye(3)[None]*.0004
    inv = jnp.linalg.inv(cov)
    delta = positions[:,None]-bs.blob_means[None]
    geometry = -.5*jnp.einsum('nli,lij,nlj->nl', delta, inv, delta)
    geometry -= .5*jnp.linalg.slogdet(cov)[1][None]
    distance = (jnp.sum(features**2,1)[:,None] + jnp.sum(bs.blob_features**2,1)[None]
                - 2*features@bs.blob_features.T)
    appearance = -.5*distance/jnp.maximum(state.hypers.sigma_F, .05)
    logits = geometry + feature_weight*appearance + jnp.log(jnp.maximum(bs.blob_weights,1e-7))[None]
    logits += memory_weight*object_evidence[:,blob_ids]
    logits += persistence*(previous_labels[:,None] == blob_ids[None])*reliable[:,None]
    labels = jnp.argmax(logits,1).astype(jnp.int32)
    # Sum particle posteriors into stable object identities. No RGB-label EMA.
    probabilities = jax.nn.softmax(logits, axis=1)
    objects = jax.nn.one_hot(blob_ids, object_evidence.shape[1])
    return labels, probabilities@objects, probabilities


@jax.jit
def refine_geometry(state, labels, previous_means, previous_covs, blob_ids, delta=3.,
                    mean_prior_count=1.5,covariance_prior_count=4.,particle_weight_prior=3.,
                    covariance_anchor=None,covariance_growth_limit=9.,responsibilities=None):
    """Robust conditional means/covariances written back into the tracker.

    Uses posterior responsibilities, a weak motion prior and actual Huber
    residual weights. Soft sufficient statistics avoid starving all but a few
    winning particles. Bounded scatter and a frame-0 covariance prior prevent
    depth-boundary outliers from growing long ellipsoids. This updates inference
    state; it is not renderer-only smoothing.
    """
    ds, bs = state.datapoints_state, state.blobs_state
    L = bs.blob_means.shape[0]
    x, velocity = ds.datapoint_positions, ds.datapoint_vels
    residual = x[:,None]-previous_means[None]
    inv = jnp.linalg.inv(previous_covs + jnp.eye(3)[None]*.0004)
    distance = jnp.sqrt(jnp.maximum(jnp.einsum('nli,lij,nlj->nl',residual,inv,residual), 0)+1e-6)
    membership = jax.nn.one_hot(labels,L) if responsibilities is None else responsibilities
    w = membership*jnp.minimum(1., delta/distance)
    count = w.sum(0)
    prior = mean_prior_count
    means = (w.T@x + prior*previous_means)/(count[:,None]+prior)
    centered = x[:,None]-means[None]
    distance = jnp.sqrt(jnp.maximum(jnp.einsum('nli,lij,nlj->nl',centered,inv,centered),0)+1e-6)
    bounded = centered*jnp.minimum(1.,delta/distance)[...,None]
    cov = jnp.einsum('nl,nli,nlj->lij',w,bounded,bounded)
    anchor = previous_covs if covariance_anchor is None else covariance_anchor
    cov = (cov + covariance_prior_count*anchor)/(count[:,None,None]+covariance_prior_count)
    e, v = jnp.linalg.eigh(cov)
    ceiling = jnp.minimum(.15,jnp.linalg.eigvalsh(anchor)[:,-1]*covariance_growth_limit)
    cov = jnp.einsum('lij,lj,lkj->lik',v,jnp.maximum(.0004,jnp.minimum(e,ceiling[:,None])),v)
    vm = (w.T@velocity)/(count[:,None]+1)
    # Residual flow is observed in world coordinates; no free-running velocity.
    speed = jnp.linalg.norm(vm,axis=1,keepdims=True)
    vm *= jnp.minimum(1., .2/(speed+1e-8))
    # A symmetric Dirichlet prior avoids irreversible rich-get-richer collapse
    # when a small object briefly supplies only one or two samples per particle.
    weights = (count+particle_weight_prior)/(count.sum()+L*particle_weight_prior)
    return state.replace({'datapoints_state':{'blob_assignments':labels},
                          'blobs_state':{'blob_means':means,'blob_covs':cov,
                                         'blob_vel_means':vm,'blob_weights':weights}})


@jax.jit
def appearance_estimate(responsibilities,rgb,old_colors,blend):
    """Expected particle colors and RGB reconstruction under the same posterior."""
    mass = responsibilities.sum(0)
    observed = responsibilities.T@rgb/jnp.maximum(mass[:,None],1e-5)
    colors = jnp.where(mass[:,None]>.05,blend*observed+(1-blend)*old_colors,old_colors)
    return colors,responsibilities@colors
