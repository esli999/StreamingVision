"""Balanced, instance-pure initialization of the actual GenMatter choice map."""
import numpy as np
from sklearn.cluster import KMeans


def covariance(points, floor):
    if len(points) < 2:
        return np.eye(3, dtype=np.float32)*floor
    c = np.cov(points.T).astype(np.float32)
    e, v = np.linalg.eigh(c)
    return ((v*np.maximum(e, floor))@v.T).astype(np.float32)


def balanced_seed(positions, velocities, labels, num_blobs, foreground_fraction=.65, seed=42):
    """Reserve particle capacity for each initial object, including small ones.

    Background uses one semantic hyperblob. Its spatial parts remain separate
    Gaussians; identity is not inferred from arbitrary RGB mask colors.
    """
    from genjax import ChoiceMapBuilder as C
    import jax.numpy as jnp
    ids, sizes = np.unique(labels, return_counts=True)
    if 0 not in ids:
        raise ValueError('Initial mask must include background')
    if num_blobs < len(ids) or num_blobs > len(labels):
        raise ValueError('Particle count must lie between object count and sample count')
    allocation = np.ones(len(ids), int)
    weights = np.sqrt(sizes.astype(float))
    weights[ids == 0] = weights[ids != 0].sum()*(1-foreground_fraction)/foreground_fraction
    for _ in range(num_blobs-len(ids)):
        score = weights/(allocation+1)
        score[allocation >= sizes] = -1
        allocation[score.argmax()] += 1
    means, covs, vm, vc, objects, counts = [], [], [], [], [], []
    assignments = np.empty(len(labels), np.int32)
    hb_means, hb_covs, hb_weights = [], [], []
    for h, (label, number) in enumerate(zip(ids, allocation)):
        select = np.flatnonzero(labels == label)
        p, v = positions[select], velocities[select]
        km = KMeans(n_clusters=int(number), n_init=1, random_state=seed).fit(p)
        assignments[select] = km.labels_ + len(means)
        hb_means.append(p.mean(0)); hb_covs.append(covariance(p, .01)); hb_weights.append(number)
        for b in range(number):
            take = km.labels_ == b
            means.append(p[take].mean(0)); covs.append(covariance(p[take], .0004))
            vm.append(v[take].mean(0)); vc.append(covariance(v[take], .002))
            objects.append(int(label)); counts.append(take.sum())
    arrays = {
        ('datapoints','blob_assignments'): assignments,
        ('datapoints','datapoint_positions'): positions,
        ('datapoints','datapoint_vels'): velocities,
        ('blobs','blob_means'): means, ('blobs','blob_covs'): covs,
        ('blobs','blob_vel_means'): vm, ('blobs','blob_vel_covs'): vc,
        ('blobs','blob_weights'): np.asarray(counts)/sum(counts),
        ('blobs','hyperblob_assignments'): np.repeat(np.arange(len(ids)), allocation),
        ('hyperblobs','hyperblob_means'): hb_means,
        ('hyperblobs','hyperblob_covs'): hb_covs,
        ('hyperblobs','hyperblob_weights'): np.asarray(hb_weights)/sum(hb_weights),
        ('hyperblobs','hyperblob_rot_vels'): np.tile(np.eye(3), (len(ids),1,1)),
        ('hyperblobs','hyperblob_trans_vels'): np.zeros((len(ids),3)),
    }
    result = None
    for address, value in arrays.items():
        value = np.asarray(value)
        value = value.astype(np.int32 if 'assignments' in address[1] else np.float32)
        choice = C[address].set(jnp.asarray(value))
        result = choice if result is None else result | choice
    return result, np.asarray(objects, np.int32)
