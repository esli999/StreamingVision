"""Small seed identities must not disappear between or within sampled grid cells."""
import numpy as np
from unified.tracker import seed_coordinates,sample_indices


def test_tiny_instances_use_actual_pixels_and_do_not_displace_other_objects():
    labels=np.zeros((40,64),np.int32)
    labels[1,1]=1;labels[1,2]=2;labels[10:25,30:50]=3
    yy,xx=seed_coordinates(labels,4)
    assert set(np.unique(labels[yy,xx]))=={0,1,2,3}
    for label,point in [(1,(1,1)),(2,(1,2))]:
        slot=np.where(labels[yy,xx]==label)
        assert (yy[slot][0],xx[slot][0])==point


def test_regular_seed_coordinates_are_unchanged_when_every_id_is_present():
    labels=np.zeros((40,64),np.int32);labels[10:25,30:50]=1
    yy,xx=seed_coordinates(labels,4)
    np.testing.assert_array_equal(labels[yy,xx],labels[2::4,2::4])
    np.testing.assert_array_equal(yy[:,0],np.arange(2,40,4))
    np.testing.assert_array_equal(xx[0],np.arange(2,64,4))


def test_foreground_quota_preserves_a_single_pixel_instance_among_large_objects():
    labels=np.ones((90,160),np.int32);labels[:20]=0;labels[20,0]=2
    indices=sample_indices(labels,number=100,seed=42)
    assert len(indices)==len(np.unique(indices))==100
    assert set(labels.reshape(-1)[indices])=={0,1,2}
