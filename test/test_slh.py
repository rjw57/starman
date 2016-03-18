import numpy as np
from numpy.random import multivariate_normal as sample_mvn

from starman import slh_associate, MultivariateNormal

def new_feature():
    # Sample location
    mean = 10*np.random.rand(2)

    # Sample measurement error
    cov = np.diag(1e-1 + 3e-1*np.random.rand(2))**2

    return MultivariateNormal(mean=mean, cov=cov)

def peturb_feature(f):
    # Sample measurement error
    meas_cov = np.diag(1e-1 + 1e-1*np.random.rand(2))
    return MultivariateNormal(
        mean=f.mean + sample_mvn(mean=np.zeros_like(f.mean), cov=meas_cov + f.cov),
        cov=meas_cov
    )

def test_slh():
    # Set seed for consistent results
    np.random.seed(0xdeadbeef)

    N = 20

    # Generate A features
    a_features = [new_feature() for _ in range(N)]

    ground_truth = set()

    # Generate B features
    b_features = []
    for a_idx, a in enumerate(a_features):
        if np.random.rand() < 0.8:
            ground_truth.add((a_idx, len(b_features)))
            b_features.append(peturb_feature(a))

    # Add some spurious features to be
    b_features.extend([new_feature() for _ in range(1+int(N/10))])

    # Compute association
    associations = slh_associate(a_features, b_features)
    assert associations.shape[1] == 2

    # Tot up stats
    false_positives, true_positives = 0, 0
    for a_idx, b_idx in associations:
        if (a_idx, b_idx) in ground_truth:
            true_positives += 1
        else:
            false_positives += 1
    n_missed = len(ground_truth) - true_positives

    print("A feature count: {}, B feature count: {}".format(
        len(a_features), len(b_features)))
    print("Number of ground truth associations: {}".format(len(ground_truth)))
    print("True associations: {}".format(true_positives))
    print("False associations: {}".format(false_positives))
    print("Associations missed: {}".format(n_missed))

    assert true_positives > 0.75 * len(ground_truth)
    assert false_positives < 0.25 * len(ground_truth)
    assert n_missed < 0.25 * len(ground_truth)

def test_slh_no_features():
    assert slh_associate([], []).shape == (0, 2)

def test_slh_no_features_in_one_frame():
    assert slh_associate([new_feature()], []).shape == (0, 2)
    assert slh_associate([], [new_feature()]).shape == (0, 2)
