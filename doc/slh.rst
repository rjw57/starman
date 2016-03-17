The Scott and Longuet-Higgins algorithm
=======================================

.. plot:: plotutils.py
    :context:
    :include-source: false

.. plot::
    :context:

    from matplotlib.pylab import *
    from numpy.random import multivariate_normal as sample_mvn
    from starman import slh_associate, MultivariateNormal

    np.random.seed(0xdeadbeef)


    new_feature = lambda: MultivariateNormal(
        mean=10 * np.random.rand(2),
        cov=diag(1e-1 + 2e-1 * np.random.rand(2))**2
    )

    measure_feature = lambda f, meas_cov: MultivariateNormal(
        mean=f.mean + sample_mvn(np.zeros_like(f.mean), cov=meas_cov + f.cov),
        cov=meas_cov
    )

    N = 20
    R = diag([1e-1, 1e-1])**2

    true_associations = set()
    a_features = [new_feature() for _ in range(N)]
    b_features = []
    for a_idx, a in enumerate(a_features):
        if np.random.rand() > 0.8:
            continue
        true_associations.add((a_idx, len(b_features)))
        b_features.append(measure_feature(a, R))

    # Add some noise
    b_features.extend([new_feature() for _ in range(1 + int(N/10))])

    figure()
    plot_feature_means(a_features, marker='x', c='b', ls='none', label="A")
    plot_feature_covariances(a_features, fill=False, ec='b')
    plot_feature_means(b_features, marker='x', c='g', ls='none', label="B")
    plot_feature_covariances(b_features, fill=False, ec='g')
    legend(loc='best')
    grid(True); axis('equal')

    figure()
    plot_feature_means(a_features, marker='x', c='b', ls='none', label="A")
    plot_feature_means(b_features, marker='x', c='g', ls='none', label="B")
    for a_idx, b_idx in true_associations:
        a, b = a_features[a_idx], b_features[b_idx]
        means = np.vstack((a.mean, b.mean))
        plot(means[:, 0], means[:, 1], 'k:', label="True")

    assocs = slh_associate(a_features, b_features)
    for a_idx, b_idx, _ in assocs:
        a_idx, b_idx = int(a_idx), int(b_idx)
        a, b = a_features[a_idx], b_features[b_idx]
        means = np.vstack((a.mean, b.mean))
        plot(means[:, 0], means[:, 1], 'g', label="SLH")
