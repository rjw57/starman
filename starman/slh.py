"""
Scott and Longuet-Higgins algorithm for association of features.

"""
import numpy as np

def slh_associate(a_features, b_features, max_sigma=5):
    """
    An implementation of the Scott and Longuet-Higgins algorithm [SLH]_ for
    feature association.

    This function takes two lists of features. Each feature is a
    :py:class:`MultivariateNormal` instance representing a feature
    location and its associated uncertainty.

    Args:
        a_features (list of MultivariateNormal)
        b_features (list of MultivariateNormal)
        max_sigma (float or int): maximum number of standard deviations two
            features can be separated and still considered "associated".

    Returns:
        (array): A Nx2 array of feature associations. Column 0 is the index into
        the a_features list, column 1 is the index into the b_features list.

    .. [SLH] Scott, Guy L., and H. Christopher Longuet-Higgins. "An algorithm
       for associating the features of two images." Proceedings of the Royal
       Society of London B: Biological Sciences 244.1309 (1991): 21-26.

    """
    # Compute proximity matrix
    proximity = _weighted_proximity(a_features, b_features)

    # Compute association matrix
    association_matrix = _proximity_to_association(proximity)

    # Now build up list of associations
    associations = []

    if association_matrix.shape[0] == 0:
        return np.zeros((0, 2))

    # Compute column-wise maxima
    col_max_idxs = np.argmax(association_matrix, axis=0)

    prox_threshold = np.exp(-0.5*max_sigma*max_sigma)

    # Find associations
    for row_idx, row in enumerate(association_matrix):
        if row.shape[0] == 0:
            continue

        # ... compute index of maximum element
        col_idx = np.argmax(row)

        # Is this row also the maximum in that column?
        if col_max_idxs[col_idx] == row_idx:
            prox = proximity[row_idx, col_idx]
            if prox > prox_threshold:
                associations.append((row_idx, col_idx))

    if len(associations) == 0:
        return np.zeros((0, 2))

    return np.vstack(associations)

def _weighted_proximity(a_features, b_features):
    # Form Gaussian weighted proximity matrix
    proximity = np.zeros((len(a_features), len(b_features)))
    for a_idx, a_feat in enumerate(a_features):
        for b_idx, b_feat in enumerate(b_features):
            cov = a_feat.cov + b_feat.cov
            delta = a_feat.mean - b_feat.mean
            proximity[a_idx, b_idx] = np.exp(
                -0.5 * delta.dot(np.linalg.inv(cov)).dot(delta)
            )
    return proximity

def _proximity_to_association(proximity):
    """SLH algorithm for increasing orthogonality of a matrix."""
    # pylint:disable=invalid-name
    # I'm afraid that the short names here are just a function of the
    # mathematical nature of the code.

    # Special case: zero-size matrix
    if proximity.shape[0] == 0 or proximity.shape[1] == 0:
        return proximity

    # Take SVD of proximity matrix
    U, sing_vals, Vh = np.linalg.svd(proximity)

    # Count the number of non-zero singular values
    nsv = np.count_nonzero(sing_vals)

    # Trim U and Vh to match
    U, Vh = U[:, :nsv], Vh[:nsv, :]

    # Compute new matrix as if singular values were unity
    return U.dot(Vh)
