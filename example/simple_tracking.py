#!/usr/bin/env python3
"""
Usage:
    simple_tracking.py [--seed=INTEGER] [<output>]

Options:
    --seed=INTEGER  Seed for random number generator. [default: 1234]

"""
import collections
import itertools
import os

from matplotlib.pyplot import *
import numpy as np
import docopt

import starman
from starman.linearsystem import generate_states, measure_states

# Constant velocity model:
PROCESS_MAT = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

MEASUREMENT_MAT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])

PROCESS_COV = np.diag([1e-1, 1e-1, 1e-2, 1e-2]) ** 2
MEASUREMENT_COV = np.diag([5e-1, 5e-1]) ** 2

NEW_TARGET_STATE_COV = np.diag([1e1, 1e1, 1e-2, 1e-2]) ** 2

FRAME_COUNT=100
TARGET_DEATH_PROB=0.01
MEAN_NEW_TARGETS_PER_FRAME=0.5

GroundTruthTarget = collections.namedtuple(
    'GroundTruthTarget', 'birth_time states'
)

Target = collections.namedtuple(
    'Target', 'birth_time filter'
)

def create_new_target(birth_time, measurement):
    new_target = Target(
        birth_time=birth_time,
        filter=starman.KalmanFilter(
            process_matrix=PROCESS_MAT,
            process_covariance=PROCESS_COV,
            state_length=4,
        )
    )
    new_target.filter.predict()
    new_target.filter.update(measurement, MEASUREMENT_MAT)
    return new_target

def main():
    opts = docopt.docopt(__doc__)

    np.random.seed(int(opts['--seed']))

    gt = generate_gt_targets()
    meas = generate_measurements(gt)

    live, dead = set(), set()
    m_by_frame = dict((int(k), np.atleast_2d(list(v))[:, 1:])
                      for k, v in itertools.groupby(meas, key=lambda m: m[0]))
    for frame_idx in range(FRAME_COUNT):
        m = m_by_frame.get(frame_idx, np.zeros((0, 2)))
        frame_meas = [
            starman.MultivariateNormal(mean=row, cov=MEASUREMENT_COV)
            for row in m
        ]

        live_list = list(live)

        measure_ests = []
        for t in live_list:
            t.filter.predict()
            est = t.filter.prior_state_estimates[-1]

            m_mean = MEASUREMENT_MAT.dot(est.mean)
            m_cov = MEASUREMENT_MAT.dot(est.cov).dot(MEASUREMENT_MAT.T)
            measure_ests.append(starman.MultivariateNormal(
                mean=m_mean, cov=m_cov))

        assocs = starman.slh_associate(measure_ests, frame_meas)

        new_live = set()
        meas_w_no_t = set(range(len(frame_meas)))
        for t_idx, m_idx in assocs:
            t = live_list[t_idx]
            m = frame_meas[m_idx]
            t.filter.update(m, MEASUREMENT_MAT)
            new_live.add(t)
            meas_w_no_t.remove(m_idx)

        for m_idx in meas_w_no_t:
            new_target = create_new_target(frame_idx, frame_meas[m_idx])
            new_live.add(new_target)

        dead |= live - new_live
        live = new_live

    targets = set((
        t for t in dead | live
        if t.filter.measurement_count > 4
    ))

    print('Number of ground truth targets: {}'.format(len(gt)))
    print('Tracked {} targets'.format(len(targets)))

    f1 = figure()
    plot_axes = new_axes()
    plot_measurements(plot_axes, meas, marker='x', ls='none', c='k', alpha=0.2)

    for t in targets:
        ests = starman.rts_smooth(t.filter, t.filter.measurement_count)
        ests = t.filter.posterior_state_estimates[:t.filter.measurement_count]
        means = np.vstack([e.mean for e in ests])
        plot_states(plot_axes, means, t.birth_time)

    f2 = figure()
    subplot(1, 2, 1)
    for t in gt:
        plot(t.states[:, 0], t.states[:, 1])
    axis('equal')
    subplot(1, 2, 2)
    for t in targets:
        ests = starman.rts_smooth(t.filter, t.filter.measurement_count)
        ests = t.filter.posterior_state_estimates[:t.filter.measurement_count]
        means = np.vstack([e.mean for e in ests])
        plot(means[:, 0], means[:, 1])
    axis('equal')

    if opts['<output>'] is not None:
        tight_layout()
        base, ext = os.path.splitext(opts['<output>'])
        f1.savefig(base + '_1' + ext)
        f2.savefig(base + '_2' + ext)
    else:
        show()

def generate_gt_targets():
    targets = []

    for birth_t in range(FRAME_COUNT):
        # How many new targets?
        new_count = np.random.poisson(MEAN_NEW_TARGETS_PER_FRAME)

        for _ in range(new_count):
            # How many states for the new target?
            n_states = 1
            while n_states + birth_t <= FRAME_COUNT and \
                    np.random.rand() > TARGET_DEATH_PROB:
                n_states += 1

            initial_state = np.random.multivariate_normal(
                mean=np.zeros(4), cov=NEW_TARGET_STATE_COV
            )
            states = generate_states(n_states, PROCESS_MAT, PROCESS_COV,
                                     initial_state)

            targets.append(GroundTruthTarget(birth_time=birth_t, states=states))

    return targets

def generate_measurements(gt_targets):
    measurements = []

    for t in gt_targets:
        m = measure_states(t.states, MEASUREMENT_MAT, MEASUREMENT_COV)
        ks = np.arange(m.shape[0]) + t.birth_time
        measurements.append(
            np.hstack((ks.reshape((-1, 1)), m))
        )

    measurements = np.vstack(measurements)
    measurements = measurements[np.argsort(measurements[:, 0]), :]

    return measurements

def new_axes():
    ax_x = subplot(2, 2, 1)
    ax_x.set_ylabel('X')
    ax_y = subplot(2, 2, 2, sharey=ax_x)
    ax_y.set_ylabel('Y')
    ax_vx = subplot(2, 2, 3, sharex=ax_x)
    ax_vx.set_ylabel('velocity X')
    ax_vy = subplot(2, 2, 4, sharex=ax_y, sharey=ax_vx)
    ax_vy.set_ylabel('velocity Y')

    ax_vx.set_xlabel('Frame')
    ax_vy.set_xlabel('Frame')

    return ax_x, ax_y, ax_vx, ax_vy

def plot_states(plot_axes, states, birth_time=0, **kwrgs):
    ks = np.arange(states.shape[0]) + birth_time
    ax_x, ax_y, ax_vx, ax_vy = plot_axes
    ax_x.plot(ks, states[:, 0], **kwrgs)
    ax_y.plot(ks, states[:, 1], **kwrgs)
    ax_vx.plot(ks, states[:, 2], **kwrgs)
    ax_vy.plot(ks, states[:, 3], **kwrgs)

def plot_measurements(plot_axes, meas, **kwrgs):
    ax_x, ax_y, ax_vx, ax_vy = plot_axes
    ax_x.plot(meas[:, 0], meas[:, 1], **kwrgs)
    ax_y.plot(meas[:, 0], meas[:, 2], **kwrgs)

if __name__ == '__main__':
    main()
