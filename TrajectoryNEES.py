#!/usr/bin/env python
# Software License Agreement (GNU GPLv3  License)
#
# Copyright (c) 2020, Roland Jung (roland.jung@aau.at) , AAU, KPK, NAV
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Requirements:
# sudo pip install numpy trajectory
########################################################################################################################

import numpy as np
import transformations as tf
from trajectory.Trajectory import Trajectory
from trajectory.TrajectoryEstimated import TrajectoryEstimated
from numpy_utils.numpy_statistics import numpy_statistics


class TrajectoryNEES:
    NEES_p_vec = None
    NEES_q_vec = None
    ANEES_p = None
    ANEES_q = None

    def __init__(self, traj_est, traj_err):
        assert (isinstance(traj_est, TrajectoryEstimated))
        assert (isinstance(traj_err, Trajectory))

        self.NEES_p_vec = TrajectoryNEES.toNEES_arr(False, traj_est.Sigma_p_vec, traj_err.p_vec)
        self.NEES_q_vec = TrajectoryNEES.toNEES_arr(True, traj_est.Sigma_q_vec, traj_err.q_vec)

        self.ANEES_p = np.mean(self.NEES_p_vec)
        self.ANEES_q = np.mean(self.NEES_q_vec)

    # https://de.mathworks.com/help/fusion/ref/trackerrormetrics-system-object.html
    @staticmethod
    def toNEES_arr(is_angle, P_arr, err_arr):
        # if is_angle:
        #    err_arr = tf.euler_from_quaternion(err_arr, 'rzyx')

        l = err_arr.shape[0]
        nees_arr = np.zeros((l, 1))
        for i in range(0, l):
            nees_arr[i] = TrajectoryNEES.toNEES(is_angle=is_angle, P=P_arr[i], err=err_arr[i])
        return nees_arr

    @staticmethod
    def toNEES(is_angle, P, err):
        if is_angle:
            err = np.array(tf.euler_from_quaternion(err, 'rzyx'))

        P_inv = np.linalg.inv(P)

        nees = np.matmul(np.matmul(err.reshape(1, 3), P_inv), err.reshape(3, 1))

        return nees


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time
from AbsoluteTrajectoryError import AbsoluteTrajectoryError
from trajectory.TrajectoryPlotter import TrajectoryPlotter, TrajectoryPlotConfig


class TrajectoryNEES_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]"

    def get_trajectories(self):
        traj_est = TrajectoryEstimated()
        self.assertTrue(traj_est.load_from_CSV('../sample_data/ID1-pose-est-cov.csv'))
        traj_gt = Trajectory()
        self.assertTrue(traj_gt.load_from_CSV('../sample_data/ID1-pose-gt.csv'))
        return traj_est, traj_gt

    def test_nees(self):
        self.start()
        traj_est, traj_gt = self.get_trajectories()
        ATE = AbsoluteTrajectoryError(traj_est, traj_gt)
        self.stop('Loading + ATE')
        self.start()
        NEES = TrajectoryNEES(ATE.traj_est, ATE.traj_err)
        self.stop('NEES computation')
        print('ANEES_p: ' + str(NEES.ANEES_p))
        print('ANEES_q: ' + str(NEES.ANEES_q))


if __name__ == "__main__":
    unittest.main()
