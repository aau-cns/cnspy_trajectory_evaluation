#!/usr/bin/env python
# Software License Agreement (GNU GPLv3  License)
#
# Copyright (c) 2022, Roland Jung (roland.jung@aau.at) , AAU, KPK, NAV
#
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
########################################################################################################################
import os
from sys import version_info

import numpy as np

from cnspy_spatial_csv_formats.EstimationErrorType import EstimationErrorType
from cnspy_trajectory.SpatialConverter import SpatialConverter
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryError import TrajectoryError
from cnspy_trajectory.TrajectoryErrorType import TrajectoryErrorType


class DifferentialTrajectoryError(TrajectoryError):
    def __init__(self, fn=None, df=None, traj_est=None, traj_gt=None):
        traj_err_type=TrajectoryErrorType(err_type=EstimationErrorType.type1)

        if df is not None:
            TrajectoryError.__init__(self, df=df, traj_err_type=traj_err_type)
        elif fn is not None:
            TrajectoryError.__init__(self, fn=fn, traj_err_type=traj_err_type)
        elif traj_gt is not None and traj_gt is not None:
            p_err_vec, q_err_vec, t_vec = DifferentialTrajectoryError.compute_differential_error(traj_gt=traj_gt,
                                                                                                 traj_est=traj_est)
            TrajectoryError.__init__(self, p_vec=p_err_vec, q_vec=q_err_vec, t_vec=t_vec, traj_err_type=traj_err_type)
        pass

    @staticmethod
    def compute_differential_error(traj_gt, traj_est):
        """
        Computes the difference between pose increments (between consecutive timestamps) of the true and estimated
        trajectory, assuming same bases between increments. Thus, it show growth in drift from the estimated to the true
        trajectory. No trajectory alignment is needed, but the trajectories need to be associated (have the same
        timesteps)

        Parameters
        ----------
        traj_gt
        traj_est

        Returns
        -------
        p_err_vec
        q_err_vec
        t_err_vec
        """
        assert isinstance(traj_gt, Trajectory) and isinstance(traj_est, Trajectory)
        assert traj_est.num_elems() == traj_gt.num_elems(), 'traj_gt and traj_est should be first associated by timestamps'
        assert np.sum(traj_gt.t_vec - traj_est.t_vec) < 0.1, 'traj_gt and traj_est should be first associated by timestamps'

        p_err_vec = np.zeros([traj_gt.num_elems(), 3])
        q_err_vec = np.zeros([traj_gt.num_elems(), 4])
        t_err_vec = traj_gt.t_vec

        T_G_B_at_0 = None
        T_G_Best_at_0 = None
        for i in range(traj_est.num_elems()):
            if i > 0:
                T_G_B_at_i = SpatialConverter.p_q_HTMQ_to_SE3(traj_gt.p_vec[i, :], traj_gt.q_vec[i, :])
                T_G_Best_at_i = SpatialConverter.p_q_HTMQ_to_SE3(traj_est.p_vec[i, :], traj_est.q_vec[i, :])

                T_B0_Bi = T_G_B_at_0.inv()*T_G_B_at_i
                T_B0_Biest = T_G_Best_at_0.inv()*T_G_Best_at_i

                # Local pose error definition: type1
                # T_A_B_err = T_A_B_est.inv() T_A_B;
                # T_A_B = T_A_B_est * T_A_B_err

                # Note: this error assumes that B0 in gt and est are aligned/equal!
                # It computes the pose the true Bi and Bi_est as error measure
                T_Bi_Biest = T_B0_Bi.inv()*T_B0_Biest
                p_err_vec[i,:], q_err_vec[i,:] = SpatialConverter.SE3_to_p_q_HTMQ(T_Bi_Biest)

                T_G_B_at_0 = T_G_B_at_i
                T_G_Best_at_0 = T_G_Best_at_i
            else:
                T_G_B_at_0 = SpatialConverter.p_q_HTMQ_to_SE3(traj_gt.p_vec[i, :], traj_gt.q_vec[i, :])
                T_G_Best_at_0 = SpatialConverter.p_q_HTMQ_to_SE3(traj_est.p_vec[i, :], traj_est.q_vec[i, :])
                q_err_vec[i, :] = SpatialConverter.HTMQ_quaternion_identity()

        return p_err_vec, q_err_vec, t_err_vec