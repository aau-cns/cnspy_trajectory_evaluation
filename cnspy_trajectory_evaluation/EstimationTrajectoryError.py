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
# Requirements:
# numpy, matplotlib
########################################################################################################################
import math
import os
import numpy as np
from cnspy_trajectory.SpatialConverter import SpatialConverter
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated
from cnspy_trajectory.TrajectoryEstimationError import TrajectoryEstimationError

from cnspy_trajectory_evaluation.AbsoluteTrajectoryError import AbsoluteTrajectoryError
from cnspy_spatial_csv_formats.EstimationErrorType import EstimationErrorType
from cnspy_spatial_csv_formats.ErrorRepresentationType import ErrorRepresentationType
from cnspy_trajectory.TrajectoryErrorType import TrajectoryErrorType


class EstimationTrajectoryError:
    traj_est_err = None  # TrajectoryEstimationError

    def __init__(self, traj_est, traj_gt):

        self.traj_est_err = EstimationTrajectoryError.compute_trajectory_estimation_error(traj_est,
                                                                                          traj_gt)

    @staticmethod
    def compute_trajectory_estimation_error(traj_est, traj_gt):

        assert (isinstance(traj_est, TrajectoryEstimated))
        assert (isinstance(traj_gt, Trajectory))

        est_err_type = traj_est.format.estimation_error_type  # EstimationErrorType.type1,
        err_rep_type = traj_est.format.rotation_error_representation  # ErrorRepresentationType.theta_R
        traj_err_type = TrajectoryErrorType(err_type=est_err_type)
        traj_err = AbsoluteTrajectoryError.compute_trajectory_error(traj_est=traj_est, traj_gt=traj_gt,
                                                                         traj_err_type=traj_err_type)

        # convert the traj_err into the error representation used
        theta_vec = SpatialConverter.convert_q_vec_to_theta_vec(traj_err.q_vec, rot_err_rep=err_rep_type)
        traj_est_err = TrajectoryEstimationError(t_vec=traj_err.t_vec, nu_vec=traj_err.p_vec, theta_vec=theta_vec,
                                                 est_err_type=est_err_type, err_rep_type=err_rep_type)
        return traj_est_err