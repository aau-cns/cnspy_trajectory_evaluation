#!/usr/bin/env python
# Software License Agreement (GNU GPLv3  License)
#
# Copyright (c) 2020, Roland Jung (roland.jung@aau.at) , AAU, KPK, NAV
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
# references:
#
# [1] "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry", Zichao Zhang, Davide Scaramuzza, 2018
# [2] "Gaussian Process Kernels for Rotations and 6D Rigid Body Motions", Muriel Lang, Oliver Dunkley and Sandra Hirche
# [3] "Metrics for 3D Rotations: Comparison and Analysis", Du Q. Huynh, 2009
# adapted from:
# * https://github.com/uzh-rpg/rpg_trajectory_evaluation/blob/master/src/rpg_trajectory_evaluation/compute_trajectory_errors.py
#
# Requirements:
# cnspy_numpy_utils, cnspy_trajectory
########################################################################################################################

from cnspy_numpy_utils.accumulated_distance import *
from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryError import TrajectoryError
from cnspy_trajectory.SpatialConverter import SpatialConverter
from cnspy_trajectory.TrajectoryErrorType import TrajectoryErrorType

# - TODO: ARMSE_p should result in the same error using different ATE types!
class AbsoluteTrajectoryError:
    traj_err = None  # TrajectoryError
    traj_est = None  # Trajectory/TrajectoryEstimated
    traj_gt = None   # Trajectory

    def __init__(self, traj_est, traj_gt, traj_err_type=TrajectoryErrorType()):
        self.traj_est = traj_est
        self.traj_gt = traj_gt
        self.traj_err = AbsoluteTrajectoryError.compute_trajectory_error(traj_est=traj_est, traj_gt=traj_gt,
                                                                         traj_err_type=traj_err_type)
        pass

    def plot_pose_err(self, fig=None, cfg=None, angles=False, plot_rpy=False):
        TrajectoryError.plot_pose_err(traj_est=self.traj_est, traj_err=self.traj_err, traj_gt=self.traj_gt,
                                      cfg=cfg, angles=angles, plot_rpy=plot_rpy)
    @staticmethod
    def compute_scale(p_gt_vec, p_est_vec):
        # scale drift
        dist_gt = total_distance(p_gt_vec)
        dist_es = total_distance(p_est_vec)
        e_scale = 1.0

        if dist_gt > 0:
            e_scale = abs((dist_gt / dist_es))
        return e_scale

    @staticmethod
    def compute_trajectory_error(traj_est, traj_gt, traj_err_type=TrajectoryErrorType()):
        assert (isinstance(traj_est, Trajectory))
        assert (isinstance(traj_gt, Trajectory))
        assert (isinstance(traj_err_type, TrajectoryErrorType))

        assert traj_est.num_elems() == traj_gt.num_elems(), "Traj. have to be matched in time and aligned first"

        if traj_err_type.is_global_p_local_q():
            e_p_vec, e_q_vec, e_scale = \
                AbsoluteTrajectoryError.compute_absolute_global_p_local_q_error(p_est=traj_est.p_vec,
                                                                                q_est=traj_est.q_vec,
                                                                                p_gt=traj_gt.p_vec,
                                                                                q_gt=traj_gt.q_vec)
        elif traj_err_type.is_global_p_global_q():
            e_p_vec, e_q_vec, e_scale = \
                AbsoluteTrajectoryError.compute_absolute_global_p_local_q_error(p_est=traj_est.p_vec,
                                                                                q_est=traj_est.q_vec,
                                                                                p_gt=traj_gt.p_vec,
                                                                                q_gt=traj_gt.q_vec)
        elif traj_err_type.is_global_pose():
            e_p_vec, e_q_vec, e_scale = \
                AbsoluteTrajectoryError.compute_absolute_global_pose_error(p_est=traj_est.p_vec,
                                                                           q_est=traj_est.q_vec,
                                                                           p_gt=traj_gt.p_vec,
                                                                           q_gt=traj_gt.q_vec)
        elif traj_err_type.is_local_pose():
            e_p_vec, e_q_vec, e_scale = \
                AbsoluteTrajectoryError.compute_absolute_local_pose_error(p_est=traj_est.p_vec, q_est=traj_est.q_vec,
                                                                          p_gt=traj_gt.p_vec, q_gt=traj_gt.q_vec)
        else:
            assert False, "Error type not supported! " + str(traj_err_type)

        return TrajectoryError(t_vec=traj_gt.t_vec, p_vec=e_p_vec, q_vec=e_q_vec,
                                        scale=e_scale,
                                        traj_err_type=traj_err_type)

    @staticmethod
    def compute_absolute_global_p_local_q_error(p_est, q_est, p_gt, q_gt):
        """
        computes the absolute cnspy_trajectory error between a estimated and ground-truth cnspy_trajectory
        - The position error is global between the true body frame "B" and the estimated body frame "\hat{B}"
        - expressed in the global frame
        - The orientation error is local between the true body frame "B" and the estimated body frame "\hat{B}"
        > p_B_Best_in_G_err =  p_GBest_in_G_est - p_GB_in_B_gt
        > R_B_Best_err = R_GB_gt^{T} * R_G_Best_est
        """
        p_rows, p_cols = p_est.shape
        q_rows, q_cols = q_est.shape
        assert (p_est.shape == p_gt.shape)
        assert (q_est.shape == q_gt.shape)
        assert (p_rows == q_rows)
        assert (p_cols == 3)
        assert (q_cols == 4)

        e_p_vec = (p_est - p_gt)

        # orientation error
        e_q_vec = np.zeros(np.shape(q_est))  # [x, y, z, w]

        for i in range(np.shape(p_est)[0]):
            q_wb_est = q_est[i, :]  # [x,y,z,w] quaternion vector
            q_wb_gt = q_gt[i, :]  # [x,y,z,w] quaternion vector

            q_wb_est = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_est).unit()
            q_wb_gt = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_gt).unit()

            # Error definitions for local perturbation:
            # "Quaternion kinematics for the error-state Kalman filter" by Joan Solà, Chapter 4.4.1
            # First:  S = R oplus theta  = R * Exp(theta); with R = reference/gt, theta = perturbation, S = estimate
            # R_wb_gt =  R_wb_est  * R_wb_err
            q_wb_err = q_wb_est.conj() * q_wb_gt

            e_q_vec[i, :] = SpatialConverter.UnitQuaternion_to_HTMQ_quaternion(q_wb_err)


        # scale drift
        e_scale = AbsoluteTrajectoryError.compute_scale(p_gt, p_est)

        return e_p_vec, e_q_vec, e_scale

    @staticmethod
    def compute_absolute_global_p_global_q_error(p_est, q_est, p_gt, q_gt):
        """
        computes the absolute cnspy_trajectory error between a estimated and ground-truth cnspy_trajectory
        - The position error is global between the true body frame "B" and the estimated body frame "\hat{B}"
        - expressed in the global frame
        - The orientation error is local between the true body frame "B" and the estimated body frame "\hat{B}"
        > p_B_Best_in_G_err =  p_GBest_in_G_est - p_GB_in_B_gt
        > R_B_Best_err = R_G_Best_est^{T} * R_GB_gt
        """
        p_rows, p_cols = p_est.shape
        q_rows, q_cols = q_est.shape
        assert (p_est.shape == p_gt.shape)
        assert (q_est.shape == q_gt.shape)
        assert (p_rows == q_rows)
        assert (p_cols == 3)
        assert (q_cols == 4)

        e_p_vec = (p_est - p_gt)

        # orientation error
        e_q_vec = np.zeros(np.shape(q_est))  # [x, y, z, w]

        for i in range(np.shape(p_est)[0]):
            q_wb_est = q_est[i, :]  # [x,y,z,w] quaternion vector
            q_wb_gt = q_gt[i, :]  # [x,y,z,w] quaternion vector

            q_wb_est = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_est).unit()
            q_wb_gt = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_gt).unit()

            # R_wb_gt =  R_wb_err * R_wb_est
            q_wb_err =  q_wb_gt * q_wb_est.conj()

            e_q_vec[i, :] = SpatialConverter.UnitQuaternion_to_HTMQ_quaternion(q_wb_err)


        # scale drift
        e_scale = AbsoluteTrajectoryError.compute_scale(p_gt, p_est)

        return e_p_vec, e_q_vec, e_scale

    @staticmethod
    def compute_absolute_global_pose_error(p_est, q_est, p_gt, q_gt):
        """
        computes the absolute cnspy_trajectory error in the **global** frame, between a estimated and ground-truth cnspy_trajectory
        - according to "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry" - Zhang and Scaramuzza
        - The position error is global between the true world frame "G" and the estimated world frame "\hat{G}"
        - The orientation error global between the true world frame "G" and the estimated world frame "\hat{G}"
        - "W" and "G" are the same "Global/World" reference
        > p_G_Gest_in_G_err = p_GB_in_G_gt - R_G_Gest_err * p_Gest_B_in_Gest_est
        > R_G_Gest_err = R_G_B_gt * R_G_est_B_est^{T}
        """
        p_rows, p_cols = p_est.shape
        q_rows, q_cols = q_est.shape
        assert (p_est.shape == p_gt.shape)
        assert (q_est.shape == q_gt.shape)
        assert (p_rows == q_rows)
        assert (p_cols == 3)
        assert (q_cols == 4)

        # global absolute position error
        e_p_vec = np.zeros(np.shape(p_est))

        # global orientation error
        e_q_vec = np.zeros(np.shape(q_est))  # [x,y,z,w]

        for i in range(np.shape(p_est)[0]):
            T_est = SpatialConverter.p_q_HTMQ_to_SE3(p_est[i, :], q_est[i, :])
            T_gt = SpatialConverter.p_q_HTMQ_to_SE3(p_gt[i, :], q_gt[i, :])

            # T_wb_gt =  T_wb_err * T_wb_est
            T_err = T_gt*T_est.inv()

            p_err, q_err = SpatialConverter.SE3_to_p_q_HTMQ(T_err)
            e_p_vec[i, :] = p_err
            e_q_vec[i, :] = q_err

        # global absolute position RMSE

        # scale drift
        e_scale = AbsoluteTrajectoryError.compute_scale(p_gt, p_est)
        return e_p_vec, e_q_vec, e_scale

    @staticmethod
    def compute_absolute_local_pose_error(p_est, q_est, p_gt, q_gt):
        """
        computes the absolute cnspy_trajectory error in the **local/body** frame, between a estimated and ground-truth cnspy_trajectory
        - according to "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry" - Zhang and Scaramuzza
        - The position error is local between the estimated body frame "\hat{B}" and the true body frame "B"
        - The orientation error local between the estimated body frame "\hat{B}" and the true body frame "B"
        - "W" and "G" are the same "Global/World" reference
        > p_Best_B_in_Best_err = R_G_Best_est^{T} (p_GB_in_B - p_G_Best_in_G_est)
        > R_Best_B_err = R_G_Best_est^{T} * R_GB

        """
        p_rows, p_cols = p_est.shape
        q_rows, q_cols = q_est.shape
        assert (p_est.shape == p_gt.shape)
        assert (q_est.shape == q_gt.shape)
        assert (p_rows == q_rows)
        assert (p_cols == 3)
        assert (q_cols == 4)

        # global absolute position error
        e_p_vec = np.zeros(np.shape(p_est))

        # global orientation error
        e_q_vec = np.zeros(np.shape(q_est))  # [x,y,z,w]

        for i in range(np.shape(p_est)[0]):
            T_est = SpatialConverter.p_q_HTMQ_to_SE3(p_est[i, :], q_est[i, :])
            T_gt = SpatialConverter.p_q_HTMQ_to_SE3(p_gt[i, :], q_gt[i, :])

            # T_wb_gt =  T_wb_err * T_wb_est
            T_err = T_est.inv()*T_gt

            p_err, q_err = SpatialConverter.SE3_to_p_q_HTMQ(T_err)
            e_p_vec[i, :] = p_err
            e_q_vec[i, :] = q_err

        # scale drift
        e_scale = AbsoluteTrajectoryError.compute_scale(p_gt, p_est)
        return e_p_vec, e_q_vec, e_scale
