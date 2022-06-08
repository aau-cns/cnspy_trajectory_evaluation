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
# refences:
#
# [1] "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry", Zichao Zhang, Davide Scaramuzza, 2018
# [2] "Gaussian Process Kernels for Rotations and 6D Rigid Body Motions", Muriel Lang, Oliver Dunkley and Sandra Hirche
# [3] "Metrics for 3D Rotations: Comparison and Analysis", Du Q. Huynh, 2009
# adapted from:
# * https://github.com/uzh-rpg/rpg_trajectory_evaluation/blob/master/src/rpg_trajectory_evaluation/compute_trajectory_errors.py
#
# Requirements:
# numpy, matplotlib
########################################################################################################################
import matplotlib.pyplot as plt

from cnspy_numpy_utils.accumulated_distance import *
from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory.TrajectoryPlotTypes import TrajectoryPlotTypes
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryPlotter import TrajectoryPlotter
from cnspy_trajectory.SpatialConverter import SpatialConverter
from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated
from cnspy_trajectory_evaluation.AbsoluteTrajectoryErrorType import AbsoluteTrajectoryErrorType

# TODOs
# - TODO: ARMSE_p should result in the same error using different ATE types!


class AbsoluteTrajectoryError:
    err_p_vec = None  # position error vector defined via "error_type" [m]
    err_q_vec = None  # rotation error defined via "error_type"  [quaternion]
    err_rpy_vec = None  # rotation error in rpy [rad] ``'rpy/zyx'``  roll-pitch-yaw angles in ZYX axis order:
    # (R=rotz(yaw) @ roty(pitch) @ rotx(roll)): spatialmath.transform2d.rpy2r()
    err_scale = None
    rmse_p_vec = None  # norm of position error vector [m]
    rmse_q_deg_vec = None  # rotation error angle [deg]
    t_vec = None

    traj_err = None
    traj_est = None
    traj_gt = None
    ARMSE_p = None      # [m]
    ARMSE_q_deg = None  # [deg]
    error_type = None

    def __init__(self, traj_est, traj_gt, err_type=AbsoluteTrajectoryErrorType.global_pose):
        assert (isinstance(traj_est, Trajectory))
        assert (isinstance(traj_gt, Trajectory))

        assert traj_est.num_elems() == traj_gt.num_elems(), "Traj. have to be matched in time and aligned first"

        self.error_type = err_type
        if err_type == AbsoluteTrajectoryErrorType.global_p_local_q:
            e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale = \
                AbsoluteTrajectoryError.compute_absolute_global_p_local_q_error(p_est=traj_est.p_vec,
                                                                                q_est=traj_est.q_vec,
                                                                                p_gt=traj_gt.p_vec,
                                                                                q_gt=traj_gt.q_vec)
        elif err_type == AbsoluteTrajectoryErrorType.global_pose:
            e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale = \
                AbsoluteTrajectoryError.compute_absolute_global_pose_error(p_est=traj_est.p_vec,
                                                                           q_est=traj_est.q_vec,
                                                                           p_gt=traj_gt.p_vec,
                                                                           q_gt=traj_gt.q_vec)
        elif err_type == AbsoluteTrajectoryErrorType.local_pose:
            e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale = \
                AbsoluteTrajectoryError.compute_absolute_local_pose_error(p_est=traj_est.p_vec,
                                                                          q_est=traj_est.q_vec,
                                                                          p_gt=traj_gt.p_vec,
                                                                          q_gt=traj_gt.q_vec)
        else:
            print("Error type not supported!")
            e_p_vec = []
            e_p_rmse_vec = []
            e_q_vec = []
            e_rpy_vec = []
            e_scale = []

        self.err_p_vec = e_p_vec
        self.err_q_vec = e_q_vec
        self.err_rpy_vec = e_rpy_vec
        self.err_scale = e_scale
        self.rmse_p_vec = e_p_rmse_vec
        tmp, self.rmse_q_deg_vec = AbsoluteTrajectoryError.compute_rmse_q(e_q_vec)
        self.t_vec = traj_est.t_vec - traj_est.t_vec[0]

        self.traj_err = Trajectory(t_vec=self.t_vec, p_vec=self.err_p_vec, q_vec=self.err_q_vec)
        self.traj_est = traj_est
        self.traj_gt = traj_gt
        self.ARMSE_p = np.mean(self.rmse_p_vec)
        self.ARMSE_q_deg = np.mean(self.rmse_q_deg_vec)

    def plot_pose_err(self, cfg=TrajectoryPlotConfig(), angles=False):
        # TODO: ugly, as error plot are defined here below, but are not used!

        [p_err_text, R_err_text] = self.error_type.error_def()

        return TrajectoryPlotter.plot_pose_err(TrajectoryPlotter(traj_obj=self.traj_est, config=cfg),
                                               TrajectoryPlotter(traj_obj=self.traj_err, config=cfg), cfg=cfg,
                                               angles=angles,
                                               plotter_gt=TrajectoryPlotter(traj_obj=self.traj_gt, config=cfg),
                                               local_p_err=self.error_type.is_local_p(),
                                               local_R_err=self.error_type.is_local_R())

    def plot_p_err(self, cfg=TrajectoryPlotConfig(), fig=None, ax=None):
        plotter = TrajectoryPlotter(traj_obj=self.traj_err, config=cfg)

        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
        if ax is None:
            ax = fig.add_subplot(111)

        plotter.ax_plot_pos(ax=ax, cfg=cfg)

        [p_err_text, R_err_text] = self.error_type.error_def()

        ax.set_ylabel(p_err_text + ' ARMSE ={:.2f} [m]'.format(self.ARMSE_p))

        TrajectoryPlotConfig.show_save_figure(cfg, fig)
        return fig, ax, plotter

    def plot_rpy_err(self, cfg=TrajectoryPlotConfig(), fig=None, ax=None):
        plotter = TrajectoryPlotter(traj_obj=self.traj_err, config=cfg)

        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
            ax = fig.add_subplot(111)
        plotter.ax_plot_rpy(ax=ax, cfg=cfg)

        [p_err_text, R_err_text] = self.error_type.error_def()

        if cfg.radians:
            ax.set_ylabel(R_err_text + ' ARMSE ={:.2f} [rad]'.format(np.deg2rad(self.ARMSE_q_deg)))
        else:
            ax.set_ylabel(R_err_text + ' ARMSE ={:.2f} [deg]'.format(self.ARMSE_q_deg))

            TrajectoryPlotConfig.show_save_figure(cfg, fig)
        return fig, ax, plotter

    @staticmethod
    def compute_rmse_q(q_err_arr):
        length = np.shape(q_err_arr)[0]
        rmse_rad_vec = np.zeros((length, 1))
        for i in range(length):
            R_SO3 = SpatialConverter.HTMQ_quaternion_to_SO3(q_err_arr[i, :])
            # Eq. 24 in Zhang and Scaramuzza [1]
            # Bi-invariate metric for rotations (length of geodesic on unit-sphere from identity element) [3]
            [theta, v] = R_SO3.angvec()
            rmse_rad_vec[i, :] = abs(theta)

        rmse_deg_vec = np.rad2deg(rmse_rad_vec)
        return rmse_rad_vec, rmse_deg_vec

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
        e_p_rmse_vec = np.sqrt(np.sum(e_p_vec ** 2, 1))

        # orientation error
        e_q_rmse_deg_vec = np.zeros(np.shape(p_est))
        e_rpy_vec = np.zeros(np.shape(p_est))
        e_q_vec = np.zeros(np.shape(q_est))  # [x, y, z, w]

        for i in range(np.shape(p_est)[0]):
            q_wb_est = q_est[i, :]  # [x,y,z,w] quaternion vector
            q_wb_gt = q_gt[i, :]  # [x,y,z,w] quaternion vector

            q_wb_est = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_est).unit()
            q_wb_gt = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_gt).unit()

            # Error definitions for local perturbation:
            # "Quaternion kinematics for the error-state Kalman filter" by Joan Solà, Chapter 4.4.1
            # First:  S = R oplus theta  = R * Exp(theta); with R = reference/gt, theta = perturbation, S = estimate
            # R_wb_est =  R_wb_gt  * R_wb_err
            # R_wb_gt = R_wb_est * R_wb_err'
            # R_wb_err = R_wb_gt' * R_wb_est
            q_wb_err = q_wb_gt.conj() * q_wb_est

            e_q_vec[i, :] = SpatialConverter.UnitQuaternion_to_HTMQ_quaternion(q_wb_err)
            e_rpy_vec[i, :] = q_wb_err.rpy(order='xyz')
            e_q_rmse_deg_vec[i] = np.rad2deg(np.linalg.norm(e_rpy_vec[i, :]))

        # scale drift
        dist_gt = total_distance(p_gt)
        dist_es = total_distance(p_est)
        e_scale = 1.0

        if dist_gt > 0:
            e_scale = abs((dist_gt / dist_es))

        return e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale

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
        e_q_rmse_deg_vec = np.zeros(np.shape(p_est))
        e_rpy_vec = np.zeros(np.shape(p_est))
        e_q_vec = np.zeros(np.shape(q_est))  # [x,y,z,w]

        for i in range(np.shape(p_est)[0]):
            q_wb_est = q_est[i, :]  # [x,y,z,w] quaternion vector
            q_wb_gt = q_gt[i, :]  # [x,y,z,w] quaternion vector

            q_wb_est = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_est).unit()
            q_wb_gt = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_gt).unit()

            # Error definitions for global orientation perturbation:
            # R_wb_gt = R_wb_err * R_wb_est
            # R_wb_err = R_wb_gt * R_wb_est'
            q_wb_err = q_wb_gt * q_wb_est.conj()

            # Error definition for global position perturbation
            R_G_Gest_err = q_wb_err.R
            p_GB_in_G_gt = p_gt[i, :]
            p_Gest_B_in_Gest_est = p_est[i, :]
            p_G_Gest_in_G_err = p_GB_in_G_gt - np.dot(R_G_Gest_err, p_Gest_B_in_Gest_est)

            e_p_vec[i, :] = p_G_Gest_in_G_err
            e_q_vec[i, :] = SpatialConverter.UnitQuaternion_to_HTMQ_quaternion(q_wb_err)
            e_rpy_vec[i, :] = q_wb_err.rpy(order='xyz', unit='rad')
            e_q_rmse_deg_vec[i] = np.rad2deg(np.linalg.norm(e_rpy_vec[i, :]))

        # global absolute position RMSE
        e_p_rmse_vec = np.sqrt(np.sum(e_p_vec ** 2, 1))

        # scale drift
        dist_gt = total_distance(p_gt)
        dist_es = total_distance(p_est)
        e_scale = 1.0

        if dist_gt > 0:
            e_scale = abs((dist_gt / dist_es))

        return e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale

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
        e_q_rmse_deg_vec = np.zeros(np.shape(p_est))
        e_rpy_vec = np.zeros(np.shape(p_est))
        e_q_vec = np.zeros(np.shape(q_est))  # [x,y,z,w]

        for i in range(np.shape(p_est)[0]):
            q_wb_est = q_est[i, :]  # [x,y,z,w] quaternion vector
            q_wb_gt = q_gt[i, :]  # [x,y,z,w] quaternion vector

            q_wb_est = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_est).unit()
            q_wb_gt = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_wb_gt).unit()

            # Error definitions for global orientation perturbation:
            # R_wb_gt = R_wb_est * R_wb_err
            # R_wb_err = R_wb_est' * R_wb_gt
            q_wb_err = q_wb_est.conj() * q_wb_gt

            # Error definition for global position perturbation
            R_G_Best_est = q_wb_est.R
            p_GB_in_B = p_gt[i, :]
            p_G_Best_in_G_est = p_est[i, :]
            p_Best_B_in_Best_err = np.dot(np.transpose(R_G_Best_est), (p_GB_in_B - p_G_Best_in_G_est))

            e_p_vec[i, :] = p_Best_B_in_Best_err
            e_q_vec[i, :] = SpatialConverter.UnitQuaternion_to_HTMQ_quaternion(q_wb_err)
            e_rpy_vec[i, :] = q_wb_err.rpy(order='xyz', unit='rad')
            e_q_rmse_deg_vec[i] = np.rad2deg(np.linalg.norm(e_rpy_vec[i, :]))

        # global absolute position RMSE
        e_p_rmse_vec = np.sqrt(np.sum(e_p_vec ** 2, 1))

        # scale drift
        dist_gt = total_distance(p_gt)
        dist_es = total_distance(p_est)
        e_scale = 1.0

        if dist_gt > 0:
            e_scale = abs((dist_gt / dist_es))

        return e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale
