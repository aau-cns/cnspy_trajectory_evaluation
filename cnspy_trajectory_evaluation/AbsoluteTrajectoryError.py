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


class AbsoluteTrajectoryError:
    err_p_vec = None
    err_q_vec = None
    err_rpy_vec = None
    err_scale = None
    rmse_p_vec = None      # [m]
    rmse_q_deg_vec = None  # [deg]
    t_vec = None

    traj_err = None
    traj_est = None
    traj_gt = None
    ARMSE_p = None          # [m]
    ARMSE_q_deg = None      # [deg]

    def __init__(self, traj_est, traj_gt):
        assert (isinstance(traj_est, Trajectory))
        assert (isinstance(traj_gt, Trajectory))

        assert traj_est.num_elems() == traj_gt.num_elems(), "Traj. have to be matched in time and aligned first"

        self.err_p_vec = np.abs(traj_gt.p_vec - traj_est.p_vec)
        # self.err_p_vec = traj_gt.p_vec.sub(traj_est.p_vec).abs()

        e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale = \
            AbsoluteTrajectoryError.compute_absolute_error(p_est=traj_est.p_vec, q_est=traj_est.q_vec, p_gt=traj_gt.p_vec, q_gt=traj_gt.q_vec)

        self.err_p_vec = e_p_vec
        self.err_q_vec = e_q_vec
        self.err_rpy_vec = e_rpy_vec
        self.err_scale = e_scale
        self.rmse_p_vec = e_p_rmse_vec
        self.rmse_q_deg_vec = e_q_rmse_deg_vec
        self.t_vec = traj_est.t_vec - traj_est.t_vec[0]

        self.traj_err = Trajectory(t_vec=self.t_vec, p_vec=self.err_p_vec, q_vec=self.err_q_vec)
        self.traj_est = traj_est
        self.traj_gt = traj_gt
        self.ARMSE_p = np.mean(self.rmse_p_vec)
        self.ARMSE_q_deg = np.mean(self.rmse_q_deg_vec)

    def plot_pose_err(self, cfg=TrajectoryPlotConfig(), angles=False):
        # TODO: ugly, as error plot are defined here below, but are not used!
        return TrajectoryPlotter.plot_pose_err(TrajectoryPlotter(traj_obj=self.traj_est, config=cfg),
                                               TrajectoryPlotter(traj_obj=self.traj_err, config=cfg), cfg=cfg,
                                               angles=angles,
                                               plotter_gt=TrajectoryPlotter(traj_obj=self.traj_gt, config=cfg))

    def plot_p_err(self, cfg=TrajectoryPlotConfig(), fig=None, ax=None):
        plotter = TrajectoryPlotter(traj_obj=self.traj_err, config=cfg)

        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
        if ax is None:
            ax = fig.add_subplot(111)

        plotter.ax_plot_pos(ax=ax, cfg=cfg)
        ax.set_ylabel('(p_EST - p_GT) ARMSE ={:.2f} [m]'.format(self.ARMSE_p))

        TrajectoryPlotConfig.show_save_figure(cfg, fig)
        return fig, ax, plotter

    def plot_rpy_err(self, cfg=TrajectoryPlotConfig(), fig=None, ax=None):
        plotter = TrajectoryPlotter(traj_obj=self.traj_err, config=cfg)

        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
            ax = fig.add_subplot(111)
        plotter.ax_plot_rpy(ax=ax, cfg=cfg)
        if cfg.radians:
            ax.set_ylabel('(inv(q_GT) * q_EST) ARMSE ={:.2f} [rad]'.format(np.deg2rad(self.ARMSE_q_deg)))
        else:
            ax.set_ylabel('(inv(q_GT) * q_EST) ARMSE ={:.2f} [deg]'.format(self.ARMSE_q_deg))

            TrajectoryPlotConfig.show_save_figure(cfg, fig)
        return fig, ax, plotter

    @staticmethod
    def compute_absolute_error(p_est, q_est, p_gt, q_gt):
        """
        computes the absolute cnspy_trajectory error between a estimated and ground-truth cnspy_trajectory
        - The position error is global w.r.t to  ground truth
        - The orientation error is a local error/perturbation w.r.t to ground truth
        > p_AB_err = p_AB_est - p_AB_gt
        > R_AB_err = R_AB_gt' * R_AB_est
        """
        p_rows, p_cols = p_est.shape
        q_rows, q_cols = q_est.shape
        assert(p_est.shape == p_gt.shape)
        assert(q_est.shape == q_gt.shape)
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
            q_wb_est = q_est[i, :]
            q_wb_gt = q_gt[i, :]

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

