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
# Requirements:
# numpy, matplotlib
########################################################################################################################
from numpy_utils import transformations as tf
import matplotlib.pyplot as plt

from numpy_utils.accumulated_distance import *
from trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from trajectory.TrajectoryPlotTypes import TrajectoryPlotTypes
from trajectory.Trajectory import Trajectory
from trajectory.TrajectoryEstimated import TrajectoryEstimated


# TODO: NEES requires a Covariance of the pose
#  * PoseWithCovarianceStamped: covariance must be exported in CSV file and loaded:
#    * New loader like TUMCSV2DataFrame: PoseCovCSV2DataFrame: load and store to CSV
#    * rosbag2cvs: new header and extract features
#  * NEES class
#    * covariance inverse
#    * mahanobis stuff
#    * PoseCov-DataFrame needs to be matched

class AbsoluteTrajectoryError:
    err_p_vec = None
    err_q_vec = None
    err_rpy_vec = None
    err_scale = None
    rmse_p_vec = None
    rmse_q_deg_vec = None  # degree
    t_vec = None

    traj_err = None
    traj_est = None
    traj_gt = None
    ARMSE_p = None
    ARMSE_q_deg = None

    def __init__(self, traj_est, traj_gt):
        assert (isinstance(traj_est, Trajectory))
        assert (isinstance(traj_gt, Trajectory))

        assert (traj_est.num_elems() == traj_gt.num_elems())

        self.err_p_vec = np.abs(traj_gt.p_vec - traj_est.p_vec)
        # self.err_p_vec = traj_gt.p_vec.sub(traj_est.p_vec).abs()

        e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale = \
            AbsoluteTrajectoryError.compute_absolute_error(traj_est.p_vec, traj_est.q_vec, traj_gt.p_vec, traj_gt.q_vec)

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
        ax.set_ylabel('(p_GT - p_EST) ARMSE ={:.2f} [m]'.format(self.ARMSE_p))

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
            ax.set_ylabel('(q_GT - q_EST) ARMSE ={:.2f} [rad]'.format(np.deg2rad(self.ARMSE_q_deg)))
        else:
            ax.set_ylabel('(q_GT - q_EST) ARMSE ={:.2f} [deg]'.format(self.ARMSE_q_deg))

            TrajectoryPlotConfig.show_save_figure(cfg, fig)
        return fig, ax, plotter

    @staticmethod
    def compute_absolute_error(p_est, q_est, p_gt, q_gt):
        e_p_vec = (p_gt - p_est)
        e_p_rmse_vec = np.sqrt(np.sum(e_p_vec ** 2, 1))

        # orientation error
        e_q_rmse_deg_vec = np.zeros((len(e_p_rmse_vec, )))
        e_rpy_vec = np.zeros(np.shape(p_est))
        e_q_vec = np.zeros((len(e_p_rmse_vec), 4))  # x0, y0, z0, w0

        for i in range(np.shape(p_est)[0]):
            R_wb_est = tf.matrix_from_quaternion(q_est[i, :])
            R_wb_gt = tf.matrix_from_quaternion(q_gt[i, :])

            # R_wb_gt = R_wb_est * R_wb_err; w=World, b=Body
            # R_wb_err = R_wb_est' * R_wb_gt
            e_R = np.matmul(R_wb_est.T, R_wb_gt)
            e_q_vec[i, :] = tf.quaternion_from_matrix(e_R)
            e_rpy_vec[i, :] = tf.euler_from_matrix(e_R, 'rxyz')
            # e_q_rmse_deg_vec[i] = np.rad2deg(np.linalg.norm(tf.logmap_so3(e_R[:3, :3])))
            e_q_rmse_deg_vec[i] = np.rad2deg(np.linalg.norm(e_rpy_vec[i, :]))

        # scale drift
        dist_gt = total_distance(p_gt)
        dist_es = total_distance(p_est)
        e_scale = 1.0

        if dist_gt > 0:
            e_scale = abs((dist_gt / dist_es))

        return e_p_vec, e_p_rmse_vec, e_q_vec, e_rpy_vec, e_q_rmse_deg_vec, e_scale


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
from trajectory.TrajectoryPlotter import TrajectoryPlotter, TrajectoryPlotConfig


class AbsoluteTrajectoryError_Test(unittest.TestCase):

    def get_trajectories(self):
        traj_est = TrajectoryEstimated()
        self.assertTrue(traj_est.load_from_CSV('../sample_data/ID1-pose-est-cov.csv'))
        traj_gt = Trajectory()
        self.assertTrue(traj_gt.load_from_CSV('../sample_data/ID1-pose-gt.csv'))
        return traj_est, traj_gt

    def test_ATE(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est, traj_gt)
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        print('ATE1 done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist),
            angles=True)
        print('ATE1 done')

    def test_ATE2(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est, traj_gt)
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        print('ATE2 done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
            angles=True)
        print('ATE2 done:ARMSE p={:.2f}, q={:.2f}'.format(ATE.ARMSE_p, ATE.ARMSE_q_deg))


if __name__ == "__main__":
    unittest.main()
