import os
from Trajectory import Trajectory
from TrajectoryPlotter import TrajectoryPlotConfig
import math
import numpy as np
import transformations as tf
import matplotlib.pyplot as plt
from TrajectoryPlotter import TrajectoryPlotConfig, TrajectoryPlotTypes, TrajectoryPlotter


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
    err_scale_perc = None
    rmse_p_vec = None
    rmse_rot_vec = None  # degree
    t_vec = None

    traj_err = None
    traj_est = None

    def __init__(self, traj_est, traj_gt):
        assert (isinstance(traj_est, Trajectory))
        assert (isinstance(traj_gt, Trajectory))

        assert (traj_est.num_elems() == traj_gt.num_elems())

        self.err_p_vec = np.abs(traj_gt.p_vec - traj_est.p_vec)
        # self.err_p_vec = traj_gt.p_vec.sub(traj_est.p_vec).abs()

        e_trans_rmse, e_trans_vec, e_rot_rmse_deg, e_rpy, e_scale_perc, e_q_vec = \
            AbsoluteTrajectoryError.compute_absolute_error(traj_est.p_vec, traj_est.q_vec, traj_gt.p_vec, traj_gt.q_vec)

        self.err_p_vec = e_trans_vec
        self.err_q_vec = e_q_vec
        self.err_rpy_vec = e_rpy
        self.err_scale_perc = e_scale_perc
        self.rmse_p_vec = e_trans_rmse
        self.rmse_rot_vec = e_rot_rmse_deg
        self.t_vec = traj_est.t_vec - traj_est.t_vec[0]

        self.traj_err = Trajectory(t_vec=self.t_vec, p_vec=self.err_p_vec, q_vec=self.err_q_vec)
        self.traj_est = traj_est

    def plot_pose_err(self, cfg=TrajectoryPlotConfig(), angles=False):
        return TrajectoryPlotter.plot_pose_err(TrajectoryPlotter(traj_obj=self.traj_est, config=cfg),
                                               TrajectoryPlotter(traj_obj=self.traj_err, config=cfg), cfg=cfg,
                                               angles=angles)

    def plot_p_err(self, cfg=TrajectoryPlotConfig(), fig=None, ax=None):
        plotter = TrajectoryPlotter(traj_obj=self.traj_err, config=cfg)

        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
        if ax is None:
            ax = fig.add_subplot(111)

        plotter.ax_plot_pos(ax=ax, cfg=cfg)
        ax.set_ylabel('position err [m]')

        TrajectoryPlotter.show_save_figure(cfg, fig)
        return fig, ax, plotter

    def plot_rpy_err(self, cfg=TrajectoryPlotConfig(), fig=None, ax=None):
        plotter = TrajectoryPlotter(traj_obj=self.traj_err, config=cfg)

        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
            ax = fig.add_subplot(111)
        plotter.ax_plot_rpy(ax=ax, cfg=cfg)
        if cfg.radians:
            ax.set_ylabel('rotation err [rad]')
        else:
            ax.set_ylabel('rotation err [deg]')

        TrajectoryPlotter.show_save_figure(cfg, fig)
        return fig, ax, plotter

    @staticmethod
    def compute_absolute_error(p_est, q_est, p_gt, q_gt):
        e_trans_vec = (p_gt - p_est)
        e_trans_rmse = np.sqrt(np.sum(e_trans_vec ** 2, 1))

        # orientation error
        e_rot_rmse_deg = np.zeros((len(e_trans_rmse, )))
        e_rpy = np.zeros(np.shape(p_est))
        e_q_vec = np.zeros((len(e_trans_rmse), 4))  # x0, y0, z0, w0
        for i in range(np.shape(p_est)[0]):
            R_we = tf.matrix_from_quaternion(q_est[i, :])
            R_wg = tf.matrix_from_quaternion(q_gt[i, :])
            e_R = np.dot(R_we, np.linalg.inv(R_wg))
            e_q_vec[i, :] = tf.quaternion_from_matrix(e_R)
            e_rpy[i, :] = tf.euler_from_matrix(e_R, 'rxyz')
            e_rot_rmse_deg[i] = np.rad2deg(np.linalg.norm(tf.logmap_so3(e_R[:3, :3])))

        # scale drift
        motion_gt = np.diff(p_gt, 0)
        motion_es = np.diff(p_est, 0)
        dist_gt = np.sqrt(np.sum(np.multiply(motion_gt, motion_gt), 1))
        dist_es = np.sqrt(np.sum(np.multiply(motion_es, motion_es), 1))
        e_scale_perc = np.abs((np.divide(dist_es, dist_gt) - 1.0) * 100)

        return e_trans_rmse, e_trans_vec, e_rot_rmse_deg, e_rpy, e_scale_perc, e_q_vec


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time
import csv
from TrajectoryPlotter import TrajectoryPlotter, TrajectoryPlotConfig


class AbsoluteTrajectoryError_Test(unittest.TestCase):

    def get_trajectories(self):
        traj_est = Trajectory()
        self.assertTrue(traj_est.load_from_CSV('../results/traj_est_matched_aligned.csv'))
        traj_gt = Trajectory()
        self.assertTrue(traj_gt.load_from_CSV('../results/traj_gt_matched_aligned.csv'))
        return traj_est, traj_gt

    def test_ATE(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est, traj_gt)
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        print('done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=False, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist),
            angles=True)
        print('done')


if __name__ == "__main__":
    unittest.main()
