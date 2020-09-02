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
from trajectory.PlotLineStyle import PlotLineStyle
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
from trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig


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

    def plot(self, fig=None, cfg=None):
        if cfg is None:
            cfg = TrajectoryPlotConfig()
        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))

        ax1 = fig.add_subplot(211)
        TrajectoryNEES.ax_plot_nees(ax1, self.NEES_p_vec, 3, conf_ival=0.997)
        ax1.set_ylabel('NEES pos')
        ax1.legend(shadow=True, fontsize='x-small')
        ax1.grid()
        ax2 = fig.add_subplot(212)
        TrajectoryNEES.ax_plot_nees(ax2, self.NEES_q_vec, 3, conf_ival=0.997)
        ax2.set_ylabel('NEES rot')
        ax2.legend(shadow=True, fontsize='x-small')
        ax2.grid()

        TrajectoryPlotConfig.show_save_figure(cfg, fig)

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

    @staticmethod
    def chi_square_confidence_bounds(confidence_interval=0.95, degrees_of_freedom=3):
        # https://stackoverflow.com/questions/53019080/chi2inv-in-python
        # ppf(q, df, loc=0, scale=1) Percent point function (inverse of cdf percentiles).
        return (chi2.ppf(q=(1.0 - confidence_interval), df=degrees_of_freedom),
                chi2.ppf(q=confidence_interval, df=degrees_of_freedom))

    @staticmethod
    def ax_plot_nees(ax, NEES_vec, dim, conf_ival, color='r', ls=PlotLineStyle()):
        l = NEES_vec.shape[0]
        ANEES = np.mean(NEES_vec)
        x_linespace = range(0, l)
        conf_ival = float(conf_ival)
        TrajectoryPlotter.ax_plot_n_dim(ax, x_linespace=x_linespace, values=NEES_vec,
                                        colors=[color], labels=['ANEES={:.3f}'.format(ANEES)], ls=ls)

        interval = TrajectoryNEES.chi_square_confidence_bounds(confidence_interval=conf_ival,
                                                               degrees_of_freedom=dim)
        y_values = np.ones((l, 1))
        TrajectoryPlotter.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * interval[1],
                                        colors=['k'],
                                        labels=['p={:.3f}->{:.3f}'.format(conf_ival, interval[1])],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='--'))
        TrajectoryPlotter.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * interval[0],
                                        colors=['k'],
                                        labels=['p={:.3f}->{:.3f}'.format(1.0 - conf_ival, interval[0])],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='--'))

        pass


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

        NEES.plot(cfg=TrajectoryPlotConfig(show=True))


if __name__ == "__main__":
    unittest.main()
