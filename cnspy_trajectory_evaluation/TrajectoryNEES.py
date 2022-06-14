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
# numpy, pandas, cnspy_numpy_utils, cnspy_trajectory, scipy, matplotlib
########################################################################################################################

import numpy as np
import pandas as pandas
from spatialmath import base

from cnspy_spatial_csv_formats.ErrorRepresentationType import ErrorRepresentationType
from cnspy_spatial_csv_formats.EstimationErrorType import EstimationErrorType
from cnspy_trajectory.SpatialConverter import SpatialConverter
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryEstimationError import TrajectoryEstimationError
from cnspy_trajectory.TrajectoryPlotUtils import TrajectoryPlotUtils
from cnspy_trajectory.TrajectoryPlotter import TrajectoryPlotter
from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated
from cnspy_trajectory.PlotLineStyle import PlotLineStyle
from scipy.stats.distributions import chi2

import matplotlib.pyplot as plt


class TrajectoryNEES:
    NEES_p_vec = None
    NEES_q_vec = None
    NEES_T_vec = None
    ANEES_p = None
    ANEES_q = None
    ANEES_T = None
    t_vec = None

    def __init__(self, traj_est, traj_err):
        assert (isinstance(traj_est, TrajectoryEstimated))
        assert (isinstance(traj_err, TrajectoryEstimationError))

        assert traj_est.format.rotation_error_representation == traj_err.err_rep_type
        assert traj_est.format.estimation_error_type == traj_err.est_err_type


        self.t_vec = traj_est.t_vec
        if traj_err.err_rep_type == ErrorRepresentationType.se3_tau:
            T_err_vec = np.hstack((traj_err.p_vec, traj_err.theta_q_vec))
            self.NEES_T_vec = TrajectoryNEES.toNEES_arr(traj_est.Sigma_T_vec, T_err_vec)
            self.ANEES_T = np.mean(self.NEES_T_vec)
        else:
            self.NEES_p_vec = TrajectoryNEES.toNEES_arr(traj_est.Sigma_p_vec, traj_err.p_vec)
            self.NEES_q_vec = TrajectoryNEES.toNEES_arr(traj_est.Sigma_R_vec, traj_err.theta_q_vec)
            self.ANEES_p = np.mean(self.NEES_p_vec)
            self.ANEES_q = np.mean(self.NEES_q_vec)

    def plot(self, fig=None, cfg=None):
        if cfg is None:
            cfg = TrajectoryPlotConfig()
        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))

        ax1 = fig.add_subplot(211)
        TrajectoryNEES.ax_plot_nees(ax1, 3, conf_ival=0.997, NEES_vec=self.NEES_p_vec, x_linespace=self.t_vec)
        ax1.set_ylabel('NEES pos')
        ax1.legend(shadow=True, fontsize='x-small')
        ax1.grid()
        ax2 = fig.add_subplot(212)
        TrajectoryNEES.ax_plot_nees(ax2, 3, conf_ival=0.997, NEES_vec=self.NEES_q_vec, x_linespace=self.t_vec)
        ax2.set_ylabel('NEES rot')
        ax2.legend(shadow=True, fontsize='x-small')
        ax2.grid()

        TrajectoryPlotConfig.show_save_figure(cfg, fig)

    def save_to_CSV(self, filename):
        t_rows, t_cols = self.t_vec.shape
        p_rows, p_cols = self.NEES_p_vec.shape
        q_rows, q_cols = self.NEES_q_vec.shape
        assert (t_rows == p_rows)
        assert (t_rows == q_rows)
        assert (t_cols == 1)
        assert (p_cols == 1)
        assert (q_cols == 1)
        data_frame = pandas.DataFrame({'t': self.t_vec[:, 0],
                                       'nees_p': self.NEES_p_vec[:, 0],
                                       'nees_q': self.NEES_q_vec[:, 0]})
        data_frame.to_csv(filename, sep=',', index=False,
                          header=['#t', 'nees_p', 'nees_rpy'],
                          columns=['t', 'nees_q', 'nees_rpy'])

    # https://de.mathworks.com/help/fusion/ref/trackerrormetrics-system-object.html
    @staticmethod
    def toNEES_arr(P_arr, err_arr):
        # if is_angle:
        #    err_arr = tf.euler_from_quaternion(err_arr, 'rzyx')

        l = err_arr.shape[0]
        nees_arr = np.zeros((l, 1))
        for i in range(0, l):
            try:
                nees_arr[i] = TrajectoryNEES.toNEES(P=P_arr[i], err=err_arr[i])
            except np.linalg.LinAlgError:
                print("TrajectoryNEES.toNEES(): covariance causes np.linalg.LinAlgError! ")
                print(P_arr[i])

        return nees_arr

    @staticmethod
    def toNEES(P, err):
        """
        computes the Normalized Estimation Error Square (Mahalanobis Distance Squared)
        The corresponding covariance matrix (P) must be in the same space and unit as the error!
        E.g., for local/body rotations the uncertainty and e.g. the error
        (err = theta_so3  with R_err = expm(theta_so3)) must be in the same space and unit
        """

        tr = np.trace(P)
        eig_vals, eig_vec = np.linalg.eig(P)
        if tr < 1e-16:
            nees = 0
        elif any(eig_vals < 0):
            nees = 0
        else:
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
    def ax_plot_nees(ax, dim, conf_ival, NEES_vec, x_linespace=None, color='r', ls=PlotLineStyle()):
        l = NEES_vec.shape[0]
        avg_NEES = np.mean(NEES_vec)

        x_label = 'rel. t [sec]'
        if x_linespace is None:
            x_linespace = range(0, l)
            x_label = ''

        conf_ival = float(conf_ival)
        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=NEES_vec,
                                        colors=[color], labels=['avg. NEES={:.3f}'.format(avg_NEES)], ls=ls)

        interval = TrajectoryNEES.chi_square_confidence_bounds(confidence_interval=conf_ival,
                                                               degrees_of_freedom=dim)
        y_values = np.ones((l, 1))
        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * interval[1],
                                        colors=['k'],
                                        labels=['p={:.3f}->{:.3f}'.format(conf_ival, interval[1])],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='-.'))

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * interval[0],
                                        colors=['k'],
                                        labels=['p={:.3f}->{:.3f}'.format(1.0 - conf_ival, interval[0])],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='-.'))

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * dim,
                                        colors=['k'],
                                        labels=['mean={:.1f}'.format(dim)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='--'))

        ax.set_xlabel(x_label)
        pass
