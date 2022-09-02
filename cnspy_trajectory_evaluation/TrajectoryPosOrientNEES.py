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
from sys import version_info
import numpy as np
import pandas as pandas
from spatialmath import base

from cnspy_spatial_csv_formats.ErrorRepresentationType import ErrorRepresentationType
from cnspy_spatial_csv_formats.EstimationErrorType import EstimationErrorType
from cnspy_trajectory.SpatialConverter import SpatialConverter
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryBase import TrajectoryBase
from cnspy_trajectory.TrajectoryEstimationError import TrajectoryEstimationError
from cnspy_trajectory.TrajectoryError import TrajectoryError
from cnspy_trajectory.TrajectoryPlotUtils import TrajectoryPlotUtils
from cnspy_trajectory.TrajectoryPlotter import TrajectoryPlotter
from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated
from cnspy_trajectory.PlotLineStyle import PlotLineStyle
from scipy.stats.distributions import chi2

import matplotlib.pyplot as plt

# TODO: the NEES depends on the state/error representation, meaning if the the state is in SE(3) we only obtain one
#  NEES value per time step, while if the state is a product manifold of R(3)xSO(3) we obtain two values! Per definition,
#  we should always consider a trajectory NEES from elements of SE(3). Meaning separation position and orientation is a
#  (dated) special case. So the error representation type decides in the end with NEES is computed, thus a factory needs
#  to be created that created the corresponding NEES object.

# Single run NEES of an estimated trajectory. Take multiple corresponding to one true trajectory, to obtain a propper ANEES
from cnspy_trajectory_evaluation.EstimationTrajectoryError import EstimationTrajectoryError


class TrajectoryPosOrientNEES(TrajectoryBase):
    NEES_p_vec = None
    NEES_R_vec = None
    num_run = 0
    # private metrics:
    # force access to average NEES via method get_avg_NEES()!
    __avg_NEES_p = None
    __avg_NEES_R = None


    def __init__(self, traj_est=None, traj_err=None, num_run=0, df=None, fn=None):
        TrajectoryBase.__init__(self)
        if fn is not None:
            self.load_from_CSV(fn)
        elif df is not None:
            self.load_from_DataFrame(df)
        elif traj_est is not None and traj_err is not None:
            assert (isinstance(traj_est, TrajectoryEstimated))
            assert (isinstance(traj_err, TrajectoryEstimationError))

            assert traj_est.format.rotation_error_representation == traj_err.err_rep_type
            assert traj_est.format.estimation_error_type == traj_err.est_err_type

            self.num_run = num_run
            self.t_vec = traj_est.t_vec
            if traj_err.err_rep_type == ErrorRepresentationType.tau_se3:
                Sigma_p_vec = traj_est.Sigma_T_vec[:, 0:3, 0:3]
                Sigma_R_vec = traj_est.Sigma_T_vec[:, 3:6, 3:6]
            else:
                Sigma_p_vec = traj_est.Sigma_p_vec
                Sigma_R_vec = traj_est.Sigma_R_vec

            self.NEES_p_vec = TrajectoryPosOrientNEES.toNEES_arr(Sigma_p_vec, traj_err.nu_vec)
            self.NEES_R_vec = TrajectoryPosOrientNEES.toNEES_arr(Sigma_R_vec, traj_err.theta_vec)

    # overriding abstract method
    def subsample(self, step=None, num_max_points=None, verbose=False):
        sparse_indices = Trajectory.subsample(self, step=step, num_max_points=num_max_points, verbose=verbose)

        self.NEES_p_vec = self.NEES_p_vec[sparse_indices]
        self.NEES_R_vec = self.NEES_R_vec[sparse_indices]
        return sparse_indices

    # overriding abstract method
    def sample(self, indices_arr, verbose=False):
        TrajectoryBase.sample(self, indices_arr=indices_arr)
        self.NEES_p_vec = self.NEES_p_vec[indices_arr]
        self.NEES_R_vec = self.NEES_R_vec[indices_arr]

    # overriding abstract method
    def clone(self):
        obj = TrajectoryPosOrientNEES()
        obj.NEES_p_vec = self.NEES_p_vec.copy()
        obj.NEES_R_vec = self.NEES_R_vec.copy()
        obj.num_run = self.num_run
        return obj

    def get_avg_NEES(self):
        if self.__avg_NEES_p is None:
            self.compute_avg_NEES()
        return self.__avg_NEES_p, self.__avg_NEES_R

    def compute_avg_NEES(self):
        self.__avg_NEES_p = np.mean(self.NEES_p_vec)
        self.__avg_NEES_R = np.mean(self.NEES_R_vec)

    def plot(self, fig=None, cfg=None):
        if cfg is None:
            cfg = TrajectoryPlotConfig()
        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))

        ax1 = fig.add_subplot(211)
        TrajectoryPosOrientNEES.ax_plot_nees(ax1, 3, conf_ival=0.997, NEES_vec=self.NEES_p_vec, x_linespace=self.t_vec)
        ax1.set_ylabel('NEES pos')
        ax1.legend(shadow=True, fontsize='x-small')
        ax1.grid()
        ax2 = fig.add_subplot(212)
        TrajectoryPosOrientNEES.ax_plot_nees(ax2, 3, conf_ival=0.997, NEES_vec=self.NEES_R_vec, x_linespace=self.t_vec)
        ax2.set_ylabel('NEES rot')
        ax2.legend(shadow=True, fontsize='x-small')
        ax2.grid()

        TrajectoryPlotConfig.show_save_figure(cfg, fig)

    # overriding abstract method
    def to_DataFrame(self):
        t_rows, t_cols = self.t_vec.shape
        p_rows, p_cols = self.NEES_p_vec.shape
        q_rows, q_cols = self.NEES_R_vec.shape
        assert (t_rows == p_rows)
        assert (t_rows == q_rows)
        assert (t_cols == 1)
        assert (p_cols == 1)
        assert (q_cols == 1)

        run_vec = np.repeat(self.num_run, self.num_elems(), axis=0)
        df = pandas.DataFrame({'t': self.t_vec[:, 0],
                               'NEES_p': self.NEES_p_vec[:, 0],
                               'NEES_R': self.NEES_R_vec[:, 0],
                               'num_run': run_vec.tolist()})
        return df

    # overriding abstract method
    def load_from_DataFrame(self, df, fmt_type=None):
        assert (isinstance(df, pandas.DataFrame))
        if version_info[0] < 3:
            self.t_vec = df.as_matrix(['t'])
            self.NEES_p_vec = df.as_matrix(['NEES_p'])
            self.NEES_R_vec = df.as_matrix(['NEES_R'])
            num_run_vec = df.as_matrix(['num_run'])
        else:
            self.t_vec = df[['t']].to_numpy()
            self.NEES_p_vec = df[['NEES_p']].to_numpy()
            self.NEES_R_vec = df[['NEES_R']].to_numpy()
            num_run_vec = df[['num_run']].to_numpy()
        self.num_run = num_run_vec[0]
        pass

    # https://de.mathworks.com/help/fusion/ref/trackerrormetrics-system-object.html
    @staticmethod
    def toNEES_arr(P_arr, err_arr):
        # if is_angle:
        #    err_arr = tf.euler_from_quaternion(err_arr, 'rzyx')

        l = err_arr.shape[0]
        nees_arr = np.zeros((l, 1))
        for i in range(0, l):
            try:
                nees_arr[i] = TrajectoryPosOrientNEES.toNEES(P=P_arr[i], err=err_arr[i])
            except np.linalg.LinAlgError:
                print("TrajectoryPosOrientNEES.toNEES(): covariance causes np.linalg.LinAlgError! ")
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
    def chi_square_confidence_bounds(confidence_region=0.95, degrees_of_freedom=3):
        # returns the [r_lower, r_upper] confidence regions
        # https://stackoverflow.com/questions/53019080/chi2inv-in-python
        # ppf(q, df, loc=0, scale=1) Percent point function (inverse of cdf percentiles).

        alpha = 1 - confidence_region
        r_upper = chi2.ppf(q=(1.0 - alpha), df=degrees_of_freedom)
        r_lower = chi2.ppf(q=alpha, df=degrees_of_freedom)
        return r_lower, r_upper


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

        r_lower, r_upper = TrajectoryPosOrientNEES.chi_square_confidence_bounds(confidence_region=conf_ival,
                                                                                degrees_of_freedom=dim)
        y_values = np.ones((l, 1))
        alpha = 1.0 - conf_ival
        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * r_upper,
                                        colors=['k'],
                                        labels=['r1(p={:.3f})={:.3f}'.format(conf_ival, r_upper)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='-.'))

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * r_lower,
                                        colors=['k'],
                                        labels=['r2(p={:.3f})={:.3f}'.format(conf_ival, r_lower)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='-.'))

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * dim,
                                        colors=['k'],
                                        labels=['mean={:.0f}'.format(dim)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='--'))

        ax.set_xlabel(x_label)
        pass
