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
from cnspy_trajectory.PlotLineStyle import PlotLineStyle
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryBase import TrajectoryBase
from cnspy_trajectory.TrajectoryEstimationError import TrajectoryEstimationError

from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated
from cnspy_trajectory_evaluation.NEES import toNEES_arr, ax_plot_nees

import matplotlib.pyplot as plt

# TODO: the NEES depends on the state/error representation, meaning if the the state is in SE(3) we only obtain one
#  NEES value per time step, while if the state is a product manifold of R(3)xSO(3) we obtain two values! Per definition,
#  we should always consider a trajectory NEES from elements of SE(3). Meaning separation position and orientation is a
#  (dated) special case. So the error representation type decides in the end with NEES is computed, thus a factory needs
#  to be created that created the corresponding NEES object.

# Single run NEES of an estimated trajectory. Take multiple corresponding to one true trajectory, to obtain a propper ANEES
from cnspy_trajectory_evaluation.EstimationTrajectoryError import EstimationTrajectoryError


class TrajectoryPoseNEES(TrajectoryBase):
    NEES_T_vec = None
    num_run = 0
    # private metrics:
    # force access to average NEES via method get_avg_NEES()!
    __avg_NEES = None

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
            N = len(traj_est.t_vec)
            if traj_err.err_rep_type == ErrorRepresentationType.tau_se3 and traj_est.Sigma_T_vec is not None:
                Sigma_T_vec = traj_est.Sigma_T_vec
            else:
                N = len(traj_est.t_vec)
                Sigma_T_vec = np.zeros((N, 6, 6))
                Sigma_T_vec[:, 0:3, 0:3] = traj_est.Sigma_p_vec
                Sigma_T_vec[:, 3:6, 3:6] = traj_est.Sigma_R_vec

            tau_vec = np.zeros((N, 6))
            tau_vec[:, 0:3] = traj_err.nu_vec
            tau_vec[:, 3:6] = traj_err.theta_vec

            self.NEES_T_vec = toNEES_arr(Sigma_T_vec, tau_vec)

    # overriding abstract method
    def subsample(self, step=None, num_max_points=None, verbose=False):
        sparse_indices = Trajectory.subsample(self, step=step, num_max_points=num_max_points, verbose=verbose)

        self.NEES_T_vec = self.NEES_T_vec[sparse_indices]
        return sparse_indices

    # overriding abstract method
    def sample(self, indices_arr, verbose=False):
        TrajectoryBase.sample(self, indices_arr=indices_arr)
        self.NEES_T_vec = self.NEES_T_vec[indices_arr]

    # overriding abstract method
    def clone(self):
        obj = TrajectoryPoseNEES()
        obj.NEES_T_vec = self.NEES_T_vec.copy()
        obj.num_run = self.num_run
        return obj

    def get_avg_NEES(self):
        if self.__avg_NEES is None:
            self.compute_avg_NEES()
        return self.__avg_NEES

    def compute_avg_NEES(self):
        self.__avg_NEES = np.mean(self.NEES_T_vec)

    def plot(self, fig=None, cfg=None):
        if cfg is None:
            cfg = TrajectoryPlotConfig()
        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))

        ax1 = fig.add_subplot(111)
        self.plot_NEES(ax1=ax1, relative_time=cfg.relative_time)
        ax1.grid()

        TrajectoryPlotConfig.show_save_figure(cfg, fig)

    def plot_NEES(self, ax1, conf_ival=0.997, relative_time=True,
                  ls=PlotLineStyle()):
        ax_plot_nees(ax1, 6, conf_ival=conf_ival,
                     NEES_vec=self.NEES_T_vec,
                     x_linespace=self.t_vec,
                     relative_time=relative_time,
                     ls=ls,
                     y_label='NEES SE(3)')
        ax1.legend(shadow=True, fontsize='x-small')

    # overriding abstract method
    def to_DataFrame(self):
        t_rows, t_cols = self.t_vec.shape
        p_rows, p_cols = self.NEES_T_vec.shape
        assert (t_rows == p_rows)
        assert (t_cols == 1)
        assert (p_cols == 1)

        run_vec = np.repeat(self.num_run, self.num_elems(), axis=0)
        df = pandas.DataFrame({'t': self.t_vec[:, 0],
                               'NEES': self.NEES_T_vec[:, 0],
                               'num_run': run_vec.tolist()})
        return df

    # overriding abstract method
    def load_from_DataFrame(self, df, fmt_type=None):
        assert (isinstance(df, pandas.DataFrame))
        if version_info[0] < 3:
            self.t_vec = df.as_matrix(['t'])
            self.NEES_T_vec = df.as_matrix(['NEES'])
            num_run_vec = df.as_matrix(['num_run'])
        else:
            self.t_vec = df[['t']].to_numpy()
            self.NEES_T_vec = df[['NEES']].to_numpy()
            num_run_vec = df[['num_run']].to_numpy()
        self.num_run = num_run_vec[0]
        pass
