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
# numpy, pandas, cnspy_numpy_utils, cnspy_trajectory, scipy, matplotlib
########################################################################################################################
import math
from joblib import Parallel, delayed
import numpy as np
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

from cnspy_spatial_csv_formats.ErrorRepresentationType import ErrorRepresentationType
from cnspy_spatial_csv_formats.EstimationErrorType import EstimationErrorType
from cnspy_trajectory.TrajectoryBase import TrajectoryBase
from cnspy_trajectory.PlotLineStyle import PlotLineStyle
from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory.TrajectoryPlotTypes import TrajectoryPlotTypes
from cnspy_trajectory.TrajectoryPlotUtils import TrajectoryPlotUtils


# - TODO: for now only p and q separately, se3 is not supported
# - How to compute the ANEES?
#   - given a set of estimated trajectories relating to one true trajectory
#   - perform timestamp association with respect to first associated true trajectory
#      - TODO: trajectory sampling using index array Trajectory.resample(index_array or t_arr)
#      - TODO: AssociatedTrajectories.associateTrajectoriesInplace(list<TrajectoryBase>):
#              iteratively associate the matched ones with unmatched ones
#   - compute  N-EstimationTrajectoryError error from N-Estimates and 1-GT
#   - compute  N-TrajectoryNEES given the Error and Estimate with Covariance
#   - compute  the Average over N-TrajectoryNEES, TODO: compute the NEES (Chi-Square) boundaries .
from cnspy_trajectory_evaluation.AlignedTrajectories import AlignedTrajectories
from cnspy_trajectory_evaluation.AssociatedTrajectories import AssociatedTrajectories
from cnspy_trajectory_evaluation.EstimationTrajectoryError import EstimationTrajectoryError
from cnspy_trajectory_evaluation.TrajectoryAlignmentTypes import TrajectoryAlignmentTypes
from cnspy_trajectory_evaluation.TrajectoryPosOrientNEES import TrajectoryPosOrientNEES


class TrajectoryANEES(TrajectoryBase):
    # t_vec = None

    # private metrics:
    # force access to ANEES via method get_ANEES()!
    ANEES_p_vec = None  # [-]
    ANEES_R_vec = None  # [-]
    __ANEES_p = None  # [-]
    __ANEES_R = None  # [-]
    M = None        # number of Monte-Carlo simulation runs

    def __init__(self, NEES_arr=None):
        TrajectoryBase.__init__(self)
        if NEES_arr is not None:
            M = len(NEES_arr)
            self.M = M
            if M > 0:
                self.ANEES_p_vec = NEES_arr[0].NEES_p_vec
                self.ANEES_R_vec = NEES_arr[0].NEES_R_vec
                self.t_vec = NEES_arr[0].t_vec
                if M > 1:
                    for i in range(M-1):
                        self.ANEES_p_vec = np.add(self.ANEES_p_vec, NEES_arr[i + 1].NEES_p_vec)
                        self.ANEES_R_vec = np.add(self.ANEES_R_vec, NEES_arr[i + 1].NEES_R_vec)


                    self.ANEES_p_vec = self.ANEES_p_vec / M
                    self.ANEES_R_vec = self.ANEES_R_vec / M
        pass

    def get_ANEES(self):
        if self.__ANEES_p is None or self.__ANEES_R is None:
            self.compute_ANEES()
        return self.__ANEES_p, self.__ANEES_R

    def compute_ANEES(self):
        self.__ANEES_p = np.mean(self.ANEES_p_vec)
        self.__ANEES_R = np.mean(self.ANEES_R_vec)

    # overriding abstract method
    def to_DataFrame(self):
        pass

    # overriding abstract method
    def load_from_DataFrame(self, df, fmt_type=None):
        pass

    # overriding abstract method
    def subsample(self, step=None, num_max_points=None, verbose=False):
        pass

    # overriding abstract method
    def sample(self, indices_arr, verbose=False):
        TrajectoryBase.sample(self, indices_arr=indices_arr)
        self.ANEES_p_vec = self.ANEES_p_vec[indices_arr]
        self.ANEES_R_vec = self.ANEES_R_vec[indices_arr]

    # overriding abstract method
    def clone(self):
        obj = TrajectoryANEES()
        obj.ANEES_p_vec = self.ANEES_p_vec.copy()
        obj.ANEES_R_vec = self.ANEES_R_vec.copy()
        obj.M = self.M
        return obj

    def plot(self, cfg=TrajectoryPlotConfig(), fig=None,
             x_label_prefix='',
             y_label_prefix='',
             color='k',
             color_bounds='k',
             ls=PlotLineStyle()):
        assert (isinstance(cfg, TrajectoryPlotConfig))

        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))

        ax1 = fig.add_subplot(211)
        x_arr = TrajectoryPlotUtils.ax_x_linespace(ax=ax1,
                                                   ts=self.t_vec,
                                                   relative_time=cfg.relative_time,
                                                   plot_type=cfg.plot_type,
                                                   x_label_prefix=x_label_prefix)

        ax1.set_ylabel(y_label_prefix + ' ANEES(p)')
        TrajectoryANEES.ax_plot_anees(ax1, self.ANEES_p_vec, self.M, dim=3, x_linespace=x_arr,
                                      color=color, color_bounds=color_bounds, ls=ls)
        ax1.legend(shadow=True, fontsize='x-small')
        ax1.grid()

        ax2 = fig.add_subplot(212)
        x_arr = TrajectoryPlotUtils.ax_x_linespace(ax=ax2,
                                                   ts=self.t_vec,
                                                   relative_time=cfg.relative_time,
                                                   plot_type=cfg.plot_type,
                                                   x_label_prefix=x_label_prefix)
        ax2.set_ylabel(y_label_prefix + ' ANEES(R)')
        TrajectoryANEES.ax_plot_anees(ax2, self.ANEES_R_vec, self.M, dim=3, x_linespace=x_arr,
                                      color=color, color_bounds=color_bounds, ls=ls)
        ax2.legend(shadow=True, fontsize='x-small')
        ax2.grid()
        TrajectoryPlotConfig.show_save_figure(cfg, fig)
        return fig, ax1, ax2


    @staticmethod
    def twosided_chi_squared_confidence_boundaries(M, confidence_region=0.95, degrees_of_freedom=3):
        # returns the [r_lower, r_upper] confidence bounds
        # tests: [2.96, 5.18] = twosided_chi_squared_confidence_boundaries(25, 0.95, 4)
        # M Monte Carlo runs
        # Sec. 3.7.6 in  Y. Bar-Shalom, X. Li, and T. Kirubarajan, Estimation with Applications to Tracking and Navigation: Theory Algorithms ... Wiley, 2001.
        # EQ (14) in "Scalable and Modular Ultra-Wideband Aided Inertial Navigation", Roland Jung and Stephan Weiss, IEEE IROS, 2022.

        alpha = 1 - confidence_region
        r_upper = chi2.ppf(q=(1.0 - 0.5*alpha), df=degrees_of_freedom*M)/M
        r_lower = chi2.ppf(q=(0.5*alpha), df=degrees_of_freedom*M)/M
        return r_lower, r_upper


    @staticmethod
    def process_trajectory_i(i, traj_gt, traj_est_arr, alignment_type, num_aligned_samples, est_err_type, rot_err_rep):
        traj_i = traj_est_arr[i]
        assert isinstance(traj_i, TrajectoryBase)
        aligned = AlignedTrajectories(traj_gt_matched=traj_gt.clone(),
                                      traj_est_matched=traj_i,
                                      alignment_type=alignment_type,
                                      num_frames=num_aligned_samples)

        # Manually specifying the estimation error type
        if isinstance(est_err_type, EstimationErrorType):
            aligned.traj_est_matched_aligned.format.estimation_error_type = est_err_type  # EstimationErrorType.type5
        if isinstance(rot_err_rep, ErrorRepresentationType):
            aligned.traj_est_matched_aligned.format.rotation_error_representation = rot_err_rep  # ErrorRepresentationType.theta_R

        ETE = EstimationTrajectoryError(traj_est=aligned.traj_est_matched_aligned, traj_gt=aligned.traj_gt_matched)
        NEES = TrajectoryPosOrientNEES(traj_est=aligned.traj_est_matched_aligned, traj_err=ETE.traj_est_err)

        traj_est = aligned.traj_est_matched_aligned
        return traj_est, ETE, NEES

    @staticmethod
    def evaluate(traj_gt, traj_est_arr, max_difference=0, round_decimals=9, unique_timestamps=False,
                 alignment_type=TrajectoryAlignmentTypes.se3, num_aligned_samples=-1,
                 est_err_type=None, rot_err_rep=None, max_processes=8):

        if isinstance(traj_est_arr, list):
            traj_arr = [traj_gt] + traj_est_arr
        else:
            traj_arr = [traj_gt, traj_est_arr]

        # timestamp association between all trajectories
        AssociatedTrajectories.associate_trajectories(traj_arr=traj_arr,
                                                      max_difference=max_difference,
                                                      round_decimals=round_decimals,
                                                      unique_timestamps=unique_timestamps)



        # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
        N = len(traj_est_arr)
        results = Parallel(n_jobs=min(N, max_processes), backend='multiprocessing')(delayed(TrajectoryANEES.process_trajectory_i)(i=i,
                                         traj_gt=traj_gt, traj_est_arr=traj_est_arr, alignment_type=alignment_type,
                                         num_aligned_samples=num_aligned_samples, est_err_type=est_err_type,
                                         rot_err_rep=rot_err_rep) for i in range(N))

        ETE_arr = []
        NEES_arr = []
        EST_aligned_arr = []
        for traj_est, ETE, NEES in results:
            EST_aligned_arr.append(traj_est)
            ETE_arr.append(ETE)
            NEES_arr.append(NEES)

        return NEES_arr, ETE_arr, EST_aligned_arr, traj_gt



    @staticmethod
    def ax_plot_anees(ax, ANEES_vec, M, dim=3, conf_ival=0.95, x_linespace=None,
                      color='r', color_bounds='darkred',
                      ls=PlotLineStyle()):
        l = ANEES_vec.shape[0]
        avg_ANEES = np.mean(ANEES_vec)

        if x_linespace is None:
            x_linespace = range(0, l)
            ax.set_xlabel('steps')

        conf_ival = float(conf_ival)
        r_lower, r_upper = TrajectoryANEES.twosided_chi_squared_confidence_boundaries(M, confidence_region=conf_ival,
                                                                                      degrees_of_freedom=dim)

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=ANEES_vec,
                                          colors=[color], labels=['avg. ANEES={:.3f}'.format(avg_ANEES)], ls=ls)

        y_values = np.ones((l, 1))
        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * r_lower,
                                        colors=[color_bounds],
                                        labels=['r1(p={:.3f})={:.3f}'.format(conf_ival, r_lower)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='-.'))

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * r_upper,
                                        colors=[color_bounds],
                                        labels=['r2(p={:.3f})={:.3f}'.format(conf_ival, r_upper)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='-.'))

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * dim,
                                        colors=[color_bounds],
                                        labels=['mean={:.1f}'.format(dim)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='--'))
