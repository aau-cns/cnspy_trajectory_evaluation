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
import os
from sys import version_info

import numpy as np
from matplotlib import pyplot as plt

from cnspy_csv2dataframe.CSV2DataFrame import CSV2DataFrame
from cnspy_spatial_csv_formats.CSVSpatialFormatType import CSVSpatialFormatType
from cnspy_timestamp_association.TimestampAssociation import TimestampAssociation
from cnspy_trajectory.PlotLineStyle import PlotLineStyle
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryBase import TrajectoryBase
from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated
from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory.TrajectoryPlotUtils import TrajectoryPlotUtils


class AssociatedTrajectories:
    csv_df_gt = None
    csv_df_est = None

    data_frame_gt_matched = None
    data_frame_est_matched = None

    matches_est_gt = None  # list of tuples containing [(idx_est, idx_gt), ...]

    def __init__(self, fn_gt, fn_est, relative_timestamps=False, max_difference=0.02):
        assert (os.path.exists(fn_gt))
        assert (os.path.exists((fn_est)))

        self.csv_df_gt = CSV2DataFrame(fn=fn_gt)
        assert (self.csv_df_gt.data_loaded)
        self.csv_df_est = CSV2DataFrame(fn=fn_est)
        assert (self.csv_df_est.data_loaded)

        if version_info[0] < 3:
            t_vec_gt = self.csv_df_gt.data_frame.as_matrix(['t'])
            t_vec_est = self.csv_df_est.data_frame.as_matrix(['t'])
            t_zero = min(t_vec_gt[0], t_vec_est[0])
            if relative_timestamps:
                self.csv_df_gt.data_frame[['t']] = self.csv_df_gt.data_frame[['t']] - t_zero
                self.csv_df_est.data_frame[['t']] = self.csv_df_est.data_frame[['t']] - t_zero
        else:
            # FIX(scm): for newer versions as_matrix is deprecated, using to_numpy instead
            # from https://stackoverflow.com/questions/60164560/attributeerror-series-object-has-no-attribute-as-matrix-why-is-it-error
            t_vec_gt = self.csv_df_gt.data_frame[['t']].to_numpy()
            t_vec_est = self.csv_df_est.data_frame[['t']].to_numpy()
            t_zero = min(t_vec_gt[0], t_vec_est[0])
            if relative_timestamps:
                self.csv_df_gt.data_frame[['t']] = self.csv_df_gt.data_frame[['t']] - t_zero
                self.csv_df_est.data_frame[['t']] = self.csv_df_est.data_frame[['t']] - t_zero


        if relative_timestamps:
            # only relative time stamps:
            t_vec_gt = t_vec_gt - t_zero
            t_vec_est = t_vec_est - t_zero


        idx_est, idx_gt, t_est_matched, t_gt_matched = TimestampAssociation.associate_timestamps(
            t_vec_est,
            t_vec_gt,
            max_difference=max_difference,
            round_decimals=6,
            unique_timestamps=True)


        self.data_frame_est_matched = self.csv_df_est.data_frame.loc[idx_est, :]
        self.data_frame_gt_matched = self.csv_df_gt.data_frame.loc[idx_gt, :]

        self.matches_est_gt = zip(idx_est, idx_gt)
        # using zip() and * operator to
        # perform Unzipping
        # res = list(zip(*test_list))

    def plot_timestamps(self, cfg=TrajectoryPlotConfig(), fig=None, ax=None, colors=['r', 'g'], labels=['gt', 'est'],
                    ls_vec=[PlotLineStyle(linestyle='-'), PlotLineStyle(linestyle='-.')]):
        assert (isinstance(cfg, TrajectoryPlotConfig))
        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
        if ax is None:
            ax = fig.add_subplot(111)
        if cfg.title:
            ax.set_title(cfg.title)

        if version_info[0] < 3:
            t_vec_gt = self.data_frame_gt_matched.as_matrix(['t'])
            t_vec_est = self.data_frame_est_matched.as_matrix(['t'])
        else:
            t_vec_gt = self.data_frame_gt_matched[['t']].to_numpy()
            t_vec_est = self.data_frame_est_matched[['t']].to_numpy()

        x_arr = range(len(t_vec_gt))
        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_arr, t_vec_gt, colors=[colors[0]], labels=[labels[0]], ls=ls_vec[0])
        x_arr = range(len(t_vec_est))
        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_arr, t_vec_est, colors=[colors[1]], labels=[labels[1]], ls=ls_vec[1])

        ax.grid(b=True)
        ax.set_xlabel('idx')
        ax.set_ylabel('time [s]')
        TrajectoryPlotConfig.show_save_figure(cfg, fig=fig)

        return fig, ax

    def save(self, result_dir=None, prefix=None):
        if not result_dir:
            fn_est_ = str(os.path.splitext(self.csv_df_est.fn)[0]) + "_matched.csv"
            fn_gt_ = str(os.path.splitext(self.csv_df_gt.fn)[0]) + "_matched.csv"
        else:
            if not prefix:
                prefix = ""
            if not os.path.exists(result_dir):
                os.makedirs(os.path.abspath(result_dir))
            fn_est_ = str(result_dir) + '/' + str(prefix) + "est_matched.csv"
            fn_gt_ = str(result_dir) + '/' + str(prefix) + "gt_matched.csv"

        CSV2DataFrame.save_CSV(self.data_frame_est_matched, filename=fn_est_, fmt=self.csv_df_est.format)
        CSV2DataFrame.save_CSV(self.data_frame_gt_matched, filename=fn_gt_, fmt=self.csv_df_gt.format)

    def get_trajectories(self):
        # returns Tr_est_matched, Tr_gt_matched
        if self.csv_df_est.format.has_uncertainty():
            return TrajectoryEstimated(df=self.data_frame_est_matched), Trajectory(df=self.data_frame_gt_matched)
        else:
            return Trajectory(df=self.data_frame_est_matched), Trajectory(df=self.data_frame_gt_matched)

    @staticmethod
    def associate_trajectories(traj_arr, max_difference=0, round_decimals=9, unique_timestamps=False):
        if isinstance(traj_arr, list) and len(traj_arr) > 1:
            traj_reference = traj_arr[0]
            traj_other_arr = traj_arr[1:]
            assert isinstance(traj_reference, TrajectoryBase)

            # First iteration, find common timestamps among all estimated trajectories
            common_gt_indices = np.arange(0, traj_reference.num_elems(), dtype=np.int32)
            for i in range(len(traj_other_arr)):
                traj_i = traj_other_arr[i]
                assert isinstance(traj_i, TrajectoryBase)
                indices_gt_matched, t_gt_matched  = traj_i.sample_at_t_arr(traj_reference.t_vec,
                                                                              max_difference=max_difference,
                                                                              round_decimals=round_decimals,
                                                                              unique_timestamps=unique_timestamps)

                common_gt_indices = np.intersect1d(common_gt_indices, indices_gt_matched)

            # Sample ground-truth trajectories based on common samples
            traj_reference.sample(common_gt_indices)

            # Second iteration, re-sample all estimated trajectories again!
            for i in range(len(traj_other_arr)):
                traj_i = traj_other_arr[i]
                assert isinstance(traj_i, TrajectoryBase)
                traj_i.sample_at_t_arr(traj_reference.t_vec,
                                           max_difference=max_difference,
                                           round_decimals=round_decimals,
                                           unique_timestamps=unique_timestamps)
        else:
            print("AssociatedTrajectories.associate_trajectories: expected a list of trajectories!")
