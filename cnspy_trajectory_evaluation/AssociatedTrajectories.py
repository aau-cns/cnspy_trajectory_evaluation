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
from cnspy_csv2dataframe.CSV2DataFrame import CSV2DataFrame
from cnspy_spatial_csv_formats.CSVFormatPose import CSVFormatPose
from cnspy_timestamp_association.TimestampAssociation import TimestampAssociation
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated


class AssociatedTrajectories:
    csv_df_gt = None
    csv_df_est = None

    data_frame_gt_matched = None
    data_frame_est_matched = None

    matches_est_gt = None  # list of tuples containing [(idx_est, idx_gt), ...]

    def __init__(self, fn_gt, fn_est):
        assert (os.path.exists(fn_gt))
        assert (os.path.exists((fn_est)))

        self.csv_df_gt = CSV2DataFrame(filename=fn_gt)
        assert (self.csv_df_gt.data_loaded)
        self.csv_df_est = CSV2DataFrame(filename=fn_est)
        assert (self.csv_df_est.data_loaded)

        if version_info[0] < 3:
            t_vec_gt = self.csv_df_gt.data_frame.as_matrix(['t'])
            t_vec_est = self.csv_df_est.data_frame.as_matrix(['t'])
        else:
            # FIX(scm): for newer versions as_matrix is deprecated, using to_numpy instead
            # from https://stackoverflow.com/questions/60164560/attributeerror-series-object-has-no-attribute-as-matrix-why-is-it-error
            t_vec_gt = self.csv_df_gt.data_frame[['t']].to_numpy()
            t_vec_est = self.csv_df_est.data_frame[['t']].to_numpy()

        idx_est, idx_gt, t_est_matched, t_gt_matched = TimestampAssociation.associate_timestamps(
            t_vec_est,
            t_vec_gt)

        self.data_frame_est_matched = self.csv_df_est.data_frame.loc[idx_est, :]
        self.data_frame_gt_matched = self.csv_df_gt.data_frame.loc[idx_gt, :]

        self.matches_est_gt = zip(idx_est, idx_gt)
        # using zip() and * operator to
        # perform Unzipping
        # res = list(zip(*test_list))

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
        if self.csv_df_est.format == CSVFormatPose.PoseWithCov:
            return TrajectoryEstimated(df=self.data_frame_est_matched), Trajectory(df=self.data_frame_gt_matched)
        else:
            return Trajectory(df=self.data_frame_est_matched), Trajectory(df=self.data_frame_gt_matched)
