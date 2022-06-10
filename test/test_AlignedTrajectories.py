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
########################################################################################################################
import os
import unittest
import time
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryPlotter import TrajectoryPlotter
from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory_evaluation.AlignedTrajectories import *

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

class AlignedTrajectories_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print("Process time: " + str((time.time() - self.start_time)))

    def get_associated(self):
        fn_gt_csv = str(SAMPLE_DATA_DIR + '/ID1-pose-gt.csv')
        fn_est_csv = str(SAMPLE_DATA_DIR + '/ID1-pose-est-posorient-cov.csv')
        return AssociatedTrajectories(fn_est=fn_est_csv, fn_gt=fn_gt_csv)

    def test_init(self):
        self.start()
        associated = self.get_associated()
        self.stop()

    def test_align_trajectories(self):
        associated = self.get_associated()
        aligned = AlignedTrajectories(associated=associated)
        aligned.save(result_dir=str(SAMPLE_DATA_DIR + '/results/'))
        traj_est_matched = Trajectory(df=associated.data_frame_est_matched)
        plot_gt = TrajectoryPlotter(traj_obj=aligned.traj_gt_matched)
        plot_est = TrajectoryPlotter(traj_obj=traj_est_matched)
        plot_est_aligned = TrajectoryPlotter(traj_obj=aligned.traj_est_matched_aligned)

        TrajectoryPlotter.multi_plot_3D(traj_plotter_list=[plot_gt, plot_est, plot_est_aligned],
                                        cfg=TrajectoryPlotConfig(),
                                        name_list=['gt_matched', 'est_matched', 'est_matched_aligned'])


if __name__ == "__main__":
    unittest.main()
