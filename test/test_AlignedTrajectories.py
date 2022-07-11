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
import matplotlib.pyplot as plt
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
        aligned = AlignedTrajectories(associated=associated, alignment_type=TrajectoryAlignmentTypes.se3, num_frames=1)

        cfg = TrajectoryPlotConfig()
        cfg.show = False
        fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
        ax = fig.add_subplot(111, projection='3d')
        aligned.traj_gt_matched.plot_3D(fig=fig, cfg=cfg, ax=ax, label='ground truth')
        alignment_list = TrajectoryAlignmentTypes.list()

        for i in range(len(alignment_list)):
            type_  = alignment_list[i]
            num_frames = 1
            if type_ is 'sim3':
                num_frames = 2
            aligned_ = AlignedTrajectories(associated=associated, alignment_type=TrajectoryAlignmentTypes(type_), num_frames=num_frames)
            aligned_.save(result_dir=str(SAMPLE_DATA_DIR + '/results/'), prefix=str(type_))
            aligned_.traj_est_matched_aligned.plot_3D(cfg=cfg, fig=fig, ax=ax, label=str(type_))

        plt.draw()
        plt.pause(0.001)
        plt.show()


        aligned.save(result_dir=str(SAMPLE_DATA_DIR + '/results/'), prefix='default')

        traj_est_matched = Trajectory(df=associated.data_frame_est_matched)
        TrajectoryPlotter.multi_plot_3D(traj_list=[aligned.traj_gt_matched, traj_est_matched, aligned.traj_est_matched_aligned],
                                        cfg=TrajectoryPlotConfig(),
                                        name_list=['gt_matched', 'est_matched', 'est_matched_aligned'])


if __name__ == "__main__":
    unittest.main()
