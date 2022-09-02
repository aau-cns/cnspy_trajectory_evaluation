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

from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated
from cnspy_trajectory_evaluation.TrajectoryANEES import TrajectoryANEES
from cnspy_trajectory_evaluation.TrajectoryEvaluation import *

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')


class TrajectoryANNES_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print(str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]")

    def get_trajectories(self):
        fn_gt_csv = str(SAMPLE_DATA_DIR + '/MC/ID1-pose-gt.csv')
        traj_gt = Trajectory(fn=fn_gt_csv)
        traj_est_arr = []
        for i in range(10):
            fn_est_csv = str(SAMPLE_DATA_DIR + '/MC/ID1-pose-est-posorient-cov-run{}.csv'.format(i+1))
            traj_est_i = TrajectoryEstimated(fn=fn_est_csv)
            traj_est_arr.append(traj_est_i)
        return traj_gt, traj_est_arr

    def test_init(self):
        self.start()
        traj_gt, traj_est_arr = self.get_trajectories()
        self.stop(info="Loading trajectories")

        self.start()
        NEES_arr, ETE_arr, EST_aligned_arr, traj_gt_matched = \
            TrajectoryANEES.evaluate(traj_gt,
                                     traj_est_arr,
                                     max_difference=0.01,
                                     round_decimals=4,
                                     unique_timestamps=True,
                                     alignment_type=TrajectoryAlignmentTypes.none,
                                     num_aligned_samples=1)
        self.stop(info="Evaluating trajectories")

        self.start()
        ANEES = TrajectoryANEES(NEES_arr=NEES_arr)
        self.stop(info="ANEES ")

        result_dir = str(SAMPLE_DATA_DIR + '/results/MC')
        for i in range(len(NEES_arr)):
            NEES_arr[i].plot(cfg=TrajectoryPlotConfig(show=True, close_figure=False,
                                            result_dir=result_dir, save_fn='NEES_{}.jpg'.format(i)))



        fn_ANEES = 'ANEES.jpg'
        ANEES.plot(cfg=TrajectoryPlotConfig(show=True, close_figure=False,
                                            result_dir=result_dir, save_fn=fn_ANEES))


if __name__ == "__main__":
    unittest.main()
