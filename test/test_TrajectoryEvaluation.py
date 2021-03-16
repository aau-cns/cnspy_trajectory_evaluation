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
from trajectory_evaluation.TrajectoryEvaluation import *

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

class TrajectoryEvaluation_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print(str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]")

    def get_fn(self):
        fn_gt_csv = str(SAMPLE_DATA_DIR + '/ID1-pose-gt.csv')
        fn_est_csv = str(SAMPLE_DATA_DIR + '/ID1-pose-est-cov.csv')
        return fn_gt_csv, fn_est_csv

    def test_init(self):
        self.start()
        fn_gt_csv = str(SAMPLE_DATA_DIR + '/ID1-pose-gt.csv')
        fn_est_csv = str(SAMPLE_DATA_DIR + '/ID1-pose-est-cov.csv')
        eval = TrajectoryEvaluation(fn_gt_csv, fn_est_csv, result_dir=str(SAMPLE_DATA_DIR + '/results/eval'), prefix='eval-ID1-',
                                    alignment_type=TrajectoryAlignmentTypes.none)

        self.stop()


if __name__ == "__main__":
    unittest.main()
