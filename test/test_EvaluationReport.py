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
from cnspy_trajectory_evaluation.EvaluationReport import *

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

class EvaluationReport_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print(str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]")

    def get_fn(self):
        fn_gt_csv = str(SAMPLE_DATA_DIR + '/ID1-pose-gt.csv')
        fn_est_csv = str(SAMPLE_DATA_DIR + '/ID1-pose-est-posorient-cov.csv')
        return fn_gt_csv, fn_est_csv

    def test_init(self):
        self.start()
        report = EvaluationReport()
        report.directory = ''
        report.fn_gt = str(SAMPLE_DATA_DIR + '/ID1-pose-gt.csv')
        report.fn_est = str(SAMPLE_DATA_DIR + '/ID1-pose-est-posorient-cov.csv')
        report.ANEES_p = 0.1
        report.ANEES_q = 0.2
        report.ARMSE_p = 0.3
        report.ARMSE_q = 0.4

        fn = str(SAMPLE_DATA_DIR + '/results/eval-report.ini')
        report.save(fn)

        report_ = EvaluationReport()
        report_.load(fn)

        self.assertTrue(report.fn_gt == report_.fn_gt)

        self.stop()


if __name__ == "__main__":
    unittest.main()
