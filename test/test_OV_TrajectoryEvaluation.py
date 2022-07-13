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
from cnspy_trajectory_evaluation.TrajectoryEvaluation import *

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

class OV_TrajectoryEvaluation_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print(str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]")

    def get_fn(self):
        fn_gt_csv = str(SAMPLE_DATA_DIR + '/MH01_ov_gt.csv')
        fn_est_csv = str(SAMPLE_DATA_DIR + '/MH01_ov_est.csv')
        return fn_gt_csv, fn_est_csv

    def test_posyaw(self):
        self.start()
        fn_gt_csv, fn_est_csv = self.get_fn()

        alignment_type=TrajectoryAlignmentTypes.posyaw
        num_aligned_samples=1
        eval = TrajectoryEvaluation(fn_gt_csv, fn_est_csv, result_dir=str(SAMPLE_DATA_DIR + '/results/eval_ov/'),
                                    prefix='eval-ID1-'+str(alignment_type) + '_' + str(num_aligned_samples) + '-',
                                    alignment_type=alignment_type,
                                    num_aligned_samples=num_aligned_samples,
                                    plot=True, save_plot=True,
                                    est_err_type=EstimationErrorType.type5,
                                    rot_err_rep=ErrorRepresentationType.theta_R)

    def test_sim3(self):
        self.start()
        fn_gt_csv, fn_est_csv = self.get_fn()
        alignment_type=TrajectoryAlignmentTypes.sim3
        num_aligned_samples=-1

        eval = TrajectoryEvaluation(fn_gt_csv, fn_est_csv, result_dir=str(SAMPLE_DATA_DIR + '/results/eval_ov/'),
                                    prefix='eval-ID1-'+str(alignment_type) + '_' + str(num_aligned_samples) + '-',
                                    alignment_type=alignment_type,
                                    num_aligned_samples=num_aligned_samples,
                                    plot=True, save_plot=True,
                                    est_err_type=EstimationErrorType.type5,
                                    rot_err_rep=ErrorRepresentationType.theta_R)

        self.stop()


if __name__ == "__main__":
    unittest.main()
