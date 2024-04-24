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
from cnspy_trajectory_evaluation.AbsoluteTrajectoryError import AbsoluteTrajectoryError
from cnspy_trajectory.TrajectoryErrorType import TrajectoryErrorType
from cnspy_trajectory.TrajectoryPlotTypes import TrajectoryPlotTypes
from cnspy_trajectory_evaluation.EstimationTrajectoryError import EstimationTrajectoryError
from cnspy_trajectory_evaluation.TrajectoryPosOrientNEES import *
from cnspy_spatial_csv_formats.CSVSpatialFormat import CSVSpatialFormat
from cnspy_spatial_csv_formats.CSVSpatialFormatType import CSVSpatialFormatType
from cnspy_spatial_csv_formats.ErrorRepresentationType import ErrorRepresentationType
from cnspy_spatial_csv_formats.EstimationErrorType import EstimationErrorType

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

class TrajectoryPosOrientNEES_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print(str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]")

    def get_trajectories(self):
        traj_est = TrajectoryEstimated(fn=str(SAMPLE_DATA_DIR + '/ID1-pose-est-posorient-cov.csv'))
        #traj_NB.format = CSVSpatialFormat(fmt_type=CSVSpatialFormatType.PosOrientWithCov,
        #                                   est_err_type=EstimationErrorType.type5,
        #                                   err_rep_type=ErrorRepresentationType.theta_R)
        traj_gt = Trajectory(fn=str(SAMPLE_DATA_DIR + '/ID1-pose-gt.csv'))
        return traj_est, traj_gt

    def test_nees(self):
        self.start()
        traj_est, traj_gt = self.get_trajectories()
        self.stop('Loading')
        self.start()
        ATE = AbsoluteTrajectoryError(traj_est, traj_gt, traj_err_type=TrajectoryErrorType.local_pose())
        ARMSE_p, ARMSE_q_deg = ATE.traj_err.get_ARMSE()
        print('ARMSE p={:.2f} [m], q={:.2f} [deg]'.format(ARMSE_p, ARMSE_q_deg))
        self.stop('ATE')

        self.start()
        ETE = EstimationTrajectoryError(traj_est, traj_gt)
        self.stop('ETE')

        # TODO:
        #cfg = TrajectoryPlotConfig(show=True, close_figure=False,  radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t)
        #TrajectoryPlotter.plot_pose_err_cov(traj_NB=traj_NB, traj_err=ATE.traj_err, traj_GB=traj_GB, cfg=cfg)

        self.start()
        NEES = TrajectoryPosOrientNEES(traj_est, ETE.traj_est_err)
        avg_NEES_p, avg_NEES_R = NEES.get_avg_NEES()
        print('avg_NEES_p: ' + str(avg_NEES_p))
        print('avg_NEES_R: ' + str(avg_NEES_R))
        self.stop('NEES computation')

        NEES.save_to_CSV(str(SAMPLE_DATA_DIR + '/results/nees.csv'))

        NEES2 = TrajectoryPosOrientNEES(fn=str(SAMPLE_DATA_DIR + '/results/nees.csv'))
        NEES2.plot(cfg=TrajectoryPlotConfig(show=True, save_fn=str(SAMPLE_DATA_DIR + '/../../doc/pose-nees.png')))



if __name__ == "__main__":
    unittest.main()
