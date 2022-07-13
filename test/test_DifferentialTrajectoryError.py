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
from cnspy_spatial_csv_formats.CSVSpatialFormat import CSVSpatialFormat, EstimationErrorType, ErrorRepresentationType, CSVSpatialFormatType
from cnspy_trajectory.TrajectoryEstimated import TrajectoryEstimated
from cnspy_trajectory_evaluation.AbsoluteTrajectoryError import *
from cnspy_trajectory.TrajectoryPlotter import TrajectoryPlotter, TrajectoryPlotConfig, TrajectoryPlotTypes
from cnspy_trajectory.TrajectoryErrorType import TrajectoryErrorType
from cnspy_trajectory.SpatialConverter import SpatialConverter
from spatialmath import SO3

from cnspy_trajectory_evaluation.AssociatedTrajectories import AssociatedTrajectories
from cnspy_trajectory_evaluation.DifferentialTrajectoryError import DifferentialTrajectoryError

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

class DifferentialTrajectoryError_Test(unittest.TestCase):

    def get_trajectories(self):
        fn_gt  = str(SAMPLE_DATA_DIR + '/ID1-pose-est-posorient-cov.csv')
        fn_est = str(SAMPLE_DATA_DIR + '/ID1-pose-gt.csv')

        assoc = AssociatedTrajectories(fn_gt=fn_gt, fn_est=fn_est)
        assoc.plot_timestamps(cfg=TrajectoryPlotConfig(show=False))
        return assoc.get_trajectories()  # est, gt


    def test_DTE(self):
        traj_est, traj_gt = self.get_trajectories()

        R_GN = SO3.Rz(135, unit='deg')
        R_GN = np.array(R_GN.R)
        p_GN_in_G = np.array([1, 2, 4])
        traj_gt.transform(scale=1.0, p_GN_in_G=p_GN_in_G, R_GN=R_GN)

        TrajectoryPlotter.multi_plot_3D(traj_list=[traj_gt, traj_est],
                                        cfg=TrajectoryPlotConfig(show=False),
                                        name_list=['gt_matched', 'est_matched'])
        DTE = DifferentialTrajectoryError(traj_est=traj_est, traj_gt=traj_gt)
        DTE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        DTE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        DTE.plot_pose(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          plot_angle=True, plot_distance=True)
        DTE.plot_pose(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
            plot_angle=True)

        ARMSE_p, ARMSE_q = DTE.get_ARMSE()
        print('test_DTE done:ARMSE p={:.2f}, q={:.2f}'.format(ARMSE_p, ARMSE_q))



if __name__ == "__main__":
    unittest.main()
