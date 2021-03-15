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
# numpy, pandas, numpy_utils, trajectory, scipy, matplotlib
########################################################################################################################
import unittest
import time
from trajectory_evaluation.AbsoluteTrajectoryError import AbsoluteTrajectoryError
from trajectory.TrajectoryPlotter import TrajectoryPlotter, TrajectoryPlotConfig
from trajectory.TrajectoryPlotTypes import TrajectoryPlotTypes
from trajectory_evaluation.TrajectoryNEES import *

class TrajectoryNEES_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print(str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]")

    def get_trajectories(self):
        traj_est = TrajectoryEstimated()
        self.assertTrue(traj_est.load_from_CSV('./sample_data/ID1-pose-est-cov.csv'))
        traj_gt = Trajectory()
        self.assertTrue(traj_gt.load_from_CSV('./sample_data/ID1-pose-gt.csv'))
        return traj_est, traj_gt

    def test_nees(self):
        self.start()
        traj_est, traj_gt = self.get_trajectories()
        ATE = AbsoluteTrajectoryError(traj_est, traj_gt)
        print('ARMSE p={:.2f}, q={:.2f}'.format(ATE.ARMSE_p, ATE.ARMSE_q_deg))
        self.stop('Loading + ATE')
        self.start()
        NEES = TrajectoryNEES(ATE.traj_est, ATE.traj_err)
        self.stop('NEES computation')
        print('ANEES_p: ' + str(NEES.ANEES_p))
        print('ANEES_q: ' + str(NEES.ANEES_q))

        NEES.plot(cfg=TrajectoryPlotConfig(show=True, save_fn='./doc/pose-nees.png'))
        NEES.save_to_CSV('./results/nees.csv')
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t, save_fn='./doc/pose-err-plot.png'),
            angles=True)


if __name__ == "__main__":
    unittest.main()
