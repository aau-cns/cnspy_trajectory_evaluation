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
# os
########################################################################################################################
import os
from trajectory_evaluation.TrajectoryAlignmentTypes import TrajectoryAlignmentTypes
from trajectory_evaluation.AlignedTrajectories import AlignedTrajectories
from trajectory_evaluation.AssociatedTrajectories import AssociatedTrajectories
from trajectory_evaluation.AbsoluteTrajectoryError import AbsoluteTrajectoryError
from trajectory_evaluation.TrajectoryNEES import TrajectoryNEES
from trajectory_evaluation.EvaluationReport import EvaluationReport
from trajectory.Trajectory import Trajectory
from trajectory.TrajectoryPlotter import TrajectoryPlotter
from trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from trajectory.TrajectoryPlotTypes import TrajectoryPlotTypes


class TrajectoryEvaluation:
    report = None

    def __init__(self, fn_gt, fn_est, result_dir=None, prefix=None,
                 alignment_type=TrajectoryAlignmentTypes.se3, num_aligned_samples=-1, plot=False, save_plot=False):
        if not result_dir:
            result_dir = '.'
        if not prefix:
            prefix = ''

        self.report = EvaluationReport(directory=os.path.abspath(result_dir), fn_gt=os.path.abspath(fn_gt),
                                       fn_est=os.path.abspath(fn_est),
                                       alignment=str(alignment_type), num_aligned_samples=num_aligned_samples)
        assoc = AssociatedTrajectories(fn_gt=fn_gt, fn_est=fn_est)
        assoc.save(result_dir=result_dir, prefix=prefix)

        aligned = AlignedTrajectories(associated=assoc, alignment_type=alignment_type, num_frames=num_aligned_samples)
        aligned.save(result_dir=result_dir, prefix=prefix)

        ATE = AbsoluteTrajectoryError(traj_est=aligned.traj_est_matched_aligned, traj_gt=aligned.traj_gt_matched)
        self.report.ARMSE_p = ATE.ARMSE_p
        self.report.ARMSE_q = ATE.ARMSE_q_deg
        ATE.traj_err.save_to_CSV(result_dir + '/' + prefix + 'err_matched_aligned.csv')

        NEES = TrajectoryNEES(traj_est=aligned.traj_est_matched_aligned, traj_err=ATE.traj_err)
        self.report.ANEES_p = NEES.ANEES_p
        self.report.ANEES_q = NEES.ANEES_q
        NEES.save_to_CSV(result_dir + '/' + prefix + 'nees_matched_aligned.csv')

        self.report.save(result_dir + '/' + prefix + 'report.ini')

        if plot or save_plot:
            fn_ATE = ""
            fn_NEES = ""
            fn_Multi = ""
            show = True
            if save_plot:
                fn_NEES = result_dir + '/' + prefix + 'NEES.jpg'
                fn_ATE = result_dir + '/' + prefix + 'ATE.jpg'
                fn_Multi = result_dir + '/' + prefix + 'traj3D.jpg'
                show = False

            est_matched, gt_matched = assoc.get_trajectories()
            plot_gt = TrajectoryPlotter(traj_obj=gt_matched)
            plot_est = TrajectoryPlotter(traj_obj=est_matched)
            plot_est_aligned = TrajectoryPlotter(traj_obj=aligned.traj_est_matched_aligned)

            TrajectoryPlotter.multi_plot_3D(traj_plotter_list=[plot_gt, plot_est, plot_est_aligned],
                                            cfg=TrajectoryPlotConfig(show=show, close_figure=True, save_fn=fn_Multi),
                                            name_list=['gt_matched', 'est_matched', 'est_matched_aligned'])
            ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=show, close_figure=True, radians=False,
                                                       plot_type=TrajectoryPlotTypes.plot_2D_over_t,
                                                       save_fn=fn_ATE), angles=True)

            NEES.plot(cfg=TrajectoryPlotConfig(show=show, close_figure=True, radians=False, save_fn=fn_NEES,
                                               plot_type=TrajectoryPlotTypes.plot_2D_over_t))


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time


class TrajectoryEvaluation_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print(str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]")

    def get_fn(self):
        fn_gt_csv = "./sample_data/ID1-pose-gt.csv"
        fn_est_csv = "./sample_data/ID1-pose-est-cov.csv"
        return fn_gt_csv, fn_est_csv

    def test_init(self):
        self.start()
        fn_gt_csv = "./sample_data/ID1-pose-gt.csv"
        fn_est_csv = "./sample_data/ID1-pose-est-cov.csv"
        eval = TrajectoryEvaluation(fn_gt_csv, fn_est_csv, result_dir='./results/eval', prefix='eval-ID1-',
                                    alignment_type=TrajectoryAlignmentTypes.none)

        self.stop()


if __name__ == "__main__":
    unittest.main()
