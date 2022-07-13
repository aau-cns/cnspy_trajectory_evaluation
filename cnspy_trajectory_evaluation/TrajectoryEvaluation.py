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

from cnspy_spatial_csv_formats.CSVSpatialFormat import CSVSpatialFormat
from cnspy_spatial_csv_formats.ErrorRepresentationType import ErrorRepresentationType
from cnspy_spatial_csv_formats.EstimationErrorType import EstimationErrorType
from cnspy_trajectory_evaluation.EstimationTrajectoryError import EstimationTrajectoryError
from cnspy_trajectory_evaluation.TrajectoryAlignmentTypes import TrajectoryAlignmentTypes
from cnspy_trajectory_evaluation.AlignedTrajectories import AlignedTrajectories
from cnspy_trajectory_evaluation.AssociatedTrajectories import AssociatedTrajectories
from cnspy_trajectory_evaluation.AbsoluteTrajectoryError import AbsoluteTrajectoryError
from cnspy_trajectory_evaluation.TrajectoryPosOrientNEES import TrajectoryPosOrientNEES
from cnspy_trajectory_evaluation.EvaluationReport import EvaluationReport
from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory.TrajectoryPlotter import TrajectoryPlotter
from cnspy_trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig
from cnspy_trajectory.TrajectoryPlotTypes import TrajectoryPlotTypes


class TrajectoryEvaluation:
    report = None

    def __init__(self, fn_gt, fn_est, result_dir=None, prefix=None,
                 alignment_type=TrajectoryAlignmentTypes.se3, num_aligned_samples=-1, plot=False, save_plot=False,
                 est_err_type=None, rot_err_rep=None):
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

        # Manually specifying the estimation error type
        if isinstance(est_err_type, EstimationErrorType):
            aligned.traj_est_matched_aligned.format.estimation_error_type = est_err_type # EstimationErrorType.type5
        if isinstance(rot_err_rep, ErrorRepresentationType):
            aligned.traj_est_matched_aligned.format.rotation_error_representation = rot_err_rep # ErrorRepresentationType.theta_R

        ATE = AbsoluteTrajectoryError(traj_est=aligned.traj_est_matched_aligned, traj_gt=aligned.traj_gt_matched)
        self.report.ARMSE_p, self.report.ARMSE_R = ATE.traj_err.get_ARMSE()
        ATE.traj_err.save_to_CSV(result_dir + '/' + prefix + 'err_matched_aligned.csv')

        ETE = EstimationTrajectoryError(traj_est=aligned.traj_est_matched_aligned, traj_gt=aligned.traj_gt_matched)
        NEES = TrajectoryPosOrientNEES(traj_est=aligned.traj_est_matched_aligned, traj_err=ETE.traj_est_err)
        self.report.ANEES_p, self.report.ANEES_R =  NEES.get_avg_NEES()
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
            est_matched.format = aligned.traj_est_matched_aligned.format

            TrajectoryPlotter.multi_plot_3D(traj_list=[gt_matched, est_matched, aligned.traj_est_matched_aligned],
                                            cfg=TrajectoryPlotConfig(show=show, close_figure=False, save_fn=fn_Multi),
                                            name_list=['gt_matched', 'est_matched', 'est_matched_aligned'])
            TrajectoryPlotter.plot_pose_err_cov(traj_gt=gt_matched, traj_est= aligned.traj_est_matched_aligned, traj_err=ATE.traj_err,
                                                cfg=TrajectoryPlotConfig(show=show, close_figure=False, radians=False,
                                                       plot_type=TrajectoryPlotTypes.plot_2D_over_t,
                                                       save_fn=fn_ATE))

            NEES.plot(cfg=TrajectoryPlotConfig(show=show, close_figure=True, radians=False, save_fn=fn_NEES,
                                               plot_type=TrajectoryPlotTypes.plot_2D_over_t))

