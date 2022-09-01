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
# cnspy_trajectory_evaluation
########################################################################################################################
import os

from cnspy_trajectory_evaluation.AssociatedTrajectories import AssociatedTrajectories
from cnspy_trajectory_evaluation.TrajectoryAlignmentTypes import TrajectoryAlignmentTypes


# TODO: align the gt trajectory to the estimated one, as the estimated global covariance
#  would need to be transformed as well!
class AlignedTrajectories:
    traj_est_matched_aligned = None
    traj_gt_matched = None
    alignment_type = TrajectoryAlignmentTypes.none

    def __init__(self, associated=None, traj_gt_matched=None, traj_est_matched=None,
                 alignment_type=TrajectoryAlignmentTypes.sim3, num_frames=-1):
        self.alignment_type = alignment_type

        if associated is not None:
            assert (isinstance(associated, AssociatedTrajectories))
            traj_est_matched, self.traj_gt_matched = associated.get_trajectories()
        else:
            self.traj_gt_matched = traj_gt_matched


        s, R_gt_est, t_gt_est_in_gt = TrajectoryAlignmentTypes.trajectory_aligment(traj_est_matched, self.traj_gt_matched,
                                                               method=alignment_type,
                                                               num_frames=num_frames)

        self.traj_est_matched_aligned = traj_est_matched.clone()
        self.traj_est_matched_aligned.transform(scale=s, p_GN_in_G=t_gt_est_in_gt, R_GN=R_gt_est)

    def save(self, result_dir='.', prefix=None):
        if not prefix:
            prefix = ""

        if not os.path.exists(result_dir):
            os.makedirs(os.path.abspath(result_dir))
        self.traj_est_matched_aligned.save_to_CSV(
            os.path.join(result_dir, str(prefix) + 'est_matched_aligned_' + str(self.alignment_type) + '.csv'))
        self.traj_gt_matched.save_to_CSV(
            os.path.join(result_dir, str(prefix) + 'gt_matched_aligned.csv'))

