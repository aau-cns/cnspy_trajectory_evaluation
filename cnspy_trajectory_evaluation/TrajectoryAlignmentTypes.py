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
# numpy, enum, cnspy_trajectory, cnspy_trajectory_evaluation
########################################################################################################################
import numpy as np
from enum import Enum

from cnspy_trajectory.Trajectory import Trajectory
from cnspy_trajectory_evaluation.SpatialAlignment import SpatialAlignement


class TrajectoryAlignmentTypes(Enum):
    sim3 = 'sim3'
    se3 = 'se3'
    posyaw = 'posyaw'
    pos = 'pos'
    none = 'none'

    def __str__(self):
        return self.value

    @staticmethod
    def list():
        return list([str(TrajectoryAlignmentTypes.sim3), str(TrajectoryAlignmentTypes.se3),
                     str(TrajectoryAlignmentTypes.posyaw), str(TrajectoryAlignmentTypes.pos),
                     str(TrajectoryAlignmentTypes.none)])

    @staticmethod
    def trajectory_aligment(traj_NB, traj_GB, method='sim3', num_frames=-1):
        """
        calculate s, R, t so that:
            traj_GB = TF * traj_NB
            TF =  [R_GN, p_GN_in_G ]
                  [ 0  ,    1] in SE(3)
        method can be: sim3, se3, posyaw, none;
        n_aligned: -1 means using all the frames

        """
        assert (isinstance(traj_NB, Trajectory))
        assert (isinstance(traj_GB, Trajectory))

        p_es = traj_NB.p_vec
        q_es = traj_NB.q_vec
        p_gt = traj_GB.p_vec
        q_gt = traj_GB.q_vec

        assert p_es.shape[1] == 3
        assert p_gt.shape[1] == 3
        assert q_es.shape[1] == 4
        assert q_gt.shape[1] == 4
        assert p_es.shape[0] == p_gt.shape[0]
        assert q_es.shape[0] == q_gt.shape[0]

        s_GN = 1
        R_GN = np.identity(3)
        p_GN_in_G = np.zeros((3,))

        # TODO: hackish, but it is somehow an Enum bug!
        method = TrajectoryAlignmentTypes(str(method))
        if method == TrajectoryAlignmentTypes.sim3:
            assert num_frames >= 2 or num_frames == -1, "sim3 uses at least 2 frames"
            s_GN, R_GN, p_GN_in_G = SpatialAlignement.align_SIM3(p_NB_in_N_arr=p_es, p_GB_in_G_arr=p_gt,
                                                                 n_aligned=num_frames)
        elif method == TrajectoryAlignmentTypes.se3:
            R_GN, p_GN_in_G = SpatialAlignement.align_SE3(p_NB_in_N_arr=p_es, p_GB_in_G_arr=p_gt, q_NB_arr=q_es,
                                                          q_GB_arr=q_gt, n_aligned=num_frames)
        elif method == TrajectoryAlignmentTypes.posyaw:
            R_GN, p_GN_in_G = SpatialAlignement.align_position_yaw(p_NB_in_N_arr=p_es, p_GB_in_G_arr=p_gt,
                                                                   q_NB_arr=q_es, q_GB_arr=q_gt, n_aligned=num_frames)
        elif method == TrajectoryAlignmentTypes.pos:
            p_GN_in_G = SpatialAlignement.align_position(p_NB_in_N_arr=p_es, p_GB_in_G_arr=p_gt, n_aligned=num_frames)
        elif method == TrajectoryAlignmentTypes.none:
            R_GN = np.identity(3)
            p_GN_in_G = np.zeros((3,))
        else:
            assert False, 'unknown alignment method'

        return s_GN, R_GN, p_GN_in_G
