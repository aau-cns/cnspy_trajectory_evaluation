#!/usr/bin/env python
# Software License Agreement (GNU GPLv3  License)
#
# Copyright (c) 2022, Roland Jung (roland.jung@aau.at) , AAU, KPK, NAV
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
from enum import Enum


class AbsoluteTrajectoryErrorType(Enum):
    global_p_local_q = 'global_p_local_q'  # p_err = p_gt - p_est; R_err = R_gt' * R_est -> R_gt = R_est * R_err'; not in SE(3)
    global_pose = 'global_pose'  # p_err = p_gt - R_err * p_est; R_err = R_gt * R_est'; in SE(3)
    local_pose = 'local_pose'  # p_err = R_est'(p_gt - p_est); R_err = R_est' * R_gt; in SE(3)
    none = 'none'

    # HINT: if you add an entry here, please also add it to the .list() method!

    def __str__(self):
        return self.value

    def is_local_p(self):
        if self == AbsoluteTrajectoryErrorType.local_pose:
            return True
        else:
            return False

    def is_local_R(self):
        if self == AbsoluteTrajectoryErrorType.local_pose:
            return True
        if self == AbsoluteTrajectoryErrorType.global_p_local_q:
            return True
        else:
            return False

    def error_def(self):
        p_err_text = ''
        R_err_text = ''
        if self == AbsoluteTrajectoryErrorType.global_p_local_q:
            p_err_text = '(p_EST - p_GT)'
            R_err_text = '(inv(R_GT) * R_EST)'
        elif self == AbsoluteTrajectoryErrorType.local_pose:
            p_err_text = '(R_EST^(T)(p_EST - p_GT))'
            R_err_text = '(inv(R_EST) * R_GT)'
        elif self == AbsoluteTrajectoryErrorType.global_pose:
            p_err_text = '(p_GT - R_ERR*p_EST)'
            R_err_text = '(R_GT * inv(R_EST))'
        return [p_err_text, R_err_text]

    @staticmethod
    def list():
        return list([str(AbsoluteTrajectoryErrorType.global_p_local_q),
                     str(AbsoluteTrajectoryErrorType.global_pose),
                     str(AbsoluteTrajectoryErrorType.local_pose),
                     str(AbsoluteTrajectoryErrorType.none)])
