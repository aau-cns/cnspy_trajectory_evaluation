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
# adapted from:
# * https://github.com/uzh-rpg/rpg_trajectory_evaluation/blob/master/src/rpg_trajectory_evaluation/align_trajectory.py
#
# Requirements:
# numpy, cnspy_numpy_utils
########################################################################################################################
import numpy as np
from spatialmath import UnitQuaternion, SO3
from cnspy_trajectory.SpatialConverter import SpatialConverter

class SpatialAlignement:
    @staticmethod
    def get_best_yaw(C):
        """
        maximize trace(Rz(theta) * C)

        Input:
        C -- rotation matrix (3x3)

        Output:
        theta -- scalar in radians
        """
        assert C.shape == (3, 3)

        A = C[0, 1] - C[1, 0]
        B = C[0, 0] + C[1, 1]
        theta = np.pi / 2 - np.arctan2(B, A)

        return theta

    @staticmethod
    def get_indices(n_aligned, total_n):
        """
        creates a vector reaching from 0 to max(n_aligned, total_n)

        Input:
        n_aligned -- integer, desired number
        total_n   -- max possible number

        Output:
        idxs -- vector 1xN of scalars
        """

        if n_aligned == -1:
            idxs = np.arange(0, total_n)
        else:
            idxs = np.arange(0, min(total_n, max(n_aligned, 1)))
        return idxs

    @staticmethod
    def align_position_yaw_single(est_p_arr, gt_p_arr, est_q_arr, gt_q_arr):
        """
        Calculate the 4DOF transformation: yaw R and translation t so that:
            gt = R * est + t
        Using only the first poses of est and gt

        Input:
        est_p_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        gt_p_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        est_q_arr -- estimated cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        gt_q_arr -- ground-truth cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type

        Output:
        R -- rotation matrix (3x3)     (R_gt_est)
        t -- translation vector (3x1)  (t_gt_est_in_gt)
        """

        p_es_0, q_es_0 = est_p_arr[0, :], est_q_arr[0, :]
        p_gt_0, q_gt_0 = gt_p_arr[0, :], gt_q_arr[0, :]

        q_es_0 = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_es_0).unit()
        q_gt_0 = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_gt_0).unit()
        q_0 = q_es_0 * q_gt_0.conj()

        theta = SpatialAlignement.get_best_yaw(q_0.R)
        R = SO3.Rz(theta)
        t = p_gt_0 - np.dot(R, p_es_0)

        return R, t

    @staticmethod
    def align_position_yaw(est_p_arr, gt_p_arr, est_q_arr, gt_q_arr, n_aligned=1):
        """
        Calculate the 4DOF transformation: yaw R and translation t so that:
            gt = R * est + t
        Using only the first poses of est and gt

        Input:
        est_p_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        gt_p_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        est_q_arr -- estimated cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        gt_q_arr -- ground-truth cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        n_aligned -- if 1, only the first matched pair will be used, otherwise align_Umeyama() with the a certain amount
                     will be called.

        Output:
        R -- rotation matrix (3x3)     (R_gt_est)
        t -- translation vector (3x1)  (t_gt_est_in_gt)
        """

        if n_aligned == 1:
            R, t = SpatialAlignement.align_position_yaw_single(est_p_arr, gt_p_arr, est_q_arr, gt_q_arr)
            return R, t
        else:
            idxs = SpatialAlignement.get_indices(n_aligned, est_p_arr.shape[0])
            est_pos = est_p_arr[idxs, 0:3]
            gt_pos = gt_p_arr[idxs, 0:3]
            _, R, t = SpatialAlignement.align_Umeyama(gt_pos, est_pos, known_scale=True,
                                                      yaw_only=True)  # note the order
            t = np.array(t)
            t = t.reshape((3,))
            R = np.array(R)
            return R, t

    @staticmethod
    def align_SE3_single(est_p_arr, gt_p_arr, est_q_arr, gt_q_arr):
        """
        Calculate SE3 transformation R and t so that:
            gt = R * est + t
        Using only the first poses of est and gt

        Input:
        est_p_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        gt_p_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        est_q_arr -- estimated cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        gt_q_arr -- ground-truth cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type

        Output:
        R -- rotation matrix (3x3)     (R_gt_est)
        t -- translation vector (3x1)  (t_gt_est_in_gt)
        """

        p_es_0 = est_p_arr[0, :]
        p_gt_0 = gt_p_arr[0, :]

        q_est_0 = SpatialConverter.HTMQ_quaternion_to_Quaternion(est_q_arr[0, :])
        q_gt_0 = SpatialConverter.HTMQ_quaternion_to_Quaternion(gt_q_arr[0, :])
        q_0 = q_gt_0 * q_est_0.conj()
        R = q_0.unit().R
        t = p_gt_0 - np.dot(R, p_es_0)

        return R, t

    @staticmethod
    def align_SE3(est_p_arr, gt_p_arr, est_q_arr, gt_q_arr, n_aligned=-1):
        """
            Calculate SE3 transformation R and t so that:
            t_gt = R_gt_est * t_est + t_gt_est_in_gt
            R_gt = R_gt_est * R_est

        Input:
        est_p_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        gt_p_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        est_q_arr -- estimated cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        gt_q_arr -- ground-truth cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        n_aligned -- if 1, only the first matched pair will be used, otherwise align_Umeyama() with the a certain amount
                     will be called.

        Output:
        R -- rotation matrix (3x3)     (R_gt_est)
        t -- translation vector (3x1)  (t_gt_est_in_gt)
        """
        if n_aligned == 1:
            R, t = SpatialAlignement.align_SE3_single(est_p_arr, gt_p_arr, est_q_arr, gt_q_arr)
            return R, t
        else:
            idxs = SpatialAlignement.get_indices(n_aligned, est_p_arr.shape[0])
            est_pos = est_p_arr[idxs, 0:3]
            gt_pos = gt_p_arr[idxs, 0:3]
            s, R, t = SpatialAlignement.align_Umeyama(gt_pos, est_pos,
                                                      known_scale=True)  # note the order
            t = np.array(t)
            t = t.reshape((3,))
            R = np.array(R)
            return R, t

    @staticmethod
    def align_Umeyama(gt_pos_arr, est_pos_arr, known_scale=False, yaw_only=False):
        """Implementation of the paper: S. Umeyama, Least-Squares Estimation
        of Transformation Parameters Between Two Point Patterns,
        IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

        gt_pos_arr = s * R * est_pos_arr + t

        Input:
        gt_pos_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        est_pos_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type

        Output:
        s -- scale factor (scalar)
        R -- rotation matrix (3x3)     (R_gt_est)
        t -- translation vector (3x1)  (t_gt_est_in_gt)
        """

        # subtract mean
        mu_M = gt_pos_arr.mean(0)
        mu_D = est_pos_arr.mean(0)
        model_zerocentered = gt_pos_arr - mu_M
        data_zerocentered = est_pos_arr - mu_D
        n = np.shape(gt_pos_arr)[0]

        # correlation
        C = 1.0 / n * np.dot(model_zerocentered.transpose(), data_zerocentered)
        Sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
        U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
        D_svd = np.diag(D_svd)
        V_svd = np.transpose(V_svd)

        S = np.eye(3)
        if (np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0):
            S[2, 2] = -1

        if yaw_only:
            rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
            theta = SpatialAlignement.get_best_yaw(rot_C)
            R = SO3.Rz(theta)
        else:
            R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

        if known_scale:
            s = 1
        else:
            s = 1.0 / Sigma2 * np.trace(np.dot(D_svd, S))

        t = mu_M - s * np.dot(R, mu_D)

        return s, R, t

    @staticmethod
    def align_SIM3(est_p_arr, gt_p_arr, n_aligned=-1):
        """
        align by similarity transformation
        calculate s, R, t so that:
            gt = R * s * est + t

        Input:
        est_p_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        gt_p_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type

        Output:
        s -- scale factor (scalar)
        R -- rotation matrix (3x3)     (R_gt_est)
        t -- translation vector (3x1)  (t_gt_est_in_gt)
        """
        idxs = SpatialAlignement.get_indices(n_aligned, est_p_arr.shape[0])
        est_pos = est_p_arr[idxs, 0:3]
        gt_pos = gt_p_arr[idxs, 0:3]
        s, R, t = SpatialAlignement.align_Umeyama(gt_pos, est_pos)  # note the order
        return s, R, t
