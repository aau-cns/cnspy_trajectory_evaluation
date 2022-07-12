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
    def align_position_yaw_single(p_NB_in_N_arr, p_GB_in_B_arr, q_NB_arr, q_GB_arr):
        """
        Calculate the 4DOF transformation (Rz(yaw_GN) and translation t_GN_in_G)  between G (Global) and N (Navigation)
        reference frame assuming identical B (Body) so that:
          p_GB_in_G = R_GN * p_NB_in_N + p_GN_in_N
        Using only the first poses of est and gt

        Input:
        p_NB_in_N_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        p_GB_in_B_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        q_NB_arr -- estimated cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        q_GB_arr -- ground-truth cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type

        Output:
        R_GN -- rotation matrix (3x3)     (orientation from Global to Navigation)
        p_GN_in_G -- translation vector (3x1)  (position from Global to Navigation expressed in Global)
        """

        p_NB_in_N_0, q_NB_0 = p_NB_in_N_arr[0, :], q_NB_arr[0, :]
        p_GB_in_G_0, q_GB_0 = p_GB_in_B_arr[0, :], q_GB_arr[0, :]

        q_NB_0 = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_NB_0).unit()
        q_GB_0 = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_GB_0).unit()
        q_GN_0 = q_GB_0 * q_NB_0.conj()

        theta_GN = SpatialAlignement.get_best_yaw(q_GN_0.R)
        R_GN = SO3.Rz(theta_GN)

        # Convert to 3x3 np.array
        R_GN = np.array(R_GN)
        p_GN_in_G = p_GB_in_G_0 - np.dot(R_GN, p_NB_in_N_0)

        return R_GN, p_GN_in_G

    @staticmethod
    def align_position_yaw(p_NB_in_N_arr, p_GB_in_G_arr, q_NB_arr, q_GB_arr, n_aligned=1):
        """
        Calculate the 4DOF transformation (Rz(yaw_GN) and translation t_GN_in_G)  between G (Global) and N (Navigation)
        reference frame assuming identical B (Body) so that:
          p_GB_in_G = R_GN * p_NB_in_N + p_GN_in_N

        Input:
        p_NB_in_N_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        p_GB_in_B_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        q_NB_arr -- estimated cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        q_GB_arr -- ground-truth cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type

        Output:
        R_GN -- rotation matrix (3x3)     (orientation from Global to Navigation)
        p_GN_in_G -- translation vector (3x1)  (position from Global to Navigation expressed in Global)
        """

        if n_aligned == 1:
            R_GN, p_GN_in_G = SpatialAlignement.align_position_yaw_single(p_NB_in_N_arr, p_GB_in_G_arr, q_NB_arr, q_GB_arr)
            return R_GN, p_GN_in_G
        else:
            idxs = SpatialAlignement.get_indices(n_aligned, p_NB_in_N_arr.shape[0])
            p_NB_in_N = p_NB_in_N_arr[idxs, 0:3]
            p_GN_in_G = p_GB_in_G_arr[idxs, 0:3]
            _, R_GN, p_GN_in_G = SpatialAlignement.align_Umeyama(p_GN_in_G, p_NB_in_N, known_scale=True,
                                                      yaw_only=True)  # note the order
            p_GN_in_G = np.array(p_GN_in_G)
            p_GN_in_G = p_GN_in_G.reshape((3,))

            # Convert to 3x3 np.array
            R_GN = np.array(R_GN)
            return R_GN, p_GN_in_G

    @staticmethod
    def align_SE3_single(p_NB_in_N_arr, p_GB_in_G_arr, q_NB_arr, q_GB_arr):
        """
        Calculate SE3 transformation R and t between G (Global) and N (Navigation)
        reference frame assuming identical B (Body) so that:
          p_GB_in_G = R_GN * p_NB_in_N + p_GN_in_N
          R_GB = R_GN * R_NB
        Using only the first poses

        Input:
        p_NB_in_N_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        p_GB_in_B_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        q_NB_arr -- estimated cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        q_GB_arr -- ground-truth cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type

        Output:
        R_GN -- rotation matrix (3x3)     (orientation from Global to Navigation)
        p_GN_in_G -- translation vector (3x1)  (position from Global to Navigation expressed in Global)
        """

        p_NB_in_N_0 = p_NB_in_N_arr[0, :]
        p_GB_in_G_0 = p_GB_in_G_arr[0, :]

        q_NB_0 = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_NB_arr[0, :])
        q_GB_0 = SpatialConverter.HTMQ_quaternion_to_Quaternion(q_GB_arr[0, :])

        # R_G_N = R_G_B * R_N_B'
        q_GN_0 = q_GB_0 * q_NB_0.conj()
        R_GN = q_GN_0.unit().R

        # p_G_N_in_G = p_G_B_in_G - R_G_N * p_N_B_in_N
        p_GN_in_G = p_GB_in_G_0 - np.dot(R_GN, p_NB_in_N_0)

        # Convert to 3x3 np.array
        R_GN = np.array(R_GN)
        return R_GN, p_GN_in_G

    @staticmethod
    def align_SE3(p_NB_in_N_arr, p_GB_in_G_arr, q_NB_arr, q_GB_arr, n_aligned=-1):
        """
        Calculate SE3 transformation R and t between G (Global) and N (Navigation)
        reference frame assuming identical B (Body) so that:
          p_GB_in_G = R_GN * p_NB_in_N + p_GN_in_N
          R_GB = R_GN * R_NB
        Using N-poses pairs

        Input:
        p_NB_in_N_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        p_GB_in_B_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        q_NB_arr -- estimated cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type
        q_GB_arr -- ground-truth cnspy_trajectory attitude (nx4), quaternion [x, y, z, w] over n-time steps, numpy array type

        Output:
        R_GN -- rotation matrix (3x3)     (orientation from Global to Navigation)
        p_GN_in_G -- translation vector (3x1)  (position from Global to Navigation expressed in Global)
        """
        if n_aligned == 1:
            R_GN, p_GN_in_G = SpatialAlignement.align_SE3_single(p_NB_in_N_arr, p_GB_in_G_arr, q_NB_arr, q_GB_arr)
            return R_GN, p_GN_in_G
        else:
            idxs = SpatialAlignement.get_indices(n_aligned, p_NB_in_N_arr.shape[0])
            p_NB_in_N = p_NB_in_N_arr[idxs, 0:3]
            p_GB_in_G = p_GB_in_G_arr[idxs, 0:3]
            s, R_GN, p_GN_in_G = SpatialAlignement.align_Umeyama(p_GB_in_G_arr=p_GB_in_G, p_NB_in_N_arr=p_NB_in_N,
                                                                 known_scale=True)  # note the order
            p_GN_in_G = np.array(p_GN_in_G)
            p_GN_in_G = p_GN_in_G.reshape((3,))
            # Convert to 3x3 np.array
            R_GN = np.array(R_GN)
            return R_GN, p_GN_in_G

    @staticmethod
    def align_Umeyama(p_GB_in_G_arr, p_NB_in_N_arr, known_scale=False, yaw_only=False):
        """Implementation of the paper: S. Umeyama, Least-Squares Estimation
        of Transformation Parameters Between Two Point Patterns,
        IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.
        taken from: https://github.com/uzh-rpg/rpg_trajectory_evaluation/

        p_GB_in_G_arr = s_GN * R_GN * p_NB_in_N_arr + p_GN_in_G
        R_GB = R_GN * R_NB

        Input:
        p_GB_in_G_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        p_NB_in_N_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type

        Output:
        s_GN -- scale factor (scalar)     (scale from Global to Naviagiation)
        R_GN -- rotation matrix (3x3)     (orientation from Global to Navigation)
        p_GN_in_G -- translation vector (3x1)  (position from Global to Navigation expressed in Global)
        """

        # subtract mean
        mu_G = p_GB_in_G_arr.mean(0)  # model
        mu_N = p_NB_in_N_arr.mean(0)  # data
        # zero centered
        p_ZB_in_G = p_GB_in_G_arr - mu_G
        p_ZB_in_N = p_NB_in_N_arr - mu_N
        N = np.shape(p_GB_in_G_arr)[0]

        # correlation
        # C = 1/N * Sum((p_GB_in_G_arr - mu_GB_in_G_arr)*(p_NB_in_N_arr - mu_NB_in_N_arr)^\transpose)
        C = 1.0 / N * np.dot(p_ZB_in_G.transpose(), p_ZB_in_N)
        Sigma2_p_ZB_in_N = 1.0 / N * np.multiply(p_ZB_in_N, p_ZB_in_N).sum()

        # UDV' = svd(C)
        U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
        D_svd = np.diag(D_svd)
        V_svd = np.transpose(V_svd)

        W = np.eye(3)
        if (np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0):
            W[2, 2] = -1

        if yaw_only:
            rot_C = np.dot(p_ZB_in_N.transpose(), p_ZB_in_G)
            theta = SpatialAlignement.get_best_yaw(rot_C)
            R_GN = SO3.Rz(theta)
        else:
            R_GN = np.dot(U_svd, np.dot(W, np.transpose(V_svd)))

        if known_scale:
            s_GN = 1
        else:
            s_GN = 1.0 / Sigma2_p_ZB_in_N * np.trace(np.dot(D_svd, W))

        p_GN_in_G = mu_G - s_GN * np.dot(R_GN, mu_N)

        # Convert to 3x3 np.array
        R_GN = np.array(R_GN)
        return s_GN, R_GN, p_GN_in_G

    @staticmethod
    def align_SIM3(p_NB_in_N_arr, p_GB_in_G_arr, n_aligned=-1):
        """
        align by similarity transformation
        p_GB_in_G_arr = s_GN * R_GN * p_NB_in_N_arr + p_GN_in_G
        R_GB = R_GN * R_NB

        Input:
        p_GB_in_G_arr -- ground-truth cnspy_trajectory positions (nx3) over n-time steps, numpy array type
        p_NB_in_N_arr -- estimated cnspy_trajectory positions (nx3) over n-time steps, numpy array type

        Output:
        s_GN -- scale factor (scalar)     (scale from Global to Naviagiation)
        R_GN -- rotation matrix (3x3)     (orientation from Global to Navigation)
        p_GN_in_G -- translation vector (3x1)  (position from Global to Navigation expressed in Global)
        """
        idxs = SpatialAlignement.get_indices(n_aligned, p_NB_in_N_arr.shape[0])
        est_pos = p_NB_in_N_arr[idxs, 0:3]
        gt_pos = p_GB_in_G_arr[idxs, 0:3]
        s_GN, R_GN, p_GN_in_G = SpatialAlignement.align_Umeyama(p_GB_in_G_arr=gt_pos, p_NB_in_N_arr=est_pos)  # note the order
        return s_GN, R_GN, p_GN_in_G
