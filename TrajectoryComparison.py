import os
import numpy as np
from enum import Enum

from Trajectory import Trajectory
from tum_eval.TUMCSV2DataFrame import TUMCSV2DataFrame
from timestamp_association.TimestampAssociation import TimestampAssociation
import transformations as tf


class AssociatedTrajectories:
    data_frame_gt = None
    data_frame_est = None

    data_frame_gt_matched = None
    data_frame_est_matched = None

    def __init__(self, fn_gt, fn_est):
        assert (os.path.exists(fn_gt))
        assert (os.path.exists((fn_est)))

        self.data_frame_gt = TUMCSV2DataFrame.load_TUM_CSV(filename=fn_gt)
        self.data_frame_est = TUMCSV2DataFrame.load_TUM_CSV(filename=fn_est)
        t_vec_gt = self.data_frame_gt.as_matrix(['t'])
        t_vec_est = self.data_frame_est.as_matrix(['t'])
        idx_est, idx_gt, t_est_matched, t_gt_matched = TimestampAssociation.associate_timestamps(
            t_vec_est,
            t_vec_gt)

        self.data_frame_est_matched = self.data_frame_est.loc[idx_est, :]
        self.data_frame_gt_matched = self.data_frame_gt.loc[idx_gt, :]

        TUMCSV2DataFrame.save_TUM_CSV(self.data_frame_est_matched,
                                      filename=str(os.path.splitext(fn_gt)[0]) + "_matched.csv")
        TUMCSV2DataFrame.save_TUM_CSV(self.data_frame_gt_matched,
                                      filename=str(os.path.splitext(fn_gt)[0]) + "_matched.csv")

    def get_trajectories(self):
        # returns Tr_est_matched, Tr_gt_matched
        return Trajectory(df=self.data_frame_est_matched), Trajectory(df=self.data_frame_gt_matched)


class TrajectyAlignmentTypes(Enum):
    sim3 = 'sim3'
    se3 = 'se3'
    posyaw = 'posyaw'
    none = 'none'

    def __str__(self):
        return self.value


class AlignedTrajectories:
    traj_est_matched_aligned = None
    traj_gt_matched = None

    def __init__(self, associated, alignment_type=TrajectyAlignmentTypes.sim3):
        pass

    @staticmethod
    def get_best_yaw(C):
        '''
        maximize trace(Rz(theta) * C)
        '''
        assert C.shape == (3, 3)

        A = C[0, 1] - C[1, 0]
        B = C[0, 0] + C[1, 1]
        theta = np.pi / 2 - np.arctan2(B, A)

        return theta

    @staticmethod
    def rot_z(theta):
        R = tf.rotation_matrix(theta, [0, 0, 1])
        R = R[0:3, 0:3]

        return R

    @staticmethod
    def get_indices(n_aligned, total_n):
        if n_aligned == -1:
            idxs = np.arange(0, total_n)
        else:
            assert n_aligned <= total_n and n_aligned >= 1
            idxs = np.arange(0, n_aligned)
        return idxs

    @staticmethod
    def align_position_yaw_single(p_es, p_gt, q_es, q_gt):
        '''
        calcualte the 4DOF transformation: yaw R and translation t so that:
            gt = R * est + t
        '''

        p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
        p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]
        g_rot = tf.quaternion_matrix(q_gt_0)
        g_rot = g_rot[0:3, 0:3]
        est_rot = tf.quaternion_matrix(q_es_0)
        est_rot = est_rot[0:3, 0:3]

        C_R = np.dot(est_rot, g_rot.transpose())
        theta = AlignedTrajectories.get_best_yaw(C_R)
        R = AlignedTrajectories.rot_z(theta)
        t = p_gt_0 - np.dot(R, p_es_0)

        return R, t

    @staticmethod
    def align_position_yaw(p_es, p_gt, q_es, q_gt, n_aligned=1):
        if n_aligned == 1:
            R, t = AlignedTrajectories.align_position_yaw_single(p_es, p_gt, q_es, q_gt)
            return R, t
        else:
            idxs = AlignedTrajectories.get_indices(n_aligned, p_es.shape[0])
            est_pos = p_es[idxs, 0:3]
            gt_pos = p_gt[idxs, 0:3]
            _, R, t = AlignedTrajectories.align_umeyama(gt_pos, est_pos, known_scale=True,
                                                        yaw_only=True)  # note the order
            t = np.array(t)
            t = t.reshape((3,))
            R = np.array(R)
            return R, t

    @staticmethod
    def align_SE3_single(p_es, p_gt, q_es, q_gt):
        '''
        Calculate SE3 transformation R and t so that:
            gt = R * est + t
        Using only the first poses of est and gt
        '''

        p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
        p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]

        g_rot = tf.quaternion_matrix(q_gt_0)
        g_rot = g_rot[0:3, 0:3]
        est_rot = tf.quaternion_matrix(q_es_0)
        est_rot = est_rot[0:3, 0:3]

        R = np.dot(g_rot, np.transpose(est_rot))
        t = p_gt_0 - np.dot(R, p_es_0)

        return R, t

    @staticmethod
    def align_SE3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
        '''
        Calculate SE3 transformation R and t so that:
            gt = R * est + t
        '''
        if n_aligned == 1:
            R, t = AlignedTrajectories.align_SE3_single(p_es, p_gt, q_es, q_gt)
            return R, t
        else:
            idxs = AlignedTrajectories.get_indices(n_aligned, p_es.shape[0])
            est_pos = p_es[idxs, 0:3]
            gt_pos = p_gt[idxs, 0:3]
            s, R, t = AlignedTrajectories.align_umeyama(gt_pos, est_pos,
                                                        known_scale=True)  # note the order
            t = np.array(t)
            t = t.reshape((3,))
            R = np.array(R)
            return R, t

    @staticmethod
    def align_SIM3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
        '''
        align by similarity transformation
        calculate s, R, t so that:
            gt = R * s * est + t
        '''
        idxs = AlignedTrajectories.get_indices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        s, R, t = AlignedTrajectories.align_umeyama(gt_pos, est_pos)  # note the order
        return s, R, t

    @staticmethod
    def align_trajectories(p_es, p_gt, q_es, q_gt, method, num_frames=-1):
        '''
        calculate s, R, t so that:
            gt = R * s * est + t
        method can be: sim3, se3, posyaw, none;
        n_aligned: -1 means using all the frames
        '''
        assert p_es.shape[1] == 3
        assert p_gt.shape[1] == 3
        assert q_es.shape[1] == 4
        assert q_gt.shape[1] == 4

        s = 1
        R = None
        t = None
        if method == TrajectyAlignmentTypes.sim3:
            assert num_frames >= 2 or num_frames == -1, "sim3 uses at least 2 frames"
            s, R, t = AlignedTrajectories.align_SIM3(p_es, p_gt, q_es, q_gt, num_frames)
        elif method == TrajectyAlignmentTypes.se3:
            R, t = AlignedTrajectories.align_SE3(p_es, p_gt, q_es, q_gt, num_frames)
        elif method == TrajectyAlignmentTypes.posyaw:
            R, t = AlignedTrajectories.align_position_yaw(p_es, p_gt, q_es, q_gt, num_frames)
        elif method == TrajectyAlignmentTypes.none:
            R = np.identity(3)
            t = np.zeros((3,))
        else:
            assert False, 'unknown alignment method'

        return s, R, t

    @staticmethod
    def align_umeyama(model, data, known_scale=False, yaw_only=False):
        """Implementation of the paper: S. Umeyama, Least-Squares Estimation
        of Transformation Parameters Between Two Point Patterns,
        IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

        model = s * R * data + t

        Input:
        model -- first trajectory (nx3), numpy array type
        data -- second trajectory (nx3), numpy array type

        Output:
        s -- scale factor (scalar)
        R -- rotation matrix (3x3)
        t -- translation vector (3x1)
        t_error -- translational error per point (1xn)

        """

        # substract mean
        mu_M = model.mean(0)
        mu_D = data.mean(0)
        model_zerocentered = model - mu_M
        data_zerocentered = data - mu_D
        n = np.shape(model)[0]

        # correlation
        C = 1.0 / n * np.dot(model_zerocentered.transpose(), data_zerocentered)
        sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
        U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
        D_svd = np.diag(D_svd)
        V_svd = np.transpose(V_svd)

        S = np.eye(3)
        if (np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0):
            S[2, 2] = -1

        if yaw_only:
            rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
            theta = AlignedTrajectories.get_best_yaw(rot_C)
            R = AlignedTrajectories.rot_z(theta)
        else:
            R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

        if known_scale:
            s = 1
        else:
            s = 1.0 / sigma2 * np.trace(np.dot(D_svd, S))

        t = mu_M - s * np.dot(R, mu_D)

        return s, R, t


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time
import csv


class TrajectoryAssociator_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print "Process time: " + str((time.time() - self.start_time))

    def test_init(self):
        fn_gt_csv = "/home/jungr/workspace/github/rpg_trajectory_evaluation/results/euroc_mono_stereo/laptop/vio_mono/laptop_vio_mono_MH_01/stamped_groundtruth.txt"
        fn_est_csv = "/home/jungr/workspace/github/rpg_trajectory_evaluation/results/euroc_mono_stereo/laptop/vio_mono/laptop_vio_mono_MH_01/stamped_traj_estimate.txt"
        self.start()
        associated = AssociatedTrajectories(fn_est=fn_est_csv, fn_gt=fn_gt_csv)
        self.stop()


if __name__ == "__main__":
    unittest.main()
