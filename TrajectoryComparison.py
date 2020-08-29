import os
import numpy as np
from enum import Enum

from Trajectory import Trajectory
from SpatialAlignment import SpatialAlignement
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


class TrajectoryAlignmentTypes(Enum):
    sim3 = 'sim3'
    se3 = 'se3'
    posyaw = 'posyaw'
    none = 'none'

    def __str__(self):
        return self.value

    @staticmethod
    def align_trajectories(p_es, p_gt, q_es, q_gt, method='sim3', num_frames=-1):
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
        if method == TrajectoryAlignmentTypes.sim3:
            assert num_frames >= 2 or num_frames == -1, "sim3 uses at least 2 frames"
            s, R, t = SpatialAlignement.align_SIM3(p_es, p_gt, num_frames)
        elif method == TrajectoryAlignmentTypes.se3:
            R, t = SpatialAlignement.align_SE3(p_es, p_gt, q_es, q_gt, num_frames)
        elif method == TrajectoryAlignmentTypes.posyaw:
            R, t = SpatialAlignement.align_position_yaw(p_es, p_gt, q_es, q_gt, num_frames)
        elif method == TrajectoryAlignmentTypes.none:
            R = np.identity(3)
            t = np.zeros((3,))
        else:
            assert False, 'unknown alignment method'

        return s, R, t


class AlignedTrajectories:
    traj_est_matched_aligned = None
    traj_gt_matched = None

    def __init__(self, associated, alignment_type=TrajectoryAlignmentTypes.sim3):
        pass


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
