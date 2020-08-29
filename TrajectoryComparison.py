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

    matches_est_gt = None  # list of tuples containing [(idx_est, idx_gt), ...]

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

        self.matches_est_gt = zip(idx_est, idx_gt)
        # using zip() and * operator to
        # perform Unzipping
        # res = list(zip(*test_list))

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
    def trajectory_aligment(traj_est, traj_gt, method='sim3', num_frames=-1):
        '''
        calculate s, R, t so that:
            gt = R * s * est + t
        method can be: sim3, se3, posyaw, none;
        n_aligned: -1 means using all the frames
        '''
        assert (isinstance(traj_est, Trajectory))
        assert (isinstance(traj_gt, Trajectory))

        p_es = traj_est.p_vec
        q_es = traj_est.q_vec
        p_gt = traj_gt.p_vec
        q_gt = traj_gt.q_vec

        p_es, p_gt, q_es, q_gt
        assert p_es.shape[1] == 3
        assert p_gt.shape[1] == 3
        assert q_es.shape[1] == 4
        assert q_gt.shape[1] == 4
        assert p_es.shape[0] == p_gt.shape[0]
        assert q_es.shape[0] == q_gt.shape[0]

        s = 1
        R = np.identity(3)
        t = np.zeros((3,))
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

    def __init__(self, associated, alignment_type=TrajectoryAlignmentTypes.sim3, num_frames=-1):
        assert (isinstance(associated, AssociatedTrajectories))

        self.traj_gt_matched = Trajectory(df=associated.data_frame_gt_matched)
        self.traj_est_matched_aligned = Trajectory(df=associated.data_frame_est_matched)

        s, R, t = TrajectoryAlignmentTypes.trajectory_aligment(self.traj_est_matched_aligned, self.traj_gt_matched,
                                                               method=alignment_type,
                                                               num_frames=num_frames)

        self.traj_est_matched_aligned.transform_p(scale=s, t=t, R=R)

    def save_to_CSV(self, save_dir='.'):
        self.traj_est_matched_aligned.save_to_CSV(os.path.join(save_dir, 'traj_est_matched_aligned.csv'))
        self.traj_gt_matched.save_to_CSV(os.path.join(save_dir, 'traj_gt_matched_aligned.csv'))


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time
import csv
from TrajectoryPlotter import TrajectoryPlotter, TrajectoryPlotConfig


class TrajectoryAssociator_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print "Process time: " + str((time.time() - self.start_time))

    def get_associated(self):
        fn_gt_csv = "/home/jungr/workspace/github/rpg_trajectory_evaluation/results/euroc_mono_stereo/laptop/vio_mono/laptop_vio_mono_MH_01/stamped_groundtruth.txt"
        fn_est_csv = "/home/jungr/workspace/github/rpg_trajectory_evaluation/results/euroc_mono_stereo/laptop/vio_mono/laptop_vio_mono_MH_01/stamped_traj_estimate.txt"
        return AssociatedTrajectories(fn_est=fn_est_csv, fn_gt=fn_gt_csv)

    def test_init(self):
        self.start()
        associated = self.get_associated()
        self.stop()

    def test_align_trajectories(self):
        associated = self.get_associated()
        aligned = AlignedTrajectories(associated=associated)
        aligned.save_to_CSV(save_dir='./results/')
        traj_est_matched = Trajectory(df=associated.data_frame_est_matched)
        plot_gt = TrajectoryPlotter(traj_obj=aligned.traj_gt_matched)
        plot_est = TrajectoryPlotter(traj_obj=traj_est_matched)
        plot_est_aligned = TrajectoryPlotter(traj_obj=aligned.traj_est_matched_aligned)

        TrajectoryPlotter.multi_plot_3D(traj_plotter_list=[plot_gt, plot_est, plot_est_aligned],
                                        cfg=TrajectoryPlotConfig(),
                                        name_list=['gt_matched', 'est_matched', 'est_matched_aligned'])


if __name__ == "__main__":
    unittest.main()
