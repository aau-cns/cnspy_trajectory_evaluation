import os

from trajectory.Trajectory import Trajectory
from trajectory_evaluation.AssociatedTrajectories import AssociatedTrajectories
from trajectory_evaluation.TrajectoryAlignmentTypes import TrajectoryAlignmentTypes


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

        self.traj_est_matched_aligned.transform(scale=s, t=t, R=R)

    def save_to_CSV(self, save_dir='.'):
        self.traj_est_matched_aligned.save_to_CSV(os.path.join(save_dir, 'traj_est_matched_aligned.csv'))
        self.traj_gt_matched.save_to_CSV(os.path.join(save_dir, 'traj_gt_matched_aligned.csv'))


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time
from trajectory.TrajectoryPlotter import TrajectoryPlotter
from trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig


class AlignedTrajectories_Test(unittest.TestCase):
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
