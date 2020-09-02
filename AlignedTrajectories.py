import os

from trajectory_evaluation.AssociatedTrajectories import AssociatedTrajectories
from trajectory_evaluation.TrajectoryAlignmentTypes import TrajectoryAlignmentTypes


class AlignedTrajectories:
    traj_est_matched_aligned = None
    traj_gt_matched = None

    def __init__(self, associated, alignment_type=TrajectoryAlignmentTypes.sim3, num_frames=-1):

        #        assert (isinstance(associated, AssociatedTrajectories))

        self.traj_est_matched_aligned, self.traj_gt_matched = associated.get_trajectories()

        s, R, t = TrajectoryAlignmentTypes.trajectory_aligment(self.traj_est_matched_aligned, self.traj_gt_matched,
                                                               method=alignment_type,
                                                               num_frames=num_frames)

        self.traj_est_matched_aligned.transform(scale=s, t=t, R=R)

    def save(self, result_dir='.', prefix=None):
        if not prefix:
            prefix = ""

        if not os.path.exists(result_dir):
            os.makedirs(os.path.abspath(result_dir))
        self.traj_est_matched_aligned.save_to_CSV(
            os.path.join(result_dir, str(prefix) + 'est_matched_aligned.csv'))
        self.traj_gt_matched.save_to_CSV(
            os.path.join(result_dir, str(prefix) + 'gt_matched_aligned.csv'))


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time
from trajectory.Trajectory import Trajectory
from trajectory.TrajectoryPlotter import TrajectoryPlotter
from trajectory.TrajectoryPlotConfig import TrajectoryPlotConfig


class AlignedTrajectories_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print "Process time: " + str((time.time() - self.start_time))

    def get_associated(self):
        fn_gt_csv = "../sample_data/ID1-pose-gt.csv"
        fn_est_csv = "../sample_data/ID1-pose-est-cov.csv"
        return AssociatedTrajectories(fn_est=fn_est_csv, fn_gt=fn_gt_csv)

    def test_init(self):
        self.start()
        associated = self.get_associated()
        self.stop()

    def test_align_trajectories(self):
        associated = self.get_associated()
        aligned = AlignedTrajectories(associated=associated)
        aligned.save(result_dir='./results/')
        traj_est_matched = Trajectory(df=associated.data_frame_est_matched)
        plot_gt = TrajectoryPlotter(traj_obj=aligned.traj_gt_matched)
        plot_est = TrajectoryPlotter(traj_obj=traj_est_matched)
        plot_est_aligned = TrajectoryPlotter(traj_obj=aligned.traj_est_matched_aligned)

        TrajectoryPlotter.multi_plot_3D(traj_plotter_list=[plot_gt, plot_est, plot_est_aligned],
                                        cfg=TrajectoryPlotConfig(),
                                        name_list=['gt_matched', 'est_matched', 'est_matched_aligned'])


if __name__ == "__main__":
    unittest.main()
