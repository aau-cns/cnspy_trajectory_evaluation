import os
from TrajectoryAlignmentTypes import TrajectoryAlignmentTypes
from AlignedTrajectories import AlignedTrajectories
from AssociatedTrajectories import AssociatedTrajectories
from AbsoluteTrajectoryError import AbsoluteTrajectoryError
from TrajectoryNEES import TrajectoryNEES
from EvaluationReport import EvaluationReport


class TrajectoryEvaluation:
    report = None

    def __init__(self, fn_gt, fn_est, result_dir=None, prefix=None,
                 alignment_type=TrajectoryAlignmentTypes.se3, num_aligned_samples=-1):
        if not result_dir:
            result_dir = '.'
        if not prefix:
            prefix = ''

        self.report = EvaluationReport(directory=os.path.abspath(result_dir), fn_gt=os.path.abspath(fn_gt),
                                       fn_est=os.path.abspath(fn_est),
                                       alignment=str(alignment_type), num_aligned_samples=num_aligned_samples)
        assoc = AssociatedTrajectories(fn_gt=fn_gt, fn_est=fn_est)
        assoc.save(result_dir=result_dir, prefix=prefix)

        aligned = AlignedTrajectories(associated=assoc, alignment_type=alignment_type, num_frames=num_aligned_samples)
        aligned.save(result_dir=result_dir, prefix=prefix)

        ATE = AbsoluteTrajectoryError(traj_est=aligned.traj_est_matched_aligned, traj_gt=aligned.traj_gt_matched)
        self.report.ARMSE_p = ATE.ARMSE_p
        self.report.ARMSE_q = ATE.ARMSE_q_deg
        ATE.traj_err.save_to_CSV(result_dir + '/' + prefix + 'err_matched_aligned.csv')

        NEES = TrajectoryNEES(traj_est=aligned.traj_est_matched_aligned, traj_err=ATE.traj_err)
        self.report.ANEES_p = NEES.ANEES_p
        self.report.ANEES_q = NEES.ANEES_q
        NEES.save_to_CSV(result_dir + '/' + prefix + 'nees_matched_aligned.csv')

        self.report.save(result_dir + '/' + prefix + 'report.ini')


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import time


class TrajectoryEvaluation_Test(unittest.TestCase):
    start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, info="Process time"):
        print str(info) + " took : " + str((time.time() - self.start_time)) + " [sec]"

    def get_fn(self):
        fn_gt_csv = "../sample_data/ID1-pose-gt.csv"
        fn_est_csv = "../sample_data/ID1-pose-est-cov.csv"
        return fn_gt_csv, fn_est_csv

    def test_init(self):
        self.start()
        fn_gt_csv = "../sample_data/ID1-pose-gt.csv"
        fn_est_csv = "../sample_data/ID1-pose-est-cov.csv"
        eval = TrajectoryEvaluation(fn_gt_csv, fn_est_csv, result_dir='../sample_data/result/eval', prefix='eval-ID1-',
                                    alignment_type=TrajectoryAlignmentTypes.none)

        self.stop()


if __name__ == "__main__":
    unittest.main()
