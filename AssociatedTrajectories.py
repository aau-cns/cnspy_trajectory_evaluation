import os

from csv2dataframe.TUMCSV2DataFrame import TUMCSV2DataFrame
from timestamp_association.TimestampAssociation import TimestampAssociation
from trajectory.Trajectory import Trajectory


class AssociatedTrajectories:
    data_frame_gt = None
    data_frame_est = None

    data_frame_gt_matched = None
    data_frame_est_matched = None

    matches_est_gt = None  # list of tuples containing [(idx_est, idx_gt), ...]

    def __init__(self, fn_gt, fn_est):
        assert (os.path.exists(fn_gt))
        assert (os.path.exists((fn_est)))

        self.data_frame_gt = TUMCSV2DataFrame.load_CSV(filename=fn_gt)
        self.data_frame_est = TUMCSV2DataFrame.load_CSV(filename=fn_est)
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

        TUMCSV2DataFrame.save_CSV(self.data_frame_est_matched,
                                  filename=str(os.path.splitext(fn_gt)[0]) + "_matched.csv")
        TUMCSV2DataFrame.save_CSV(self.data_frame_gt_matched,
                                  filename=str(os.path.splitext(fn_gt)[0]) + "_matched.csv")

    def get_trajectories(self):
        # returns Tr_est_matched, Tr_gt_matched
        return Trajectory(df=self.data_frame_est_matched), Trajectory(df=self.data_frame_gt_matched)
