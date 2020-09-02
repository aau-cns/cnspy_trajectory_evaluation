import os

from csv2dataframe.CSV2DataFrame import CSV2DataFrame
from ros_csv_formats.CSVFormat import CSVFormat
from timestamp_association.TimestampAssociation import TimestampAssociation
from trajectory.Trajectory import Trajectory
from trajectory.TrajectoryEstimated import TrajectoryEstimated


class AssociatedTrajectories:
    csv_df_gt = None
    csv_df_est = None

    data_frame_gt_matched = None
    data_frame_est_matched = None

    matches_est_gt = None  # list of tuples containing [(idx_est, idx_gt), ...]

    def __init__(self, fn_gt, fn_est):
        assert (os.path.exists(fn_gt))
        assert (os.path.exists((fn_est)))

        self.csv_df_gt = CSV2DataFrame(filename=fn_gt)
        assert (self.csv_df_gt.data_loaded)
        self.csv_df_est = CSV2DataFrame(filename=fn_est)
        assert (self.csv_df_est.data_loaded)

        t_vec_gt = self.csv_df_gt.data_frame.as_matrix(['t'])
        t_vec_est = self.csv_df_est.data_frame.as_matrix(['t'])
        idx_est, idx_gt, t_est_matched, t_gt_matched = TimestampAssociation.associate_timestamps(
            t_vec_est,
            t_vec_gt)

        self.data_frame_est_matched = self.csv_df_est.data_frame.loc[idx_est, :]
        self.data_frame_gt_matched = self.csv_df_gt.data_frame.loc[idx_gt, :]

        self.matches_est_gt = zip(idx_est, idx_gt)
        # using zip() and * operator to
        # perform Unzipping
        # res = list(zip(*test_list))

    def save(self, result_dir=None, prefix=None):
        if not result_dir:
            fn_est_ = str(os.path.splitext(self.csv_df_est.fn)[0]) + "_matched.csv"
            fn_gt_ = str(os.path.splitext(self.csv_df_gt.fn)[0]) + "_matched.csv"
        else:
            if not prefix:
                prefix = ""
            if not os.path.exists(result_dir):
                os.makedirs(os.path.abspath(result_dir))
            fn_est_ = str(result_dir) + '/' + str(prefix) + "est_matched.csv"
            fn_gt_ = str(result_dir) + '/' + str(prefix) + "gt_matched.csv"

        CSV2DataFrame.save_CSV(self.data_frame_est_matched, filename=fn_est_, fmt=self.csv_df_est.format)
        CSV2DataFrame.save_CSV(self.data_frame_gt_matched, filename=fn_gt_, fmt=self.csv_df_gt.format)

    def get_trajectories(self):
        # returns Tr_est_matched, Tr_gt_matched
        if self.csv_df_est.format == CSVFormat.PoseWithCov:
            return TrajectoryEstimated(df=self.data_frame_est_matched), Trajectory(df=self.data_frame_gt_matched)
        else:
            return Trajectory(df=self.data_frame_est_matched), Trajectory(df=self.data_frame_gt_matched)
