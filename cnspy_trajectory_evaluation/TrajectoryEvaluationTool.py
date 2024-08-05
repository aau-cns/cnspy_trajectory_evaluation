#!/usr/bin/env python
# Software License Agreement (GNU GPLv3  License)
#
# Copyright (c) 2024, Roland Jung (roland.jung@aau.at) , AAU, KPK, NAV
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
# BASED ON: https://github.com/aau-cns/cnspy_relative_pose_evaluation
# just install "pip install cnspy_relative_pose_evaluation"
########################################################################################################################

import rosbag
import time
import os
import argparse
import yaml

from cnspy_spatial_csv_formats.CSVSpatialFormatType import CSVSpatialFormatType
from cnspy_spatial_csv_formats.EstimationErrorType import EstimationErrorType
from cnspy_spatial_csv_formats.ErrorRepresentationType import ErrorRepresentationType
from cnspy_trajectory_evaluation.TrajectoryAlignmentTypes import TrajectoryAlignmentTypes
from cnspy_rosbag2csv.ROSbag2CSV import ROSbag2CSV
from cnspy_trajectory_evaluation.TrajectoryEvaluation import TrajectoryEvaluation


class TrajectoryEvaluationTool:
    @staticmethod
    def evaluate(bagfile_in, cfg,
                 result_dir=None,
                 subsample=0,
                 alignment_type=TrajectoryAlignmentTypes.se3,
                 num_aligned_samples=-1,
                 plot=False,
                 save_plot=False,
                 show_plot=False,
                 est_err_type=None,
                 rot_err_rep=None,
                 max_difference=0.01,
                 relative_timestamps=True,
                 fmt=CSVSpatialFormatType.PoseStamped,
                 IDs=None,
                 verbose=False):

        if not os.path.isfile(bagfile_in):
            print("TrajectoryEvaluationTool: could not find file: %s" % bagfile_in)
            return False


        ## create result dir:
        if result_dir == "" or result_dir is None:
            folder = str(bagfile_in).replace(".bag", "")
        else:
            folder = result_dir

        folder = os.path.abspath(folder)
        try:  # else already exists
            os.makedirs(folder)
        except:
            pass
        result_dir = folder
        if verbose:
            print("* result_dir: \t " + str(folder))

        dict_cfg = None
        with open(cfg, "r") as yamlfile:
            dict_cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            if "sensor_topics" not in dict_cfg:
                print("[sensor_topics] does not exist in fn=" + cfg)
                return False
            if "true_pose_topics" not in dict_cfg:
                print("[true_pose_topics] does not exist in fn=" + cfg)
                return False
            print("Successfully read YAML config file.")

        Sensor_ID_arr = []
        topic_list = []
        for key, val in dict_cfg["sensor_topics"].items():
            topic_list.append(val)
        for key, val in dict_cfg["true_pose_topics"].items():
            topic_list.append(val)
            Sensor_ID_arr.append(int(key))

        if IDs:
            Sensor_ID_arr = list(IDs)

        if verbose:
            print("* topic_list= " + str(topic_list))
            print("* Sensor_ID_arr= " + str(Sensor_ID_arr))


        #topic_list=['/d01/ranging', '/a01/ranging', '/a02/ranging', '/a03/ranging']



        for ID in Sensor_ID_arr:
            prefix = 'ID' + str(ID)
            res_dir = result_dir + '/' + prefix + '/'
            fn_est = str(res_dir + '/meas-pose.csv')
            fn_gt = str(res_dir + '/true-poses.csv')

            topic_list = list([dict_cfg["true_pose_topics"][ID], dict_cfg["sensor_topics"][ID]])
            fn_list = list([fn_gt, fn_est])

            # 1) extract all measurements to CSV
            if not os.path.isfile(fn_est):
                ROSbag2CSV.extract(bagfile_name=bagfile_in,
                                   result_dir=res_dir,
                                   topic_list=topic_list,
                                   fn_list=fn_list,
                                   verbose=verbose,
                                   fmt=fmt)


            # 4) evaluate
            eval = TrajectoryEvaluation(fn_gt=fn_gt, fn_est=fn_est, result_dir=res_dir,
                                        prefix=prefix,
                                        alignment_type=alignment_type,
                                        num_aligned_samples=num_aligned_samples,
                                        max_difference=max_difference,
                                        subsample=subsample,
                                        est_err_type=est_err_type,
                                        rot_err_rep=rot_err_rep,
                                        plot=plot,
                                        save_plot=save_plot,
                                        show_plot=show_plot,
                                        relative_timestamps=relative_timestamps,
                                        verbose=verbose)


        pass  # DONE

def main():
    # TrajectoryEvaluationTool.py --bagfile  ./test/sample_data/T1_A3_loiter_2m_2023-08-31-20-58-20.bag --cfg ./test/sample_data/config.yaml
    parser = argparse.ArgumentParser(
        description='TrajectoryEvaluationTool: evaluating a set of measured poses in a ROS1 bag file')
    parser.add_argument('--result_dir', help='directory to store results [otherwise bagfile name will be a directory]',
                        default='')
    parser.add_argument('--bagfile', help='input ROS1 bag file', default="not specified!", required=True)
    parser.add_argument('--cfg',
                        help='YAML configuration file describing the setup: {sensor_topics, true_pose_topics}',
                        default="config.yaml", required=True)
    parser.add_argument('--alignment_type', help='Estimation error type', choices=TrajectoryAlignmentTypes.list(),
                        default=str(TrajectoryAlignmentTypes.none))
    parser.add_argument('--num_aligned_samples',help='number of aligned sampled', default=0)
    parser.add_argument('--max_timestamp_difference', help='Max difference between associated timestampes (t_gt - t_est)', default=0.01)
    parser.add_argument('--subsample', help='subsampling factor for input data (CSV)', default=0)
    parser.add_argument('--est_err_type', help='Estimation error type', choices=EstimationErrorType.list(),
                        default=str(EstimationErrorType.none))
    parser.add_argument('--rot_err_rep', help='Rotation Error representation type', choices=ErrorRepresentationType.list(),
                        default=str(ErrorRepresentationType.none))
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--save_plot', action='store_true', default=False)
    parser.add_argument('--show_plot', action='store_true', default=False)
    parser.add_argument('--relative_timestamp', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)


    tp_start = time.time()
    args = parser.parse_args()

    TrajectoryEvaluationTool.evaluate(bagfile_in=args.bagfile,
                                      cfg=args.cfg,
                                      result_dir=args.result_dir,
                                      alignment_type=TrajectoryAlignmentTypes(args.alignment_type),
                                      num_aligned_samples=int(args.num_aligned_samples),
                                      max_difference=args.max_timestamp_difference,
                                      subsample=int(args.subsample),
                                      est_err_type=EstimationErrorType(args.est_err_type),
                                      rot_err_rep=ErrorRepresentationType(args.rot_err_rep),
                                      plot=args.plot,
                                      save_plot=args.save_plot,
                                      show_plot=args.show_plot,
                                      relative_timestamps=args.relative_timestamp,
                                      verbose=args.verbose)
    pass
    print(" ")
    print("finished after [%s sec]\n" % str(time.time() - tp_start))
    pass


if __name__ == "__main__":
    main()
    pass