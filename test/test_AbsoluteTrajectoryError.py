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
########################################################################################################################
import os
import unittest
from cnspy_spatial_csv_formats.CSVSpatialFormat import CSVSpatialFormat, EstimationErrorType, ErrorRepresentationType, CSVSpatialFormatType
from cnspy_trajectory_evaluation.AbsoluteTrajectoryError import *
from cnspy_trajectory.TrajectoryPlotter import TrajectoryPlotter, TrajectoryPlotConfig, TrajectoryPlotTypes
from cnspy_trajectory.TrajectoryErrorType import TrajectoryErrorType
from cnspy_trajectory.SpatialConverter import SpatialConverter
from spatialmath import SO3

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

class AbsoluteTrajectoryError_Test(unittest.TestCase):

    def get_trajectories(self):
        traj_est = TrajectoryEstimated(fmt=CSVSpatialFormat(est_err_type=EstimationErrorType.type5,
                                                            err_rep_type=ErrorRepresentationType.theta_R,
                                                            fmt_type=CSVSpatialFormatType.PosOrientWithCov))
        self.assertTrue(traj_est.load_from_CSV(str(SAMPLE_DATA_DIR + '/ID1-pose-est-posorient-cov.csv')))
        traj_gt = Trajectory()
        self.assertTrue(traj_gt.load_from_CSV(str(SAMPLE_DATA_DIR + '/ID1-pose-gt.csv')))
        return traj_est, traj_gt

    def test_ATE0(self):
        t_vec= np.array([[1, 2, 3, 4, 5, 6]])
        p_vec = np.array([[1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.]], dtype=float)

        t_vec = t_vec.T

        p_vec[1, :] = p_vec[1, :]*0.2
        p_vec[2, :] = p_vec[2, :]*0.3
        p_vec[0, :] = p_vec[0, :]*0.1
        p_vec = (p_vec.T)

        q_vec = np.zeros((4, 6), dtype=float)
        q_vec[:, 0] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 20, 45], unit='deg', order='xyz'))
        q_vec[:, 1] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 25, 45], unit='deg', order='xyz'))
        q_vec[:, 2] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 30, 45], unit='deg', order='xyz'))
        q_vec[:, 3] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 35, 45], unit='deg', order='xyz'))
        q_vec[:, 4] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 40, 45], unit='deg', order='xyz'))
        q_vec[:, 5] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 45, 45], unit='deg', order='xyz'))
        q_vec = q_vec.T

        traj_gt = Trajectory(t_vec=t_vec, q_vec=q_vec, p_vec=p_vec)
        traj_est_fmt = CSVSpatialFormat(est_err_type=EstimationErrorType.type5,
                                        err_rep_type=ErrorRepresentationType.theta_R,
                                        fmt_type=CSVSpatialFormatType.TUM)
        traj_est = TrajectoryEstimated(t_vec=t_vec, q_vec=q_vec, p_vec=p_vec, fmt=traj_est_fmt)

        ATE = AbsoluteTrajectoryError(traj_gt=traj_gt, traj_est=traj_est)
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=True, radians=False,
                                                   plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)

        q_vec2 = np.zeros((4, 6), dtype=float)
        q_vec2[:, 0] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10+90, 20, 45], unit='deg', order='xyz'))
        q_vec2[:, 1] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10-90, 25, 45], unit='deg', order='xyz'))
        q_vec2[:, 2] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 30, 45], unit='deg', order='xyz'))
        q_vec2[:, 3] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 35+90, 45], unit='deg', order='xyz'))
        q_vec2[:, 4] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 40-90, 45], unit='deg', order='xyz'))
        q_vec2[:, 5] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 45, 45+90], unit='deg', order='xyz'))
        q_vec2 = q_vec2.T

        #q_vec2 = np.copy(q_vec)
        p_vec2 = np.copy(p_vec)
        p_vec2[:, 0] = p_vec2[:, 0] * 1.5
        traj_est = TrajectoryEstimated(t_vec=t_vec, q_vec=q_vec2, p_vec=p_vec2, fmt=traj_est_fmt)
        ATE2_global = AbsoluteTrajectoryError(traj_gt=traj_gt, traj_est=traj_est,
                                              traj_err_type=TrajectoryErrorType.global_pose())
        ATE2_global.plot_pose_err(cfg=TrajectoryPlotConfig(show=True, radians=False,
                                                           plot_type=TrajectoryPlotTypes.plot_2D_over_t), angles=True)
        print('ATE2_global done:ARMSE p={:.2f}, q={:.2f}'.format(ATE2_global.ARMSE_p, ATE2_global.ARMSE_q_deg))

        ATE2_local = AbsoluteTrajectoryError(traj_gt=traj_gt, traj_est=traj_est,
                                             traj_err_type=TrajectoryErrorType.local_pose())
        ATE2_local.plot_pose_err(cfg=TrajectoryPlotConfig(show=True, radians=False,
                                                          plot_type=TrajectoryPlotTypes.plot_2D_over_t), angles=True)
        print('ATE2_local done:ARMSE p={:.2f}, q={:.2f}'.format(ATE2_local.ARMSE_p, ATE2_local.ARMSE_q_deg))

        ATE2_loc_glob = AbsoluteTrajectoryError(traj_gt=traj_gt, traj_est=traj_est,
                                                traj_err_type=TrajectoryErrorType.global_p_local_q())
        ATE2_loc_glob.plot_pose_err(cfg=TrajectoryPlotConfig(show=True, radians=False,
                                                             plot_type=TrajectoryPlotTypes.plot_2D_over_t), angles=True)
        print('ATE2_loc_glob done:ARMSE p={:.2f}, q={:.2f}'.format(ATE2_loc_glob.ARMSE_p, ATE2_loc_glob.ARMSE_q_deg))

    def test_ATE1_loc_global(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est=traj_est, traj_gt=traj_gt,
                                      traj_err_type=TrajectoryErrorType.global_p_local_q())
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        print('ATE1 done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
            angles=True)
        print('test_ATE1_loc_global done:ARMSE p={:.2f}, q={:.2f}'.format(ATE.ARMSE_p, ATE.ARMSE_q_deg))

    def test_ATE1_local_pose(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est=traj_est, traj_gt=traj_gt,
                                      traj_err_type=TrajectoryErrorType.local_pose())
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        print('ATE1 local pose done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
            angles=True)

        print('test_ATE1_local_pose done:ARMSE p={:.2f}, q={:.2f}'.format(ATE.ARMSE_p, ATE.ARMSE_q_deg))

    def test_ATE1_global_pose(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est=traj_est, traj_gt=traj_gt,
                                      traj_err_type=TrajectoryErrorType.global_pose())
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        print('ATE1 global pose done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
            angles=True)

        print('test_ATE1_global_pose done:ARMSE p={:.2f}, q={:.2f}'.format(ATE.ARMSE_p, ATE.ARMSE_q_deg))

    def test_ATE2(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est=traj_est, traj_gt=traj_gt)
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        print('ATE2 done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
            angles=True)
        print('test_ATE2 done:ARMSE p={:.2f}, q={:.2f}'.format(ATE.ARMSE_p, ATE.ARMSE_q_deg))

    def test_ATE_over_distance(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est=traj_est, traj_gt=traj_gt)
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        print('ATE2 done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist),
            angles=True)
        print('test_ATE2 done:ARMSE p={:.2f}, q={:.2f}'.format(ATE.ARMSE_p, ATE.ARMSE_q_deg))

if __name__ == "__main__":
    unittest.main()
