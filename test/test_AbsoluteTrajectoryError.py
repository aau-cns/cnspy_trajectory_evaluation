
########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
from TrajectoryEvaluation.AbsoluteTrajectoryError import *
from trajectory.TrajectoryPlotter import TrajectoryPlotter, TrajectoryPlotConfig
from trajectory.SpatialConverter import SpatialConverter
from spatialmath import SO3

class AbsoluteTrajectoryError_Test(unittest.TestCase):

    def get_trajectories(self):
        traj_est = TrajectoryEstimated()
        self.assertTrue(traj_est.load_from_CSV('./sample_data/ID1-pose-est-cov.csv'))
        traj_gt = Trajectory()
        self.assertTrue(traj_gt.load_from_CSV('./sample_data/ID1-pose-gt.csv'))
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

        traj1 = Trajectory(t_vec = t_vec, q_vec=q_vec, p_vec=p_vec)
        ATE = AbsoluteTrajectoryError(traj1, traj1)
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)

        q_vec2 = np.zeros((4, 6), dtype=float)
        q_vec2[:, 0] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([11, 20, 45], unit='deg', order='xyz'))
        q_vec2[:, 1] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([12, 25, 45], unit='deg', order='xyz'))
        q_vec2[:, 2] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([13, 30, 45], unit='deg', order='xyz'))
        q_vec2[:, 3] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 37, 45], unit='deg', order='xyz'))
        q_vec2[:, 4] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 42, 45], unit='deg', order='xyz'))
        q_vec2[:, 5] = SpatialConverter.SO3_to_HTMQ_quaternion(SO3.RPY([10, 45, 45], unit='deg', order='xyz'))
        q_vec2 = q_vec2.T

        #q_vec2 = np.copy(q_vec)
        p_vec2 = np.copy(p_vec)
        p_vec2[:, 0] = p_vec2[:, 0] * 1.5
        traj2 = Trajectory(t_vec=t_vec, q_vec=q_vec2, p_vec=p_vec2)
        ATE2 = AbsoluteTrajectoryError(traj_gt=traj1, traj_est=traj2)
        ATE2.plot_pose_err(cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)


    def test_ATE(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est, traj_gt)
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        print('ATE1 done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_dist),
            angles=True)
        print('ATE1 done')

    def test_ATE2(self):
        traj_est, traj_gt = self.get_trajectories()

        ATE = AbsoluteTrajectoryError(traj_est, traj_gt)
        ATE.plot_p_err()
        ATE.plot_p_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        ATE.plot_rpy_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t))
        print('ATE2 done')
        ATE.plot_pose_err(cfg=TrajectoryPlotConfig(show=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
                          angles=True)
        ATE.plot_pose_err(
            cfg=TrajectoryPlotConfig(show=True, radians=False, plot_type=TrajectoryPlotTypes.plot_2D_over_t),
            angles=True)
        print('ATE2 done:ARMSE p={:.2f}, q={:.2f}'.format(ATE.ARMSE_p, ATE.ARMSE_q_deg))


if __name__ == "__main__":
    unittest.main()
