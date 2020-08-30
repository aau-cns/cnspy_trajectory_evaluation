import numpy as np
from enum import Enum

from trajectory.Trajectory import Trajectory
from trajectory_evaluation.SpatialAlignment import SpatialAlignement


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
