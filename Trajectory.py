#!/usr/bin/env python

import os
import numpy as np
import transformations as tf
from tum_eval.TUMCSV2DataFrame import TUMCSV2DataFrame


class Trajectory:
    p_vec = None
    q_vec = None
    t_vec = None

    def __init__(self, t_vec=None, p_vec=None, q_vec=None, df=None):
        if df is not None:
            self.load_from_DataFrame(df)
        else:
            self.t_vec = t_vec
            self.p_vec = p_vec
            self.q_vec = q_vec

    def load_from_CSV(self, filename, sep='\s+|\,', comment='#',
                      header=['t', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']):
        if not os.path.isfile(filename):
            print("Trajectory: could not find file %s" % os.path.abspath(filename))
            return False

        df = TUMCSV2DataFrame.load_TUM_CSV(filename=filename, sep=sep, comment=comment, header=header)
        self.load_from_DataFrame(df)
        return True

    def load_from_DataFrame(self, df):
        self.t_vec, self.p_vec, self.q_vec = TUMCSV2DataFrame.DataFrame_to_TPQ(data_frame=df)

    def to_DataFrame(self):
        return TUMCSV2DataFrame.TPQ_to_DataFrame(self.t_vec, self.p_vec, self.q_vec)

    def save_to_CSV(self, filename):
        if self.is_empty():
            return False
        df = TUMCSV2DataFrame.TPQ_to_DataFrame(self.t_vec, self.p_vec, self.q_vec)
        TUMCSV2DataFrame.save_TUM_CSV(df, filename=filename)
        return True

    def is_empty(self):
        return self.t_vec is None

    def num_elems(self):
        if self.is_empty():
            return 0
        else:
            return len(self.t_vec)

    def get_distance(self):
        return Trajectory.get_distance(self.p_vec)

    def get_accumulated_distances(self):
        return Trajectory.get_distances_from_start(self.p_vec)

    def get_rpy_vec(self):
        rpy_vec = np.zeros(np.shape(self.p_vec))
        for i in range(np.shape(self.p_vec)[0]):
            rpy_vec[i, :] = tf.euler_from_quaternion(self.q_vec[i, :], 'rzyx')

        return rpy_vec

    def plot(self):
        assert (False)

    def transform_p(self, scale=1.0, t=np.zeros((3,)), R=np.identity(3)):
        p_es_aligned = np.zeros(np.shape(self.p_vec))
        q_es_aligned = np.zeros(np.shape(self.q_vec))
        for i in range(np.shape(self.p_vec)[0]):
            p_es_aligned[i, :] = R.dot(scale * self.p_vec[i, :]) + t
            q_es_R = R.dot(tf.quaternion_matrix(self.q_vec[i, :])[0:3, 0:3])
            q_es_T = np.identity(4)
            q_es_T[0:3, 0:3] = q_es_R
            q_es_aligned[i, :] = tf.quaternion_from_matrix(q_es_T)

        # self.p_vec = R * (scale * self.p_vec) + t

        self.p_vec = p_es_aligned
        self.q_vec = q_es_aligned

    def transform_q(self, R=np.identity(3)):
        assert (False)

    @staticmethod
    def get_distances_from_start(p_vec):
        distances = np.diff(p_vec[:, 0:3], axis=0)
        distances = np.sqrt(np.sum(np.multiply(distances, distances), 1))
        distances = np.cumsum(distances)
        distances = np.concatenate(([0], distances))
        return distances

    @staticmethod
    def get_distance(p_vec):
        accum_distances = Trajectory.get_distances_from_start(p_vec)
        return accum_distances[-1]


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import math


class Trajectory_Test(unittest.TestCase):
    def load_trajectory_from_CSV(self):
        print('loading...')
        traj = Trajectory()
        traj.load_from_CSV(filename='../test/example/gt.csv')
        return traj

    def test_load_trajectory_from_CSV(self):
        traj = self.load_trajectory_from_CSV()
        self.assertTrue(traj.num_elems() > 0)

    def test_save_to_CSV(self):
        traj = Trajectory()

        saved = traj.save_to_CSV('/home/jungr/workspace/NAV/development/aaunav_data_analysis_py/test/example/empty.csv')
        self.assertFalse(saved)

        traj.p_vec = np.array([[0, 0, 0],
                               [1, 0, 0],
                               [2, 0, 0],
                               [3, 0, 0]])
        traj.q_vec = np.array([[0, 0, 0, 1],
                               [0, 0, 0, 1],
                               [0, 0, 0, 1],
                               [0, 0, 0, 1]])
        traj.t_vec = np.array([[0],
                               [1],
                               [2],
                               [3]])
        saved = traj.save_to_CSV('/home/jungr/workspace/NAV/development/aaunav_data_analysis_py/test/example/empty.csv')
        self.assertTrue(saved)

    def test_get_distance_from_start(self):
        p_vec = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [2, 0, 0],
                          [3, 0, 0]])

        d = Trajectory.get_distance(p_vec)
        self.assertTrue(math.floor(d - 3.0) == 0)

        p_vec = np.array([[0, 0, 0],
                          [1, 1, 0],
                          [2, 2, 0],
                          [3, 3, 0]])
        d = Trajectory.get_distance(p_vec)
        d_ = math.sqrt(9 + 9)
        self.assertTrue(math.floor(d - d_) == 0)

        p_vec = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3]])

        d = Trajectory.get_distance(p_vec)
        d_ = math.sqrt(9 + 9 + 9)
        self.assertTrue(math.floor(d - d_) == 0)

    def test_get_rpy_vec(self):
        traj = self.load_trajectory_from_CSV()
        rpy_vec = traj.get_rpy_vec()
        self.assertTrue(rpy_vec.shape[0] == traj.num_elems())
        self.assertTrue(rpy_vec.shape[1] == 3)


if __name__ == "__main__":
    unittest.main()
