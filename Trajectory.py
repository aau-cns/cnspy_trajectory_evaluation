#!/usr/bin/env python

import os
import numpy as np

from tum_eval.TUMCSV2DataFrame import TUMCSV2DataFrame


class Trajectory:
    p_vec = []
    q_vec = []
    t_vec = []

    def __init__(self, t_vec=[], p_vec=[], q_vec=[]):
        self.t_vec = t_vec
        self.p_vec = p_vec
        self.q_vec = q_vec

    def load_from_CSV(self, filename, sep='\s+|\,', comment='#',
                      header=['t', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']):
        if not os.path.isfile(filename):
            print("could not find trajectory file %s" % filename)
            return False

        df = TUMCSV2DataFrame.load_TUM_CSV(filename=filename, sep=sep, comment=comment, header=header)
        self.t_vec, self.p_vec, self.q_vec = TUMCSV2DataFrame.data_frame_to_tpq(data_frame=df)
        return True

    def save_to_CSV(self, filename):
        df = TUMCSV2DataFrame.tpq_to_data_frame(self.t_vec, self.p_vec, self.q_vec)
        TUMCSV2DataFrame.save_TUM_CSV(df, filename=filename)

    def get_distance(self):
        accum_distances = self.get_accumulated_distances()
        traj_length = accum_distances[-1]
        return traj_length

    def get_accumulated_distances(self):
        return Trajectory.get_distances_from_start(self.p_vec)

    def is_empty(self):
        return (self.t_vec.size == 0)

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
    def test_load_trajectory_from_CSV(self):
        print('loading...')
        traj = Trajectory()
        traj.load_from_CSV(filename='/home/jungr/workspace/NAV/development/aaunav_data_analysis_py/test/example/gt.csv')

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


if __name__ == "__main__":
    unittest.main()
