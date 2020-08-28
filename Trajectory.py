#!/usr/bin/env python

import os
import numpy as np
import pandas as pandas

from rosbag2csv.CSVFormat import CSVFormat
from tum_eval.TUMCSV2DataFrame import TUMCSVdata


class Trajectory:
    p_vec = []
    q_vec = []
    t_vec = []

    def __init__(self, t_vec=[], p_vec=[], q_vec=[]):
        self.t_vec = t_vec
        self.p_vec = p_vec
        self.q_vec = q_vec

    def load_trajectory_from_CSV(self, filename, format=CSVFormat.TUM):
        if not os.path.isfile(filename):
            print("could not find trajectory file %s" % filename)
            return False

    def load_from_CSV(self, filename, sep='\s+|\,', comment='#',
                      header=['t', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']):
        data = TUMCSVdata.load_TUM_CSV(filename=filename, sep=sep, comment=comment, header=header)

    def save_to_CSV(self, filename, format=CSVFormat.TUM):
        pass

    def get_distance(self):
        accum_distances = self.get_accumulated_distance()
        traj_length = accum_distances[-1]
        return traj_length

    def get_accumulated_distance(self):
        return Trajectory.get_distance_from_start(self.p_vec)

    def is_empty(self):
        return (self.t_vec.size == 0)

    @staticmethod
    def get_distance_from_start(gt_translation):
        distances = np.diff(gt_translation[:, 0:3], axis=0)
        distances = np.sqrt(np.sum(np.multiply(distances, distances), 1))
        distances = np.cumsum(distances)
        distances = np.concatenate(([0], distances))
        return distances
