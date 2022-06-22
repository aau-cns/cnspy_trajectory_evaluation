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
# Requirements:
# configparser
########################################################################################################################
import configparser

# TODO: to compute the ANEES of multiple runs, the NEES-vectors and timestamps need to be stored! As they need to be associated later again.
# TODO: the EvaluationReport does not reflect a real ANEES, only the average NEES over a single run!

class EvaluationReport:
    def __init__(self, directory='', fn_gt='', fn_est='', alignment='none',
                 num_aligned_samples=0, num_runs=0, ANEES_p=0.0, ANEES_R=0.0, ARMSE_p=0.0, ARMSE_R=0.0):
        self.directory = directory
        self.fn_gt = fn_gt
        self.fn_est = fn_est
        self.alignment = str(alignment)
        self.num_aligned_samples = int(num_aligned_samples)
        self.num_runs = int(num_runs)
        self.ANEES_p = ANEES_p
        self.ANEES_R = ANEES_R
        self.ARMSE_p = ARMSE_p
        self.ARMSE_R = ARMSE_R

    def save(self, fn):
        config = configparser.ConfigParser()
        config['EvaluationReport'] = {'directory': self.directory,
                                      'fn_gt': self.fn_gt,
                                      'fn_est': self.fn_est,
                                      'alignment': self.alignment,
                                      'num_aligned_samples': self.num_aligned_samples,
                                      'ANEES_p': self.ANEES_p,
                                      'ANEES_R': self.ANEES_R,
                                      'ARMSE_p': self.ARMSE_p,
                                      'ARMSE_R': self.ARMSE_R}
        # print('Save config file....')
        with open(fn, 'w') as configfile:
            config.write(configfile)
            configfile.close()

    def load(self, fn):
        config = configparser.ConfigParser()
        config.sections()
        config.read(fn)
        # print('load from section')
        section = config['EvaluationReport']
        self.directory = section.get('directory', 'default')
        self.fn_gt = section.get('fn_gt', 'default')
        self.fn_est = section.get('fn_est', 'default')
        self.ANEES_p = section.get('ANEES_p', 'default')
        self.ANEES_R = section.get('ANEES_R', 'default')
        self.ARMSE_p = section.get('ARMSE_p', 'default')
        self.ARMSE_R = section.get('ARMSE_R', 'default')

