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
# numpy, pandas, cnspy_numpy_utils, cnspy_trajectory, scipy, matplotlib
########################################################################################################################
from sys import version_info
import numpy as np
from scipy.stats.distributions import chi2


# https://de.mathworks.com/help/fusion/ref/trackerrormetrics-system-object.html
from cnspy_trajectory.PlotLineStyle import PlotLineStyle
from cnspy_trajectory.TrajectoryPlotUtils import TrajectoryPlotUtils


def toNEES_arr(P_arr, err_arr):
    # if is_angle:
    #    err_arr = tf.euler_from_quaternion(err_arr, 'rzyx')

    l = err_arr.shape[0]
    nees_arr = np.zeros((l, 1))
    for i in range(0, l):
        try:
            nees_arr[i] = toNEES(P=P_arr[i], err=err_arr[i])
        except np.linalg.LinAlgError:
            print("NEES.toNEES(): covariance causes np.linalg.LinAlgError! ")
            print(P_arr[i])

    return nees_arr


def toNEES(P, err):
    """
    computes the Normalized Estimation Error Square (Mahalanobis Distance Squared)
    The corresponding covariance matrix (P) must be in the same space and unit as the error!
    E.g., for local/body rotations the uncertainty and e.g. the error
    (err = theta_so3  with R_err = expm(theta_so3)) must be in the same space and unit
    """

    N = len(err)
    tr = np.trace(P)
    eig_vals, eig_vec = np.linalg.eig(P)
    if tr < 1e-16:
        nees = 0
    elif any(eig_vals < 0):
        nees = 0
    else:
        P_inv = np.linalg.inv(P)
        nees = np.matmul(np.matmul(err.reshape(1, N), P_inv), err.reshape(N, 1))

    return nees


def chi_square_confidence_bounds(confidence_region=0.95, degrees_of_freedom=3):
    # returns the [r_lower, r_upper] confidence regions
    # https://stackoverflow.com/questions/53019080/chi2inv-in-python
    # ppf(q, df, loc=0, scale=1) Percent point function (inverse of cdf percentiles).

    alpha = 1 - confidence_region
    r_upper = chi2.ppf(q=(1.0 - alpha), df=degrees_of_freedom)
    r_lower = chi2.ppf(q=alpha, df=degrees_of_freedom)
    return r_lower, r_upper


def ax_plot_nees(ax, dim, conf_ival, NEES_vec, x_linespace=None, relative_time=True, x_label_prefix='', y_label='NEES',
                 ls=PlotLineStyle(), plot_intervals=True):
    l = NEES_vec.shape[0]
    avg_NEES = np.mean(NEES_vec)

    if x_linespace is None:
        x_linespace = range(0, l)
        x_label = ''
    else:
        if relative_time:
            x_linespace = x_linespace - x_linespace[0]
            x_label = str(x_label_prefix + 'rel. time [sec]')
        else:
            x_label = str(x_label_prefix + 'time [sec]')


    TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=NEES_vec,
                                        colors=ls.linecolor, labels=['avg. '+ y_label + '={:.3f}'.format(avg_NEES)], ls=ls)
    if plot_intervals:
        conf_ival = float(conf_ival)
        r_lower, r_upper = chi_square_confidence_bounds(confidence_region=conf_ival,
                                                                                degrees_of_freedom=dim)
        y_values = np.ones((l, 1))
        alpha = 1.0 - conf_ival
        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * r_upper,
                                        colors=['k'],
                                        labels=['r1(p={:.3f})={:.3f}'.format(conf_ival, r_upper)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='-.'))

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * r_lower,
                                        colors=['k'],
                                        labels=['r2(p={:.3f})={:.3f}'.format(conf_ival, r_lower)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='-.'))

        TrajectoryPlotUtils.ax_plot_n_dim(ax, x_linespace=x_linespace, values=y_values * dim,
                                        colors=['k'],
                                        labels=['mean={:.0f}'.format(dim)],
                                        ls=PlotLineStyle(linewidth=0.5, linestyle='--'))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    pass
