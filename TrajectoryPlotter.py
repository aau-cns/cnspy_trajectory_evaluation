import os
import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler

from tum_eval.TUMCSV2DataFrame import TUMCSV2DataFrame
from Trajectory import Trajectory


class TrajectoryPlotTypes(Enum):
    scatter_3D = 'scatter_3D'
    plot_3D = 'plot_3D'
    plot_2D_over_t = 'plot_2D_over_t'
    plot_2D_over_dist = 'plot_2D_over_dist'

    def __str__(self):
        return self.value


class TrajectoryPlotConfig():
    white_list = []
    num_points = []
    plot_type = TrajectoryPlotTypes.plot_3D
    dpi = 200
    title = ""
    scale = 1.0
    save_fn = ""
    result_dir = "."
    show = True
    close_figure = False

    def __init__(self, white_list=[], num_points=[],
                 plot_type=TrajectoryPlotTypes.plot_3D, dpi=200, title="",
                 scale=1.0, save_fn="", result_dir=".", show=True, close_figure=False):
        self.white_list = white_list
        self.num_points = num_points
        self.plot_type = plot_type
        self.dpi = dpi
        self.title = title
        self.scale = scale
        self.save_fn = save_fn
        self.result_dir = result_dir
        self.show = show
        self.close_figure = close_figure


class TrajectoryPlotter:
    traj_obj = None
    traj_df = None
    config = None

    def __init__(self, traj_obj, config=TrajectoryPlotConfig()):
        if config.num_points > 0:
            # subsample trajectory first:
            df = traj_obj.to_DataFrame()
            self.traj_df = TUMCSV2DataFrame.subsample_DataFrame(df, num_max_points=config.num_points)
            self.traj_obj = Trajectory(df=self.traj_df)
        else:
            self.traj_obj = traj_obj
            self.traj_df = traj_obj.to_DataFrame()

        self.config = config

    @staticmethod
    def plot_n_dim(ax, x_linespace, values,
                   colors=['r', 'g', 'b'],
                   labels=['x', 'y', 'z']):
        assert len(colors) == len(labels)
        if len(colors) > 1:
            assert len(colors) == values.shape[1]
            for i in range(len(colors)):
                ax.plot(x_linespace, values[:, i],
                        colors[i] + '-', label=labels[i])
        else:
            ax.plot(x_linespace, values, colors[0] + '-', label=labels[0])

    def get_pos_data(self, cfg):
        data_dict = TUMCSV2DataFrame.DataFrame_to_numpy_dict(self.traj_df)
        ts = data_dict['t']
        xs = data_dict['tx']
        ys = data_dict['ty']
        zs = data_dict['tz']
        dist_vec = self.traj_obj.get_accumulated_distances()

        if cfg.white_list:
            print("white list args: " + str(cfg.white_list))
        if any([flag == 'x' for flag in cfg.white_list]):
            xs = []
            print("clear xs")
        if any([flag == 'y' for flag in cfg.white_list]):
            ys = []
            print("clear ys")
        if any([flag == 'z' for flag in cfg.white_list]):
            zs = []
            print("clear zs")

        if not (len(xs) and isinstance(xs[0], np.float64) and not math.isnan(xs[0])):
            xs = []
        if not (len(ys) and isinstance(ys[0], np.float64) and not math.isnan(ys[0])):
            ys = []
        if not (len(zs) and isinstance(zs[0], np.float64) and not math.isnan(zs[0])):
            zs = []

        if cfg.scale and cfg.scale != 1.0:
            scale = float(cfg.scale)
            xs *= scale
            ys *= scale
            zs *= scale

        return ts, xs, ys, zs, dist_vec

    def get_rpy_data(self, cfg, in_radians=True):
        rpy_vec = self.traj_obj.get_rpy_vec()

        rpy_vec = np.unwrap(rpy_vec, axis=0)
        if not in_radians:
            rpy_vec = np.rad2deg(rpy_vec)

        ts = self.traj_obj.t_vec
        rs = rpy_vec[:, 0]
        ps = rpy_vec[:, 1]
        ys = rpy_vec[:, 2]
        dist_vec = self.traj_obj.get_accumulated_distances()

        if cfg.white_list:
            print("white list args: " + str(cfg.white_list))
        if any([flag == 'roll' for flag in cfg.white_list]):
            rs = []
            print("clear rs")
        if any([flag == 'pitch' for flag in cfg.white_list]):
            ps = []
            print("clear ps")
        if any([flag == 'yaw' for flag in cfg.white_list]):
            ys = []
            print("clear ys")

        if not (len(rs) and isinstance(rs[0], np.float64) and not math.isnan(rs[0])):
            xs = []
        if not (len(ps) and isinstance(ps[0], np.float64) and not math.isnan(ps[0])):
            ps = []
        if not (len(ys) and isinstance(ys[0], np.float64) and not math.isnan(ys[0])):
            ys = []

        return ts, rs, ps, ys, dist_vec

    def plot_pos(self, ax, cfg):
        ts, xs, ys, zs, dist_vec = self.get_pos_data(cfg)

        if cfg.plot_type == TrajectoryPlotTypes.plot_2D_over_dist:
            linespace = dist_vec
            ax.set_xlabel('distance [m]')
        else:
            ts = ts - ts[0]
            linespace = ts
            ax.set_xlabel('rel. time [sec]')
        ax.set_ylabel('position [m]')

        if len(xs):
            TrajectoryPlotter.plot_n_dim(ax, linespace, xs, colors=['r'], labels=['x'])
        if len(ys):
            TrajectoryPlotter.plot_n_dim(ax, linespace, ys, colors=['g'], labels=['y'])
        if len(zs):
            TrajectoryPlotter.plot_n_dim(ax, linespace, zs, colors=['b'], labels=['z'])

    def plot_rpy(self, ax, cfg, in_radians=True):
        ts, xs, ys, zs, dist_vec = self.get_rpy_data(cfg, in_radians=in_radians)

        if cfg.plot_type == TrajectoryPlotTypes.plot_2D_over_dist:
            linespace = dist_vec
            ax.set_xlabel('distance [m]')
        else:
            ts = ts - ts[0]
            linespace = ts
            ax.set_xlabel('rel. time [sec]')

        if in_radians:
            ax.set_ylabel('rotation [rad]')
        else:
            ax.set_ylabel('rotation [deg]')

        if len(xs):
            TrajectoryPlotter.plot_n_dim(ax, linespace, xs, colors=['r'], labels=['x'])
        if len(ys):
            TrajectoryPlotter.plot_n_dim(ax, linespace, ys, colors=['g'], labels=['y'])
        if len(zs):
            TrajectoryPlotter.plot_n_dim(ax, linespace, zs, colors=['b'], labels=['z'])

    def plot_q(self, ax, cfg):
        q_vec = self.traj_obj.q_vec
        if cfg.plot_type == TrajectoryPlotTypes.plot_2D_over_dist:
            x_linespace = self.traj_obj.get_accumulated_distances()
            ax.set_xlabel('distance [m]')
        else:
            x_linespace = self.traj_obj.t_vec
            x_linespace = x_linespace - x_linespace[0]
            ax.set_xlabel('rel. time [sec]')
        ax.set_ylabel('quaternion')

        TrajectoryPlotter.plot_n_dim(ax, x_linespace, q_vec, colors=['r', 'g', 'b', 'k'],
                                     labels=['qx', 'qy', 'qz', 'qw'])

    def plot_pose(self, fig=None, cfg=None, angles=False, in_radians=True):
        if cfg is None:
            cfg = self.config

        if fig is None:
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))

        ax1 = fig.add_subplot(211)
        self.plot_pos(ax=ax1, cfg=cfg)
        ax2 = fig.add_subplot(212)

        if angles:
            self.plot_rpy(ax=ax2, cfg=cfg, in_radians=in_radians)
        else:
            self.plot_q(ax=ax2, cfg=cfg)

        plt.draw()
        plt.pause(0.001)
        if cfg.save_fn:
            filename = os.path.join(cfg.result_dir, cfg.save_fn)
            print("save to file: " + filename)
            plt.savefig(filename, dpi=int(cfg.dpi))
        if cfg.show:
            plt.show()
        if cfg.close_figure:
            plt.close(fig)

        return fig, ax1, ax2

    def plot_pos_3D(self, ax, cfg=None, label="trajectory"):
        if cfg is None:
            cfg = self.config

        ts, xs, ys, zs, dist_vec = self.get_pos_data(cfg)

        if cfg.plot_type == TrajectoryPlotTypes.scatter_3D:
            ax.scatter(xs, ys, zs, zdir='z', label=str(label))
        elif cfg.plot_type == TrajectoryPlotTypes.plot_3D:
            ax.plot3D(xs, ys, zs, label=str(label))

    def plot_3D(self, cfg=None, ax=None, fig=None):
        if cfg is None:
            cfg = self.config

        if (ax is None) or (fig is None):
            fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
            ax = fig.add_subplot(111, projection='3d')

        if cfg.title:
            ax.set_title(cfg.title)
        else:
            if cfg.plot_type == TrajectoryPlotTypes.scatter_3D:
                ax.set_title("Scatter Plot")
            else:
                ax.set_title("Plot3D")

        self.plot_pos_3D(ax=ax, cfg=cfg)

        ax.legend(shadow=True, fontsize='x-small')
        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.draw()
        plt.pause(0.001)

        if cfg.save_fn:
            filename = os.path.join(cfg.result_dir, cfg.save_fn)
            print("save to file: " + filename)
            plt.savefig(filename, dpi=int(cfg.dpi))
        if cfg.show:
            plt.show()
        if cfg.close_figure:
            plt.close(fig)

        return fig, ax

    @staticmethod
    def multi_plot_3D(traj_plotter_list, cfg, name_list=[]):

        num_plots = len(traj_plotter_list)

        fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.spectral(np.linspace(0.1, 0.9, num_plots))
        ax.set_prop_cycle('color', colors)

        idx = 0
        for traj in traj_plotter_list:
            traj.plot_pos_3D(ax=ax, label=name_list[idx])
            idx += 1

        ax.legend(shadow=True, fontsize='x-small')
        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.draw()
        plt.pause(0.001)

        if cfg.save_fn:
            filename = os.path.join(cfg.result_dir, cfg.save_fn)
            print("save to file: " + filename)
            plt.savefig(filename, dpi=int(cfg.dpi))
        if cfg.show:
            plt.show()
        if cfg.close_figure:
            plt.close(fig)

        return fig, ax


########################################################################################################################
#################################################### T E S T ###########################################################
########################################################################################################################
import unittest
import math


class TrajectoryPlotter_Test(unittest.TestCase):
    def load_trajectory_from_CSV(self):
        traj = Trajectory()
        traj.load_from_CSV(filename='../test/example/stamped_groundtruth.csv')
        self.assertFalse(traj.is_empty())
        return traj

    def load_trajectory2_from_CSV(self):
        traj = Trajectory()
        traj.load_from_CSV(filename='../test/example/est.csv')
        self.assertFalse(traj.is_empty())
        return traj

    # def test_plot_3D(self):
    #     traj = self.load_trajectory_from_CSV()
    #
    #     plotter = TrajectoryPlotter(traj_obj=traj, config=TrajectoryPlotConfig(show=False, close_figure=False))
    #     plotter.plot_3D()

    def test_plot_pose(self):
        traj = self.load_trajectory_from_CSV()

        plotter = TrajectoryPlotter(traj_obj=traj, config=TrajectoryPlotConfig(show=False, close_figure=False))
        plotter.plot_pose()
        plotter.plot_pose(angles=True, in_radians=True)
        plotter.plot_pose(angles=True, in_radians=False)
        plotter.plot_pose(angles=True, in_radians=False, cfg=TrajectoryPlotConfig(show=False, close_figure=False,
                                                                                  plot_type=TrajectoryPlotTypes.plot_2D_over_dist))
        print('done')

    # def test_plot_multi(self):
    #     plotter1 = TrajectoryPlotter(traj_obj=self.load_trajectory_from_CSV(),
    #                                  config=TrajectoryPlotConfig(num_points=120000))
    #     plotter2 = TrajectoryPlotter(traj_obj=self.load_trajectory2_from_CSV())
    #
    #     TrajectoryPlotter.multi_plot_3D([plotter1, plotter2], cfg=TrajectoryPlotConfig(show=True),
    #                                     name_list=['gt', 'est'])


if __name__ == "__main__":
    unittest.main()
