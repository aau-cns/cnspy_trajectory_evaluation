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

    def __str__(self):
        return self.value


class TrajectoryPlotConfig():
    white_list = []
    num_points = []
    plot_type = TrajectoryPlotTypes.plot_3D
    dpi = 200
    title = "plot"
    scale = 1.0
    save_fn = "result.png"
    result_dir = "."
    show = True
    close_figure = False

    def __init__(self, white_list=[], num_points=[],
                 plot_type=TrajectoryPlotTypes.plot_3D, dpi=200, title="plot",
                 scale=1.0, save_fn="result.png", result_dir=".", show=True, close_figure=False):
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

    def plot_on_axes(self, ax, cfg=None, label="trajectory"):
        if cfg is None:
            cfg = self.config

        data_dict = TUMCSV2DataFrame.DataFrame_to_numpy_dict(self.traj_df)
        ts = data_dict['t']
        xs = data_dict['tx']
        ys = data_dict['ty']
        zs = data_dict['tz']

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

        if cfg.plot_type == TrajectoryPlotTypes.scatter_3D:
            ax.scatter(xs, ys, zs, zdir='z', label=str(label))
        elif cfg.plot_type == TrajectoryPlotTypes.plot_3D:
            ax.plot3D(xs, ys, zs, label=str(label))

    def plot(self, cfg=None, ax=None, fig=None):
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

        self.plot_on_axes(ax=ax, cfg=cfg)

        ax.legend(shadow=True, fontsize='x-small')
        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

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
    def multi_plot(traj_plotter_list, cfg, name_list=[]):

        num_plots = len(traj_plotter_list)

        fig = plt.figure(figsize=(20, 15), dpi=int(cfg.dpi))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.spectral(np.linspace(0.1, 0.9, num_plots))
        ax.set_prop_cycle('color', colors)

        idx = 0
        for traj in traj_plotter_list:
            traj.plot_on_axes(ax=ax, label=name_list[idx])
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

    def test_plot(self):
        traj = self.load_trajectory_from_CSV()

        plotter = TrajectoryPlotter(traj_obj=traj, config=TrajectoryPlotConfig(show=False, close_figure=False))
        plotter.plot()

    def test_plot_multi(self):
        plotter1 = TrajectoryPlotter(traj_obj=self.load_trajectory_from_CSV(),
                                     config=TrajectoryPlotConfig(num_points=120000))
        plotter2 = TrajectoryPlotter(traj_obj=self.load_trajectory2_from_CSV())

        TrajectoryPlotter.multi_plot([plotter1, plotter2], cfg=TrajectoryPlotConfig(show=True),
                                     name_list=['gt', 'est'])


if __name__ == "__main__":
    unittest.main()
