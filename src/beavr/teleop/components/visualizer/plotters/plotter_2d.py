import logging
import os
import warnings

import cv2
import matplotlib
import matplotlib.pyplot as plt

from beavr.teleop.common.io.files import check_file, get_npz_data, make_dir
from beavr.teleop.common.network.publisher import ZMQCompressedImageTransmitter
from beavr.teleop.configs.constants import robots

from .plotter import Plotter

logger = logging.getLogger(__name__)


# Specifically suppress the FigureCanvasAgg warning
warnings.filterwarnings("ignore", category=UserWarning, message="FigureCanvasAgg is non-interactive")
matplotlib.use("TkAgg")  # Set the backend to TkAgg before importing pyplot


def plot_line(x1, x2, y1, y2):
    plt.plot([x1, x2], [y1, y2])


class PlotHand2D(Plotter):
    def __init__(self, host: str, port: int, display_plot: bool = False):
        assert isinstance(display_plot, bool), f"display_plot must be bool, got {type(display_plot)}"
        if display_plot:
            plt.ion()
        else:
            plt.switch_backend("Agg")
            plt.ioff()

        # Thumb bound info
        self.thumb_bounds = None
        self.thumb_bounds_path = robots.VR_DISPLAY_THUMB_BOUNDS_PATH
        self.bound_update_counter = 0
        self._check_thumb_bounds()

        # Checking image storage path
        make_dir(os.path.join(robots.CALIBRATION_FILES_PATH))

        # Figure settings
        self.fig = plt.figure(figsize=(6, 6), dpi=60)

        # Plot streamer settings
        self.socket = ZMQCompressedImageTransmitter(host=host, port=port)

    def _check_thumb_bounds(self):
        if check_file(self.thumb_bounds_path):
            self.thumb_bounds = get_npz_data(self.thumb_bounds_path)

    def _set_limits(self):
        plt.axis([-0.12, 0.12, -0.02, 0.2])

    def _draw_thumb_bounds(self):
        for idx in range(robots.VR_THUMB_BOUND_VERTICES):
            plot_line(
                self.thumb_bounds[idx][0],
                self.thumb_bounds[(idx + 1) % robots.VR_THUMB_BOUND_VERTICES][0],
                self.thumb_bounds[idx][1],
                self.thumb_bounds[(idx + 1) % robots.VR_THUMB_BOUND_VERTICES][1],
            )

    def draw_hand(self, x, y):
        plt.plot(x, y, "ro")

        if self.thumb_bounds is not None:
            self._draw_thumb_bounds()

        # Drawing connections fromn the wrist - 0
        for idx in robots.OCULUS_JOINTS["metacarpals"]:
            plot_line(x[0], x[idx], y[0], y[idx])

        # Drawing knuckle to knuckle connections and knuckle to finger connections
        for key in ["knuckles", "thumb", "index", "middle", "ring", "pinky"]:
            for idx in range(len(robots.OCULUS_JOINTS[key]) - 1):
                plot_line(
                    x[robots.OCULUS_JOINTS[key][idx]],
                    x[robots.OCULUS_JOINTS[key][idx + 1]],
                    y[robots.OCULUS_JOINTS[key][idx]],
                    y[robots.OCULUS_JOINTS[key][idx + 1]],
                )

    def draw(self, x, y):
        # Setting the plot limits
        self._set_limits()

        # Resetting the thumb bounds
        if self.bound_update_counter % 10 == 0:
            self._check_thumb_bounds()
            self.bound_update_counter = 0
        else:
            self.bound_update_counter += 1

        # Plotting the lines to visualize the hand
        self.draw_hand(x, y)

        # Saving and obtaining the plot
        plt.savefig(robots.VR_2D_PLOT_SAVE_PATH)
        plot = cv2.imread(robots.VR_2D_PLOT_SAVE_PATH)
        self.socket.send_image(plot)

        # Resetting and pausing the 3D plot
        try:
            plt.pause(0.001)  # Keep this for real-time updates
        except Exception as e:
            if self.debug:
                logger.warning(f"Warning: Plot update failed: {e}")
        plt.cla()
