import pandas as pd
import matplotlib.pyplot as plt
import logging

LOGGER = logging.getLogger("EventVisualizer")


class EventVisualizer:
    def __init__(self, pitch_length=106, pitch_width=56):
        LOGGER.info("Event Visualizer initialized")
        self.color_dict = {"PASS": "#3D7EDB", "SHOT": "#FFA100", "SET PIECE": "gray"}
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def plot_input_data(self, input_data):
        """Plot all events in input data

        Args:
            input_data (pd.DataFrame): input dataframe
        """

        for i, row in input_data.dropna(subset=["End X"]).iterrows():
            plt.arrow(
                row["Start X"],
                row["Start Y"],
                row["End X"] - row["Start X"],
                row["End Y"] - row["Start Y"],
                color=self.color_dict[row["Type"]],
                head_width=1,
                head_length=2,
                length_includes_head=True,
            )
        plt.xlim((0, self.pitch_length))
        plt.ylim((0, self.pitch_width))
        plt.show()
        plt.close()
