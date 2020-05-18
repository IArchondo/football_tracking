import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

LOGGER = logging.getLogger("TrackingReader")


class TrackingReader:
    def __init__(self):
        LOGGER.info("TrackingReader initiated")

        self.processed_data = None

    def __calculateDistance(self, x1, y1, x2, y2):
        """Calculate Distance between two points

        Args:
            x1 (float): x coordinate first point
            y1 (float): y coordinate first point
            x2 (float): x coordinate second point
            y2 (float): y coordinate second point

        Returns:
            float: calculated distance
        """
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def __mirror_point(self, input_point, middle):
        """Mirror point

        Args:
            input_point (float): input point in coordinate
            middle (float): Middle point on which to mirror

        Returns:
            float: mirrored point
        """
        distance = middle - input_point
        corrected_point = input_point + 2 * distance
        return corrected_point

    def read_tracking_data(self, file_path):
        """Import tracking data from csv

        Args:
            file_path (str): File path as string

        Returns:
            pd.DataFrame: output file
        """

        tracking_data = pd.read_csv(file_path, header=2)

        tracking_data["Second"] = tracking_data["Time [s]"].apply(lambda x: int(x))

        return tracking_data

    def __adjust_direction(self, input_data, middle_x=53.0, middle_y=30.0):
        """Mirror positions in second half so they correspond to the direction
        in first half

        Args:
            input_data (pd.DataFrame): input data
            middle_x (float, optional): middlepoint x axis. Defaults to 53.0.
            middle_y (float, optional): middlepoint y. Defaults to 30.

        Returns:
            pd.DataFrame: shifted dataframe
        """
        second_half = input_data[input_data["Period"] == 2]
        input_data = input_data[input_data["Period"] == 1]

        second_half[
            [col for col in second_half.columns if col[-2:] == "_x"]
        ] = second_half[[col for col in second_half.columns if col[-2:] == "_x"]].apply(
            lambda x: self.__mirror_point(x, middle_x)
        )

        second_half[
            [col for col in second_half.columns if col[-2:] == "_y"]
        ] = second_half[[col for col in second_half.columns if col[-2:] == "_y"]].apply(
            lambda y: self.__mirror_point(y, middle_y)
        )

        input_data = pd.concat([input_data, second_half])

        return input_data

    def __adjust_coordinates(self, input_data, pitch_length=106, pitch_width=68):
        """Adjust coordinates to match pitch length

        Args:
            input_data (pd.DataFrame): event dataframe

        Returns:
            pd.DataFrame: Adusted data
        """
        LOGGER.info("Adjusting coordinates")
        input_data[
            [col for col in input_data.columns if col[-2:] == "_x"]
        ] = input_data[[col for col in input_data.columns if col[-2:] == "_x"]].apply(
            lambda x: round((x) * pitch_length, 1)
        )
        input_data[
            [col for col in input_data.columns if col[-2:] == "_y"]
        ] = input_data[[col for col in input_data.columns if col[-2:] == "_y"]].apply(
            lambda x: round((x) * pitch_width, 1)
        )

        return input_data

    def __fix_col_names(self, input_df):
        """Fix data column names

        Args:
            input_df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: df with fixed column names
        """
        LOGGER.info("Fixing column names")
        corr_columns = []

        for i, col in enumerate(input_df.columns):
            if (col[:6] == "Player") | (col[:6] == "Ball"):
                corr_columns.append(col + "_x")
            elif col[:6] == "Unname":
                corr_columns.append(input_df.columns[i - 1] + "_y")
            else:
                corr_columns.append(col)

        input_df.columns = corr_columns

        return input_df

    def average_per_second(self, input_data):

        simp = (
            input_data.groupby(["Period", "Second"])[
                [
                    col
                    for col in input_data
                    if (col[:6] == "Player") or (col[:4] == "Ball")
                ]
            ]
            .mean()
            .reset_index()
        )

        return simp

    def process_tracking_data(self, file_path):
        """Process raw data file

        Args:
            file_path (str): file path

        Returns:
            pd.DataFrame: processed dataframe
        """

        raw_data = self.read_tracking_data(file_path)

        clean_data = self.__fix_col_names(raw_data)

        clean_data = self.__adjust_coordinates(clean_data)

        clean_data = self.__adjust_direction(clean_data)

        self.processed_data = self.average_per_second(clean_data)

        return clean_data

    def calculate_distance_per_player(self, player_no):
        """Calculate distance run for a given player

        Args:
            player_no (int): player number

        Returns:
            float: distance run from player
        """
        if self.processed_data is None:
            LOGGER.error(
                "Processed data not available, first run process_tracking_data"
            )

        else:
            player_no = "Player" + str(player_no)
            distance = []

            distance_df = self.processed_data.dropna(subset=[player_no + "_x"])

            for index, row in distance_df.iterrows():
                if index == 0:
                    LOGGER.info("Calculating distance")
                else:
                    calc_dist = self.__calculateDistance(
                        row[player_no + "_x"],
                        row[player_no + "_y"],
                        distance_df.iloc[index - 1][player_no + "_x"],
                        distance_df.iloc[index - 1][player_no + "_y"],
                    )
                    distance.append(calc_dist)

            return round(sum(distance), 2)

    def plot_heatmap_per_player(self, player_no):
        """Plot heatmap for a given player

        Args:
            player_no (int): player number
        """
        if self.processed_data is None:
            LOGGER.error(
                "Processed data not available, first run process_tracking_data"
            )

        else:
            player_no = "Player" + str(player_no)
            heat_df = self.processed_data.dropna(subset=[player_no + "_x"])

            x_ax = heat_df[player_no + "_x"].apply(lambda x: int(x))

            y_ax = heat_df[player_no + "_y"].apply(lambda x: int(x))

            heat = pd.DataFrame({"x": x_ax, "y": y_ax})

            heat = heat.groupby(["x", "y"]).agg({"x": "count"})

            heat.columns = ["count"]

            heat = heat.reset_index()

            heat = heat.pivot(index="y", columns="x", values="count").fillna(0)

            plt.figure(figsize=(8, 60 / 106 * 8))
            plt.rcParams["axes.facecolor"] = "lightgreen"
            sns.heatmap(heat, cmap="hot_r")
            plt.axis("off")
