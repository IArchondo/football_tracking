import pandas as pd
import numpy as np
import logging

LOGGER = logging.getLogger("EventReader")


class EventReader:
    def __init__(self):
        LOGGER.info("EventReader initiated")

    def read_event_data(self, file_path):
        """Import event data from csv

        Args:
            file_path (str): File path as string

        Returns:
            pd.DataFrame: Pandas Dataframe
        """

        event_data = pd.read_csv(file_path)

        return event_data

    def adjust_coordinates(self, input_data, pitch_length=106, pitch_width=68):
        """Adjust coordinates to match pitch length

        Args:
            input_data (pd.DataFrame): event dataframe

        Returns:
            pd.DataFrame: Adusted data
        """
        input_data[
            [col for col in input_data.columns if col[-2:] == " X"]
        ] = input_data[[col for col in input_data.columns if col[-2:] == " X"]].apply(
            lambda x: round((x) * pitch_length, 1)
        )
        input_data[
            [col for col in input_data.columns if col[-2:] == " Y"]
        ] = input_data[[col for col in input_data.columns if col[-2:] == " Y"]].apply(
            lambda x: round((x) * pitch_width, 1)
        )

        return input_data

    def get_game_stats(self, input_data):
        """Calculate game stats per team

        Args:
            input_data (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: Game Stats
        """
        home_stats = (
            input_data[input_data["Team"] == "Home"]
            .groupby("Type")
            .agg({"Team": "count"})
            .reset_index()
        )
        home_stats.columns = ["Action", "Home"]
        away_stats = (
            input_data[input_data["Team"] == "Away"]
            .groupby("Type")
            .agg({"Team": "count"})
            .reset_index()
        )
        away_stats.columns = ["Action", "Away"]

        game_stats = home_stats.merge(away_stats, on="Action", how="left")

        game_stats.loc[game_stats.shape[0] + 1] = [
            "GOALS",
            input_data[
                (input_data["Team"] == "Home")
                & (input_data["Type"] == "SHOT")
                & (input_data["Subtype"].str.contains("-GOAL", na=False))
            ].shape[0],
            input_data[
                (input_data["Team"] == "Away")
                & (input_data["Type"] == "SHOT")
                & (input_data["Subtype"].str.contains("-GOAL", na=False))
            ].shape[0],
        ]

        return game_stats

    def get_play_from_index(self, input_data, play_index):
        """Get complete play up to the requested index

        Args:
            input_data (pd.DataFrame): input dataframe
            play_index (int): index from the required play

        Returns:
            pd.DataFrame: dataframe with all steps in the play
        """
        play_team = input_data.loc[play_index]["Team"]
        ## filter data until desired point
        until_spl = input_data.loc[:play_index]

        ## get start of play
        start_play = until_spl[until_spl["Team"] != play_team].index[-1] + 1

        play_data = input_data.loc[start_play:play_index]

        return play_data
