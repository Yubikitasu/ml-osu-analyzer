import json
import os

import requests

from classes.BeatmapCollect import BeatmapCollect
from classes.BeatmapAnalyze import ScoreAnalyze

from tqdm import tqdm

class TopPlayDataset:

    """
    Main class to analyze the top plays of a user.
    Usage:

        USERNAME = "peppy"  # Replace with the desired username
        top_play_dataset = TopPlayDataset(username=USERNAME)
        json_data = top_play_dataset.analyze_top_plays() # Returns a list of dictionaries containing the analyzed data for each top play.
        top_play_dataset.save_to_json("path/to/save/top_plays.json")  # Saves the analyzed data to a JSON file
        
    """

    def __init__(self, username):
        self.username = username

    def __fetch_user_data(self):
        """
        Fetch user data from the osu! API using the provided username.
        Raises an exception if the request fails.
        """
        request_url = f"https://maru-api.vercel.app/users/{self.username}/osu"
        response = requests.get(request_url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to load user data for username {self.username}. Status code: {response.status_code}")

    def fetch_top_plays(self):
        """
        Fetch the top plays of the user from the osu! API.
        Raises an exception if the request fails.
        """
        self.__fetch_user_data()
        user_data = self.__fetch_user_data()
        user_id = user_data.get("id")
        if not user_id:
            raise Exception(f"User ID not found for username {self.username}.")

        gameMode = "osu"  # Replace with the desired game mode
        request_url = f"https://maru-api.vercel.app/users/{user_id}/best/{gameMode}"
        response = requests.get(request_url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to load top plays for user ID {user_id}. Status code: {response.status_code}")

    def analyze_top_plays(self):
        """
        Analyze the top plays of the user and return a summary.
        Raises an exception if the request fails.
        """
        ROOT_DIR = "top_plays_data"
        plays_data_array = []
    
        if not os.path.exists(ROOT_DIR):
            os.makedirs(ROOT_DIR)

        top_plays = self.fetch_top_plays()
        if not top_plays:
            raise Exception(f"No top plays found for username {self.username}.")
        for play in tqdm(top_plays):
            if not os.path.exists(f"{ROOT_DIR}/{play['beatmap']['id']}.osu"):
                beatmap_collect = BeatmapCollect(beatmap_id=play["beatmap"]["id"])
                beatmap_collect.load_beatmap()
                beatmap_collect.save_to_file(f"{ROOT_DIR}/{play['beatmap']['id']}.osu")

            beatmap_analyze = ScoreAnalyze(map_file_path=f"{ROOT_DIR}/{play['beatmap']['id']}.osu", 
                                           mods=play["mods_id"], 
                                           score_accuracy=play["score_info"]["accuracy"], 
                                           score_misses=play["score_info"]["n_miss"], 
                                           n_300=play["score_info"]["n_300"], 
                                           n_100=play["score_info"]["n_100"], 
                                           n_50=play["score_info"]["n_50"], 
                                           combo=play["score_info"]["max_combo"])
            beatmap_data = beatmap_analyze.to_json()
            plays_data_array.append(beatmap_data)

            # print(f"Processed beatmap ID: {play['beatmap']['id']} for user {self.username}")

        return plays_data_array