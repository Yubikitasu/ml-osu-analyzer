import json
import torch
from torch.utils.data import DataLoader
import os

from OsuAnalyzer.BeatmapAnalyze import BeatmapAnalyze
from OsuAnalyzer.OsuBasicPredictor import OsuBasicPredictor
from OsuAnalyzer.TopPlayDataset import TopPlayDataset
from OsuAnalyzer.OsuSkillDataset import OsuSkillDataset
from OsuAnalyzer.OsuSkillPredictor import OsuSkillPredictor

class OsuSkillPipeline:
    def __init__(self, basic_model_path="osu_basic_predictor.pkl", nn_model_path="osu_skill_predictor.pth"):
        self.basic_model_path = basic_model_path
        self.nn_model_path = nn_model_path
        self.basic_predictor = OsuBasicPredictor()
        self.model = OsuSkillPredictor()
        
        self.dataset = None
        self.dataloader = None
        self.min_sr = None
        self.max_sr = None

        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

        if not os.path.exists("models"):
            os.makedirs("models")

    def fetch_top_plays(self, username):
        """Fetches top plays from osu! API and saves them to a JSON file."""
        top_play_dataset = TopPlayDataset(username=username)
        top_plays_data = top_play_dataset.analyze_top_plays()
        
        output_file = f"top_plays_{username}.json"
        with open(output_file, 'w') as f:
            json.dump(top_plays_data, f, indent=4)
            
        return output_file

    def load_dataset(self, json_path, batch_size=1, shuffle=True):
        """Loads the parsed JSON data into PyTorch Datasets and DataLoaders."""
        self.dataset = OsuSkillDataset(scalars=[], strains=[], ratios=[])
        self.dataset.read_from_json(json_path)
        # self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle)

        all_srs = [item[4] for item in self.dataset.scalars] # Tùy chỉnh theo biến của bạn
        self.min_sr = min(all_srs)
        self.max_sr = max(all_srs)

    def train_models(self, epochs=100):
        """Trains both the Basic Predictor (Isolation Forest) and the 1D-CNN."""
        if self.dataset is None or self.train_dataloader is None or self.val_dataloader is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        # Train and save the basic anomaly detection model
        self.basic_predictor.train_model(self.dataset.X_train)
        self.basic_predictor.save_model("models/" + self.basic_model_path)
        
        # Train and save the deep learning model
        self.model.train_model(self.train_dataloader, epochs=epochs)
        self.model.validate_model(self.val_dataloader)
        self.model.save_model("models/" + self.nn_model_path)

    def predict_map(self, beatmap_id, mods_str="NM"):
        """Runs the two-stage prediction pipeline on a new beatmap."""
        if not os.path.exists("models/" + self.basic_model_path) or not os.path.exists("models/" + self.nn_model_path):
            raise Exception("Model files don't exist. Please train the models first.")

        mods_int = BeatmapAnalyze.mods_to_int(mods_str)
        
        # Stage 1: Check physical limits with Basic Predictor
        self.basic_predictor.load_model("models/" + self.basic_model_path)
        basic_prediction = self.basic_predictor.predict(beatmap_id=beatmap_id, mods=mods_int)
        map_sr = BeatmapAnalyze(map_file_path=f"beatmaps/{beatmap_id}.osu", mods=mods_int).load_beatmap()["sr"]
        
        if basic_prediction[0] == -1:
            if map_sr < self.min_sr:
                print(f"Map {beatmap_id} with mods {mods_str} is below the minimum skill requirement (Too Easy).")
                print("Predicted pp ratio: 1.0000 (Guaranteed SS/FC)")
                return 1.0
            else:
                print(f"Map {beatmap_id} with mods {mods_str} is above the maximum skill requirement (Too Hard).")
                print("Predicted pp ratio: 0.0000 (Fail/Low Acc)")
                return 0.00
            
        print(f"Basic Predictor: The player is predicted to be able to play the map {beatmap_id} with mods {mods_str}.")
        
        # Stage 2: Predict specific PP Ratio with CNN
        self.model.load_model("models/" + self.nn_model_path)
        prediction = self.model.predict(beatmap_id=beatmap_id, mods=mods_int)
        
        ratio, confidence = prediction
        print(f"Predicted pp ratio: {ratio.item():.4f} with confidence {confidence.item():.4f}")
        return ratio
