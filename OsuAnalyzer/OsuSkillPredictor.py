import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

from OsuAnalyzer.BeatmapAnalyze import BeatmapAnalyze
from OsuAnalyzer.BeatmapCollect import BeatmapCollect
from OsuAnalyzer.EarlyStopper import EarlyStopper
from OsuAnalyzer.OsuSkillDataset import UserSkillDataset

class OsuSkillPredictorModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Branch 1: 1D CNN for the 3x200 Strain Arrays
        # Input shape: [batch_size, channels=3, length=200]
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces length to 100
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces length to 50
            
            nn.Flatten(),
            nn.Linear(32 * 50, 64),
            nn.ReLU()
        )
        
        # Branch 2: Feed-forward for the 10 Scalars (SR, CS, OD, AR, HP, Maximum PP, Is EZ, Is HD, Is HR, Is DT)
        self.scalar_branch = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU()
        )
        
        # Combined Head: Takes CNN output (64) + Scalar output (16) = 80
        self.regression_head = nn.Sequential(
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3), # Prevent overfitting
            nn.Linear(32, 1)   # Output: Single PP ratio float
        )

    def forward(self, x_scalars, x_strains):
        # x_scalars shape: [batch, 10]
        # x_strains shape: [batch, 3, 200]
        
        cnn_features = self.cnn_branch(x_strains)
        scalar_features = self.scalar_branch(x_scalars)
        
        combined = torch.cat((cnn_features, scalar_features), dim=1)
        ratio = self.regression_head(combined)
        
        return ratio

class OsuSkillPredictor:
    """
    A class that encapsulates the OsuSkillPredictorModel and provides methods for training, saving, loading, and predicting.
    You could create a dataset like this:
    Usage for dataset preparation:

        USERNAME = "BiaHaNoi"  # Replace with the desired username
        top_play_dataset = TopPlayDataset(username=USERNAME)
        with open(f"top_plays_{USERNAME}.json", "w") as f:
            json_data = top_play_dataset.analyze_top_plays()
            json.dump(json_data, f, indent=4)   

    Usage for training:

        osu_skill_dataset = OsuSkillDataset(scalars=[], strains=[], ratios=[])
        osu_skill_dataset.read_from_json("top_plays_Seamie.json")
        dataloader = DataLoader(osu_skill_dataset, batch_size=2, shuffle=True)

        model = OsuSkillPredictor()
        model.train_model(dataloader, epochs=100)
        model.save_model("osu_skill_predictor.pth")

    Usage for loading and predicting:

        model = OsuSkillPredictor()
        model.load_model("osu_skill_predictor.pth")
        prediction = model.predict(beatmap_id="5159668")  # Replace with the desired beatmap ID
        print(f"Predicted pp ratio: {prediction.item():.4f}")

    """

    def __init__(self):
        self.model = OsuSkillPredictorModel()

    def train_model(self, dataloader, epochs=50):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        early_stopper = EarlyStopper(patience=3, min_delta=10)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_scalars, batch_strains, batch_ratios in dataloader:
                # Move data to the RTX 3050
                batch_scalars = batch_scalars.to(device)
                batch_strains = batch_strains.to(device)
                batch_ratios = batch_ratios.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(batch_scalars, batch_strains)
                
                # Calculate loss and backpropagate
                loss = criterion(predictions, batch_ratios)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            if early_stopper.early_stop(avg_loss):
                print("Early stopping triggered.")
                break
            # if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def validate_model(self, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        model.eval()
        
        criterion = nn.MSELoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_scalars, batch_strains, batch_ratios in dataloader:
                # Move data to the RTX 3050
                batch_scalars = batch_scalars.to(device)
                batch_strains = batch_strains.to(device)
                batch_ratios = batch_ratios.to(device)
                
                # Forward pass
                predictions = model(batch_scalars, batch_strains)
                
                # Calculate loss
                loss = criterion(predictions, batch_ratios)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.model.loaded = True  # Mark the model as loaded

    def predict(self, beatmap_id, mods=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        model = self.model.to(device)
        model.eval()

        #If model is not loaded
        if not hasattr(self.model, 'loaded') or not self.model.loaded:
            return Exception("Model is not loaded. Please load the model with .load_model(file_path) before making predictions.")

        # If no beatmaps/ dir
        if not os.path.exists("beatmaps"):
            os.makedirs("beatmaps")

        # If map don't exist
        if not os.path.exists(f"beatmaps/{beatmap_id}.osu"):
            beatmap_collect = BeatmapCollect(beatmap_id)
            beatmap_collect.load_beatmap()
            beatmap_collect.save_to_file(f"beatmaps/{beatmap_id}.osu")

        beatmap_analyze = BeatmapAnalyze(map_file_path=f"beatmaps/{beatmap_id}.osu", mods=mods)
        beatmap_analyze.load_beatmap()  # Loads the beatmap and calculates its properties
        json_data = beatmap_analyze.to_json()  # Returns a dictionary containing the analyzed data

        print(json_data["map_info"])

        cs = json_data["map_info"]["cs"]
        od = json_data["map_info"]["od"]
        ar = json_data["map_info"]["ar"]
        hp = json_data["map_info"]["hp"]
        sr = json_data["map_info"]["sr"]
        mods_id = mods
        maximum_pp = json_data["map_info"]["maximum_pp"]
        aim_strain_peaks = json_data["strains"]["aim_section_peaks"]
        speed_strain_peaks = json_data["strains"]["speed_section_peaks"]
        flashlight_strain_peaks = json_data["strains"]["flashlight_section_peaks"]

        userSkillData = UserSkillDataset(cs, od, ar, hp, sr, mods_id, maximum_pp, aim_strain_peaks, speed_strain_peaks, flashlight_strain_peaks, pp_ratio=0.0)

        x_scalars = torch.tensor(np.array([userSkillData.X_scalars]), dtype=torch.float32)
        x_strains = torch.tensor(np.array([userSkillData.X_strains]), dtype=torch.float32)

        with torch.no_grad():
            x_scalars = x_scalars.to(device)
            x_strains = x_strains.to(device)
            predictions = model(x_scalars, x_strains)
            confidence = torch.sigmoid(predictions)  # Assuming the output is a logit, apply sigmoid to get a confidence score

        return predictions, confidence

# Train the network defined in the previous step
# model = OsuSkillPredictorModel()
# train_model(model, dataloader)