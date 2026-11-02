from OsuAnalyzer.BeatmapAnalyze import BeatmapAnalyze
from OsuAnalyzer.BeatmapCollect import BeatmapCollect
import os

from sklearn.ensemble import IsolationForest
import joblib

from OsuAnalyzer.OsuSkillDataset import UserSkillDataset

class OsuBasicPredictor:
    """
    Mô hình đơn giản dự đoán rằng liệu bạn có chơi được map hay không dựa trên các thông số của map và kỹ năng của người chơi.
    
    Sử dụng Isolation Forest để xác định xem dữ liệu đầu vào có phải là ngoại lệ hay không. Nếu là ngoại lệ, dự đoán rằng người chơi sẽ không chơi được map; nếu không, dự đoán rằng người chơi sẽ chơi được map.
    """

    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    def train_model(self, data):
        self.model.fit(data)

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)
        self.model.loaded = True

    def predict(self, beatmap_id, mods=0):
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

        prediction = self.model.predict([userSkillData.X_concate_data()])  # Reshape to 2D array for prediction
        return prediction