import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

class UserSkillDataset:
    def __init__(self, cs, od, ar, hp, sr, maximum_pp, mods_id, aim_strain_peaks, speed_strain_peaks, flashlight_strain_peaks, pp_ratio):
        self.cs = cs
        self.od = od
        self.ar = ar
        self.hp = hp
        self.sr = sr
        self.maximum_pp = maximum_pp
        self.mods_id = mods_id
        self.aim_strain_peaks = aim_strain_peaks
        self.speed_strain_peaks = speed_strain_peaks
        self.flashlight_strain_peaks = flashlight_strain_peaks
        self.pp_ratio = pp_ratio

        self.is_ez = 1.0 if (int(self.mods_id) & 2) else 0.0
        self.is_hd = 1.0 if (int(self.mods_id) & 8) else 0.0
        self.is_hr = 1.0 if (int(self.mods_id) & 16) else 0.0
        self.is_dt = 1.0 if (int(self.mods_id) & 64) else 0.0

    @staticmethod
    def resample_strain(strain_array, target_length=200):
        original_indices = np.linspace(0, 1, len(strain_array))
        target_indices = np.linspace(0, 1, target_length)
        return np.interp(target_indices, original_indices, strain_array).tolist()

    @property
    def X_scalars(self):
        return [self.cs, self.od, self.ar, self.hp, self.sr, self.maximum_pp, self.is_ez, self.is_hd, self.is_hr, self.is_dt]

    @property
    def X_strains(self):
        return [self.resample_strain(self.aim_strain_peaks), self.resample_strain(self.speed_strain_peaks), self.resample_strain(self.flashlight_strain_peaks)]

    @property
    def y_ratios(self):
        return [self.pp_ratio]

    def X_concate_data(self):
        return np.array(self.X_scalars + self.X_strains[0] + self.X_strains[1] + self.X_strains[2])


class OsuSkillDataset(Dataset):
    def __init__(self, scalars, strains, ratios):
        # scalars: numpy array of shape (N, 10)
        # strains: numpy array of shape (N, 3, 200)
        # ratios: numpy array of shape (N, 1)
        self.scalars = torch.tensor(scalars, dtype=torch.float32)
        self.strains = torch.tensor(strains, dtype=torch.float32)
        self.ratios = torch.tensor(ratios, dtype=torch.float32)

        self.X_train = np.array([])
        self.scalars_list = []
        self.strains_list = []
        self.ratios_list = []
        
    def __len__(self):
        return len(self.ratios)
        
    def __getitem__(self, idx):
        return self.scalars[idx], self.strains[idx], self.ratios[idx]

    def read_from_json(self, json_file_path):
        X_train = []
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        for score in data:
            map_info = score['map_info']
            cs = map_info['cs']
            od = map_info['od']
            ar = map_info['ar']
            hp = map_info['hp']
            sr = map_info['sr']
            mods_id = map_info['mods_id']
            maximum_pp = map_info['maximum_pp']
            aim_strain_peaks = score['strains']['aim_section_peaks']
            speed_strain_peaks = score['strains']['speed_section_peaks']
            flashlight_strain_peaks = score['strains']['flashlight_section_peaks']
            pp_ratio = score['score_info']['pp_ratio']

            user_skill_data = UserSkillDataset(cs, od, ar, hp, sr, mods_id, maximum_pp, aim_strain_peaks, speed_strain_peaks, flashlight_strain_peaks, pp_ratio)

            self.scalars_list.append(user_skill_data.X_scalars)
            self.strains_list.append(user_skill_data.X_strains)
            self.ratios_list.append(user_skill_data.y_ratios)

            feature_vector = user_skill_data.X_scalars + user_skill_data.X_strains[0] + user_skill_data.X_strains[1] + user_skill_data.X_strains[2]
            X_train.append(feature_vector)

        self.scalars = torch.tensor(np.array(self.scalars_list), dtype=torch.float32)
        self.strains = torch.tensor(np.array(self.strains_list), dtype=torch.float32)
        self.ratios = torch.tensor(np.array(self.ratios_list), dtype=torch.float32)
        self.X_train = np.array(X_train)

            
