import rosu_pp_py as rosu
import numpy as np
import matplotlib.pyplot as plt

from classes.ExternalCalculator import ExternalCalculator

class BeatmapAnalyze:
    """
    Class to analyze osu! beatmaps and scores.
    Usage:

        beatmap_analyze = BeatmapAnalyze(map_file_path="path/to/beatmap.osu", mods=0)
        beatmap_data = beatmap_analyze.load_beatmap()  # Loads the beatmap and calculates its properties
        json_data = beatmap_analyze.to_json()  # Returns a dictionary containing the analyzed data
        beatmap_analyze.save_to_json("path/to/save/beatmap.json")  # Saves the analyzed data to a JSON file

    """

    def __init__(self, map_file_path, mods: int = 0):
        self.map_file_path = map_file_path
        self.mods = mods

    @property
    def mods(self):
        return self._mods

    @mods.setter
    def mods(self, value):
        if isinstance(value, str):
            if not all(mod.strip().upper() in ["NM", "EZ", "HD", "HR", "SD", "DT", "RX", "HT", "NC", "FL", "SO", "AP", "PF"] for mod in value.split(",")):
                raise ValueError("Invalid mods string. Allowed mods are: NM, EZ, HD, HR, SD, DT, RX, HT, NC, FL, SO, AP, PF.")
        elif isinstance(value, int):
            pass  # Assuming the integer is a valid mod representation
        else:
            raise ValueError("Mods must be a string or an integer.")

        self._mods = value

    @staticmethod
    def mods_to_int(mods_str):
        """
        Convert a string of mods to an integer representation.
        """
        mod_mapping = {
            "NM": 0,
            "EZ": 2,
            "HD": 8,
            "HR": 16,
            "SD": 32,
            "DT": 64,
            "RX": 128,
            "HT": 256,
            "NC": 512,  # NC includes DT
            "FL": 1024,
            "SO": 4096,
            "AP": 8192,
            "PF": 16384,  # PF includes SD
        }

        mods_int = 0
        for mod in mods_str.split(","):
            mod = mod.strip().upper()
            if mod in mod_mapping:
                mods_int |= mod_mapping[mod]
        return mods_int

    
    @staticmethod
    def resample_strain(strain_array, target_length=200):
        original_indices = np.linspace(0, 1, len(strain_array))
        target_indices = np.linspace(0, 1, target_length)
        return np.interp(target_indices, original_indices, strain_array).tolist()

    def load_beatmap(self):
        # Load the beatmap using rosu_pp_py
        map = rosu.Beatmap(path=self.map_file_path)
        diff = rosu.Difficulty(mods=self.mods)

        attrs = diff.calculate(map)

        external_calculator = ExternalCalculator(base_cs=map.cs, base_od=map.od, mods=self.mods)
        external_calculator.calculate()

        ar = attrs.ar
        sr = attrs.stars
        cs = external_calculator.get_cs()
        od = external_calculator.get_od()
        hp = attrs.hp

        maximum_calc = rosu.Performance(mods=self.mods, accuracy=100.0, misses=0)
        maximum_pp = maximum_calc.calculate(map).pp

        strains = diff.strains(map)
        aim_section_peaks = strains.aim
        speed_section_peaks = strains.speed
        flashlight_section_peaks = strains.flashlight

        return {
            "map": map,
            "ar": ar,
            "sr": sr,
            "cs": cs,
            "od": od,
            "hp": hp,
            "maximum_pp": maximum_pp,
            "aim_section_peaks": aim_section_peaks,
            "speed_section_peaks": speed_section_peaks,
            "flashlight_section_peaks": flashlight_section_peaks
        }

    def to_json(self):
        beatmap_data = self.load_beatmap()

        aim_section_peaks = beatmap_data["aim_section_peaks"]
        speed_section_peaks = beatmap_data["speed_section_peaks"]
        flashlight_section_peaks = beatmap_data["flashlight_section_peaks"]

        data = {
            "map_info": {
                "beatmap_id": self.map_file_path.split("/")[-1].split(".")[0],  # Extracting beatmap ID from file name
                "cs": beatmap_data["cs"],
                "ar": beatmap_data["ar"],
                "od": beatmap_data["od"],
                "hp": beatmap_data["hp"],
                "sr": beatmap_data["sr"],
                "maximum_pp": beatmap_data["maximum_pp"],
            },
            "strains": {
                "aim_section_peaks": aim_section_peaks,
                "speed_section_peaks": speed_section_peaks,
                "flashlight_section_peaks": flashlight_section_peaks,
            }
        }

        return data

    def save_to_json(self, file_path):
        data = self.to_json()
        with open(file_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(data, f)


class ScoreAnalyze(BeatmapAnalyze):
    """
    Một lớp con của BeatmapAnalyze để phân tích điểm số của người chơi trên một beatmap cụ thể.
    Usage:

        score_analyze = ScoreAnalyze(map_file_path="path/to/beatmap.osu", mods=0, score_accuracy=98.5, score_misses=2, n_300=500, n_100=20, n_50=5, combo=800)
        score_data = score_analyze.load_score()  # Loads the beatmap and calculates the score properties
        json_data = score_analyze.to_json()  # Returns a dictionary containing the analyzed data
        score_analyze.save_to_json("path/to/save/score.json")  # Saves the analyzed data to a JSON file

    """

    def __init__(self, map_file_path, mods: int = 0, score_accuracy: float = 100.0, score_misses: int = 0, n_300: int = 0, n_100: int = 0, n_50: int = 0, combo: int = 0):
        super().__init__(map_file_path, mods)
        self.score_accuracy = score_accuracy
        self.score_misses = score_misses
        self.n_300 = n_300
        self.n_100 = n_100
        self.n_50 = n_50
        self.combo = combo

    def load_score(self):
        beatmap_data = self.load_beatmap()
        
        performance_calc = rosu.Performance(mods=self.mods, 
                                            accuracy=self.score_accuracy, 
                                            misses=self.score_misses, 
                                            combo=self.combo, 
                                            n300=self.n_300, n100=self.n_100, n50=self.n_50
                                            )
        pp = performance_calc.calculate(beatmap_data["map"]).pp
        pp_ratio = pp / beatmap_data["maximum_pp"] if beatmap_data["maximum_pp"] > 0 else 0
        beatmap_data["score_info"] = {
            "accuracy": self.score_accuracy,
            "misses": self.score_misses,
            "n_300": self.n_300,
            "n_100": self.n_100,
            "n_50": self.n_50,
            "combo": self.combo,
            "pp": pp,
            "pp_ratio": pp_ratio
        }

        return beatmap_data


    def to_json(self):
        beatmap_data = self.load_score()

        aim_section_peaks = beatmap_data["aim_section_peaks"]
        speed_section_peaks = beatmap_data["speed_section_peaks"]
        flashlight_section_peaks = beatmap_data["flashlight_section_peaks"]

        data = {
            "map_info": {
                "beatmap_id": self.map_file_path.split("/")[-1].split(".")[0],  # Extracting beatmap ID from file name
                "cs": beatmap_data["cs"],
                "ar": beatmap_data["ar"],
                "od": beatmap_data["od"],
                "hp": beatmap_data["hp"],
                "sr": beatmap_data["sr"],
                "mods_id": self.mods,
                "maximum_pp": beatmap_data["maximum_pp"],
            },
            "score_info": {
                "accuracy": self.score_accuracy,
                "misses": self.score_misses if self.score_misses is not None else 0,
                "n_300": self.n_300 if self.n_300 is not None else 0,
                "n_100": self.n_100 if self.n_100 is not None else 0,
                "n_50": self.n_50 if self.n_50 is not None else 0,
                "combo": self.combo,
                "pp": beatmap_data["score_info"]["pp"],
                "pp_ratio": beatmap_data["score_info"]["pp_ratio"]
            },
            "strains": {
                "aim_section_peaks": aim_section_peaks,
                "speed_section_peaks": speed_section_peaks,
                "flashlight_section_peaks": flashlight_section_peaks,
            }
        }

        return data