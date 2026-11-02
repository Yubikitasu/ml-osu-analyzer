import requests

REQUEST_URL_TEMPLATE = "https://mirror.hinamizawa.ai/api/osu/{beatmap_id}"

class BeatmapCollect:
    """
    Class to collect beatmap data from the osu! API and save it to a file.
    Usage:
        
        beatmap_collect = BeatmapCollect(beatmap_id=123456)
        beatmap_collect.load_beatmap()  # Fetches the beatmap data from the API
        beatmap_collect.save_to_file("path/to/save/beatmap.osu")  # Saves the data to a file
        
    """

    def __init__(self, beatmap_id):
        self.beatmap_id = beatmap_id
        self.beatmap_data = None

    def load_beatmap(self):
        """
        Load beatmap data from the osu! API using the provided beatmap ID.
        Raises an exception if the request fails.
        """

        request_url = REQUEST_URL_TEMPLATE.format(beatmap_id=self.beatmap_id)
        response = requests.get(request_url)
        if response.status_code == 200:
            self.beatmap_data = response.text
        else:
            raise Exception(f"Failed to load beatmap data for ID {self.beatmap_id}. Status code: {response.status_code}")

        return self.beatmap_data

    def save_to_file(self, file_path):
        """
        Save the loaded beatmap data to a file.
        Raises an exception if the beatmap data is not loaded.
        """
        if self.beatmap_data is None:
            raise Exception("Beatmap data is not loaded. Call load_beatmap() first.")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.beatmap_data)
            f.close()            