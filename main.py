from classes.OsuSkillPipeline import OsuSkillPipeline

# Example usage:
if __name__ == "__main__":
    pipeline = OsuSkillPipeline()

    player = "Seamie"

    # Fetch top plays and save to JSON
    json_file = pipeline.fetch_top_plays(player)

    # # Load existing data, train, and predict
    pipeline.load_dataset(f"top_plays_{player}.json", batch_size=1)

    # Train the models (both Isolation Forest and 1D-CNN)
    # MUST train the models first run before making predictions and or switching between players, otherwise the models will not be loaded and will throw an error.

    # pipeline.train_models(epochs=50)

    print(f"------ Predictions of player {player} ------")
    pipeline.predict_map(beatmap_id="2123604", mods_str="NM")
    pipeline.predict_map(beatmap_id="2123604", mods_str="DT")