# osu! osu analyzer

An osu! beatmap analysis and player-skill prediction project. It analyzes beatmap difficulty and strain features, builds training data from a player's top plays, and predicts the expected pp ratio for a beatmap under selected mods.

## Features

- Parse `.osu` beatmaps with `rosu_pp_py`.
- Calculate star rating, CS, OD, AR, HP, maximum pp, and aim/speed/flashlight strain peaks.
- Download beatmaps and player top plays through the configured osu! API mirrors.
- Generate player-specific training data from top plays.
- Train a two-stage predictor:
  1. An Isolation Forest identifies maps outside the player's learned difficulty range.
  2. A PyTorch 1D CNN estimates the expected pp ratio for maps within that range.
- Use either the included Python entry point or the exploratory [`analyze.ipynb`](./analyze.ipynb) notebook.

## Project structure

| Path | Description |
| --- | --- |
| [`main.py`](./main.py) | Example pipeline entry point |
| [`OsuAnalyzer/`](./OsuAnalyzer/) | Beatmap collection, analysis, dataset, and model classes |
| [`beatmaps/`](./beatmaps/) | Cached beatmap files used for analysis |
| [`analyze.ipynb`](./analyze.ipynb) | Interactive analysis notebook |
| [`requirements.txt`](./requirements.txt) | Python dependencies |

## Requirements

- Python with a version compatible with the pinned packages in [`requirements.txt`](./requirements.txt)
- Network access when downloading new user or beatmap data
- A working PyTorch installation; training uses CUDA automatically when available

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Quick start

The default entry point uses the included `Seamie` dataset and predicts beatmap `2123604` with NM and DT:

```bash
python main.py
```

To use another player, edit `player` in [`main.py`](./main.py):

```python
from OsuAnalyzer.OsuSkillPipeline import OsuSkillPipeline

pipeline = OsuSkillPipeline()
player = "YourOsuUsername"

# Fetch and save the player's analyzed top plays.
json_file = pipeline.fetch_top_plays(player)

# Load the dataset and train models if new models are needed.
pipeline.load_dataset(json_file, batch_size=1)
# pipeline.train_models(epochs=100)

pipeline.predict_map(beatmap_id="2123604", mods_str="NM")
pipeline.predict_map(beatmap_id="2123604", mods_str="DT")
```

`predict_map` expects the saved models to exist in [`models/`](./models/). Uncomment `train_models` when creating models for a new dataset. Training saves:

- `models/osu_basic_predictor.pkl`
- `models/osu_skill_predictor.pth`

## Supported mods

Mod strings are comma-separated and case-insensitive. The analyzer accepts:

`NM`, `EZ`, `HD`, `HR`, `SD`, `DT`, `RX`, `HT`, `NC`, `FL`, `SO`, `AP`, and `PF`

For example:

```python
pipeline.predict_map("2123604", mods_str="HD,HR")
```

## Data flow

1. `TopPlayDataset` retrieves a user's top plays and caches each referenced beatmap.
2. `ScoreAnalyze` calculates beatmap features and the play's pp ratio.
3. `OsuSkillDataset` converts scalar features and resampled strain peaks into PyTorch tensors.
4. `OsuSkillPipeline.train_models` trains and saves both predictors.
5. `predict_map` downloads a missing beatmap, applies the requested mods, checks the learned range, and predicts a pp ratio.

The cached data and model files are local artifacts. Re-running collection may update them with newly fetched top-play data.

## Notes

- The project uses external API mirrors configured in [`BeatmapCollect.py`](./OsuAnalyzer/BeatmapCollect.py) and [`TopPlayDataset.py`](./OsuAnalyzer/TopPlayDataset.py).
- Predictions are estimates based on the training data and should not be treated as official osu! performance calculations.
- A new player's dataset should be collected and used to train models before making predictions for that player.
