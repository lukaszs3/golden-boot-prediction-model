from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


KAGGLE_DATASETS = {
    "fifa_rankings": {
        "handle": "cashncarry/fifaworldranking",
        "folder": "cashncarry__fifaworldranking",
    },
    "world_cup_database": {
        "handle": "joshfjelstul/world-cup-database",
        "folder": "joshfjelstul__world-cup-database",
    },
    "world_cup_2022": {
        "handle": "swaptr/fifa-world-cup-2022-player-data",
        "folder": "swaptr__fifa-world-cup-2022-player-data",
    },
    "club_2017_2018": {
        "handle": "diegobartoli/top5legauesplayers-statsandphys",
        "folder": "diegobartoli__top5legauesplayers-statsandphys",
    },
    "club_2021_2022": {
        "handle": "vivovinco/20212022-football-player-stats",
        "folder": "vivovinco__20212022-football-player-stats",
    },
    "current_2025_2026": {
        "handle": "hubertsidorowicz/football-players-stats-2025-2026",
        "folder": "hubertsidorowicz__football-players-stats-2025-2026",
    },
    "current_rankings_2026": {
        "handle": "zkskhurram/fifa-and-football-complete-dataset-19302022",
        "folder": "zkskhurram__fifa-and-football-complete-dataset-19302022",
    },
}


FEATURE_COLUMNS = [
    "age",
    "fifa_rank",
    "club_minutes",
    "club_goals",
    "club_non_penalty_goals",
    "club_penalty_goals",
    "club_assists",
    "club_goals_per90",
    "league_goals_rank_pct",
    "league_non_penalty_goals_rank_pct",
    "team_goal_share",
    "position_is_defender",
    "position_is_midfielder",
    "position_is_forward",
]
