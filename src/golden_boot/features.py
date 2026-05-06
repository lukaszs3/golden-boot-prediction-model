from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ALLOWED_CANDIDATE_COUNTRY_CODES,
    FEATURE_COLUMNS,
    KAGGLE_DATASETS,
    RAW_DIR,
)


COUNTRY_FIXES = {
    "usa": "united states",
    "ir iran": "iran",
    "korea republic": "south korea",
    "turkiye": "turkey",
}

def build_training_table() -> pd.DataFrame:
    """
    Build training table from current 2025/26 player data and rankings.
    Uses active player performance stats as both features and synthetic target variable.
    """
    rankings = load_current_rankings()

    # Load current 2025/26 player features
    folder = dataset_dir("current_2025_2026")
    path = folder / "players_data-2025_2026.csv"
    if not path.exists():
        path = folder / "players_data_light-2025_2026.csv"

    df = pd.read_csv(path)

    # Build base player data
    players = pd.DataFrame(
        {
            "player": df["Player"],
            "player_key": df["Player"].astype(str).str.lower().str.strip(),
            "country_code": df["Nation"].astype(str).str[-3:].str.upper(),
            "club_team": df["Squad"],
            "league_name": df["Comp"],
            "position": df["Pos"],
            "age": df["Age"],
            "club_minutes": df["Min"],
            "club_goals": df["Gls"],
            "club_non_penalty_goals": df["G-PK"],
            "club_assists": df["Ast"],
        }
    )

    # Filter to only include top-30 ranked countries
    players = players.merge(rankings[["country_code", "rank"]], on="country_code", how="inner")

    # Filter out goalkeepers
    players = players[~players["position"].astype(str).str.upper().str.contains("GK", na=False)]

    # Process player stats
    players = finish_players(players)
    players = add_competition_context_features(players)

    # Aggregate by player
    players_agg = aggregate_player_rows(
        players,["player", "player_key", "country_code", "position", "rank"]
    )

    #Syntetyczny zbior labels
    players_agg["target_goals"] = (
        (players_agg["club_goals"] * players_agg["league_goals_rank_pct"]) / 10.0
    ).round(0).astype(int)

    # Add features
    players_agg = add_position_features(players_agg)
    players_agg["fifa_rank"] = players_agg["rank"]
    players_agg["source_tournament"] = "2025-2026 Season"

    # Select and return relevant columns (avoiding duplicates)
    result_cols = [
        "source_tournament",
        "player",
        "country_code",
        "position",
        "target_goals",
        *FEATURE_COLUMNS,
    ]

    return players_agg[result_cols].reset_index(drop=True)


def dataset_dir(dataset_key: str) -> Path:
    return RAW_DIR / KAGGLE_DATASETS[dataset_key]["folder"]


def load_fifa_snapshot(event_date: str) -> pd.DataFrame:
    files = sorted(dataset_dir("fifa_rankings").glob("fifa_ranking-*.csv"))
    rankings = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"], errors="coerce")
    snapshot_date = rankings.loc[rankings["rank_date"] <= pd.Timestamp(event_date), "rank_date"].max()
    snapshot = rankings[rankings["rank_date"] == snapshot_date].copy()
    snapshot["country_key"] = (
        snapshot["country_full"].astype(str).str.lower().str.strip().replace(COUNTRY_FIXES)
    )
    snapshot["country_code"] = snapshot["country_abrv"].astype(str).str.upper()
    snapshot["rank"] = pd.to_numeric(snapshot["rank"], errors="coerce")
    return snapshot[["country_full", "country_key", "country_code", "rank"]].drop_duplicates(
        "country_code"
    ).sort_values("rank")


def load_current_rankings() -> pd.DataFrame:
    ranking_path = dataset_dir("current_rankings_2026") / "fifa_world_rankings_jan_2026.csv"
    if ranking_path.exists():
        latest = load_fifa_snapshot("2100-01-01")[["country_key", "country_code"]]
        current = pd.read_csv(ranking_path)
        current["country_key"] = (
            current["Country"].astype(str).str.lower().str.strip().replace(COUNTRY_FIXES)
        )
        current = current.merge(latest, on="country_key", how="left")
        current["rank"] = pd.to_numeric(current["Rank"], errors="coerce")
        current["country"] = current["Country"]
        return current[["country", "country_code", "rank"]].dropna(subset=["country_code"]).head(30)

    latest = load_fifa_snapshot("2100-01-01").head(30)
    return latest.rename(columns={"country_full": "country"})[["country", "country_code", "rank"]]


def finish_players(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["club_non_penalty_goals"] = pd.to_numeric(
        df.get("club_non_penalty_goals", df["club_goals"]),
        errors="coerce",
    ).fillna(0.0)
    df["club_penalty_goals"] = (
        pd.to_numeric(df["club_goals"], errors="coerce").fillna(0.0) - df["club_non_penalty_goals"]
    ).clip(lower=0.0)
    df["club_goals_per90"] = (
        df["club_goals"] / (df["club_minutes"].replace(0, np.nan) / 90.0)
    ).fillna(0.0)
    return df


def add_competition_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["league_name"] = df["league_name"].fillna("unknown").astype(str)
    df["club_team"] = df["club_team"].fillna("unknown").astype(str)

    league_size = df.groupby("league_name")["player_key"].transform("count")
    goals_rank = df.groupby("league_name")["club_goals"].rank(method="dense", ascending=False)
    npg_rank = df.groupby("league_name")["club_non_penalty_goals"].rank(
        method="dense",
        ascending=False,
    )

    denominator = (league_size - 1).replace(0, np.nan)
    df["league_goals_rank_pct"] = (1.0 - ((goals_rank - 1.0) / denominator)).fillna(1.0)
    df["league_non_penalty_goals_rank_pct"] = (
        1.0 - ((npg_rank - 1.0) / denominator)
    ).fillna(1.0)

    team_goals = df.groupby(["league_name", "club_team"])["club_goals"].transform("sum")
    df["team_goal_share"] = (df["club_goals"] / team_goals.replace(0, np.nan)).fillna(0.0)
    return df


def aggregate_player_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    def weighted_average(frame: pd.DataFrame, column: str) -> float:
        values = pd.to_numeric(frame[column], errors="coerce")
        weights = pd.to_numeric(frame["club_minutes"], errors="coerce").fillna(0.0)
        valid = values.notna()
        if not valid.any():
            return 0.0
        values = values[valid]
        weights = weights[valid]
        if float(weights.sum()) <= 0.0:
            return float(values.mean())
        return float(np.average(values, weights=weights))

    aggregated_rows: list[dict] = []
    for keys, frame in df.groupby(group_cols, dropna=False, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        row["age"] = float(pd.to_numeric(frame["age"], errors="coerce").max())
        row["club_minutes"] = float(pd.to_numeric(frame["club_minutes"], errors="coerce").sum())
        row["club_goals"] = float(pd.to_numeric(frame["club_goals"], errors="coerce").sum())
        row["club_non_penalty_goals"] = float(
            pd.to_numeric(frame["club_non_penalty_goals"], errors="coerce").sum()
        )
        row["club_penalty_goals"] = float(
            pd.to_numeric(frame["club_penalty_goals"], errors="coerce").sum()
        )
        row["club_assists"] = float(pd.to_numeric(frame["club_assists"], errors="coerce").sum())
        row["league_goals_rank_pct"] = weighted_average(frame, "league_goals_rank_pct")
        row["league_non_penalty_goals_rank_pct"] = weighted_average(
            frame,
            "league_non_penalty_goals_rank_pct",
        )
        row["team_goal_share"] = weighted_average(frame, "team_goal_share")
        aggregated_rows.append(row)

    return finish_players(pd.DataFrame(aggregated_rows))


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    position_text = df["position"].fillna("").astype(str).str.upper()
    df["position_is_defender"] = position_text.str.contains(r"\bDF\b").astype(float)
    df["position_is_midfielder"] = position_text.str.contains(r"\bMF\b").astype(float)
    df["position_is_forward"] = position_text.str.contains(r"\b(?:FW|ST|CF|SS|LW|RW)\b").astype(float)
    return df



def load_current_2026_features() -> pd.DataFrame:
    folder = dataset_dir("current_2025_2026")
    path = folder / "players_data-2025_2026.csv"
    if not path.exists():
        path = folder / "players_data_light-2025_2026.csv"

    df = pd.read_csv(path)
    club = pd.DataFrame(
        {
            "player": df["Player"],
            "player_key": df["Player"].astype(str).str.lower().str.strip(),
            "country_code": df["Nation"].astype(str).str[-3:].str.upper(),
            "club_team": df["Squad"],
            "league_name": df["Comp"],
            "position": df["Pos"],
            "age": df["Age"],
            "club_minutes": df["Min"],
            "club_goals": df["Gls"],
            "club_non_penalty_goals": df["G-PK"],
            "club_assists": df["Ast"],
        }
    )
    club = finish_players(club)
    club = add_competition_context_features(club)
    return aggregate_player_rows(club, ["player", "player_key", "country_code", "position"])







def build_current_candidates(
    min_minutes: int = 300,
    max_per_country: int = 10,
) -> pd.DataFrame:
    rankings = load_current_rankings()
    rows = load_current_2026_features()
    rows = rows.merge(rankings, on="country_code", how="inner")
    rows = rows[rows["country_code"].isin(ALLOWED_CANDIDATE_COUNTRY_CODES)]
    rows = rows[rows["club_minutes"] >= float(min_minutes)]
    rows = rows[rows["position"].astype(str).str.upper().str.contains(r"\b(?:FW|ST|CF|SS|LW|RW)\b")]
    rows["fifa_rank"] = rows["rank"]
    rows = add_position_features(rows)

    if max_per_country > 0:
        rows = (
            rows.sort_values(["country_code", "club_goals", "club_minutes"], ascending=[True, False, False])
            .groupby("country_code", group_keys=False)
            .head(max_per_country)
        )

    return rows[["player", "country", "country_code", "position", *FEATURE_COLUMNS]].sort_values(
        ["country", "player"]
    )
