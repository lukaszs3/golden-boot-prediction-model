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


def dataset_dir(dataset_key: str) -> Path:
    return RAW_DIR / KAGGLE_DATASETS[dataset_key]["folder"]

#Loads FIFA world rankings for a specific event date by retrieving the most recent ranking snapshot on or before that date.
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

#For example mbappe was playing in 2 different clubs
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


#Loading club stats for 2017 and 2021 seasons

def load_club_2017_features() -> pd.DataFrame:
    """Load club stats for the 2017-18 season (used for 2018 WC training)."""
    country_codes = load_fifa_snapshot("2100-01-01")[["country_key", "country_code"]]
    frames: list[pd.DataFrame] = []

    for path in sorted((dataset_dir("club_2017_2018") / "Football_Players").glob("2017*.json")):
        raw = pd.read_json(path, lines=True)
        general = pd.json_normalize(raw["general_stats"])
        offensive = pd.json_normalize(raw["offensive_stats"])
        frame = pd.DataFrame(
            {
                "player_key": raw["name"].astype(str).str.lower().str.strip(),
                "country_key": raw["nationality"].astype(str).str.lower().str.strip().replace(
                    COUNTRY_FIXES
                ),
                "club_team": raw["team"],
                "league_name": path.stem,
                "age": raw["age"],
                "club_minutes": general["time"],
                "club_goals": offensive["goals"],
                "club_non_penalty_goals": offensive["npg"],
                "club_assists": offensive["assists"],
            }
        )
        frames.append(frame)

    club = pd.concat(frames, ignore_index=True)
    club = club.merge(country_codes, on="country_key", how="left")
    club = finish_players(club)
    club = add_competition_context_features(club)
    club = aggregate_player_rows(club, ["player_key", "country_code"])
    club["position"] = "FW"
    return club


def load_club_2021_features() -> pd.DataFrame:
    """Load club stats for the 2021-22 season (used for 2022 WC training)."""
    df = pd.read_csv(
        dataset_dir("club_2021_2022") / "2021-2022 Football Player Stats.csv",
        encoding="latin1",
        sep=";",
    )
    club = pd.DataFrame(
        {
            "player_key": df["Player"].astype(str).str.lower().str.strip(),
            "country_code": df["Nation"].astype(str).str[-3:].str.upper(),
            "club_team": df["Squad"],
            "league_name": df["Comp"],
            "position": df["Pos"],
            "age": df["Age"],
            "club_minutes": df["Min"],
            "club_goals": df["Goals"] * df["90s"],
            "club_non_penalty_goals": (df["Goals"] - df["ShoPK"]) * df["90s"],
            "club_assists": df["Assists"] * df["90s"],
        }
    )
    club = finish_players(club)
    club = add_competition_context_features(club)
    club = aggregate_player_rows(club, ["player_key", "country_code"])
    club["position"] = "FW"
    return club


#Building data for 2018 and 2022 world cups from scratch.

def build_2018_training_rows() -> pd.DataFrame:
    """Build training rows from the 2018 World Cup with real goal labels."""
    rankings = load_fifa_snapshot("2018-06-14").head(30)
    squads = pd.read_csv(dataset_dir("world_cup_database") / "squads.csv")
    players = pd.read_csv(dataset_dir("world_cup_database") / "players.csv")
    goals = pd.read_csv(dataset_dir("world_cup_database") / "goals.csv")

    rows = squads[squads["tournament_id"] == "WC-2018"].copy()
    rows = rows[rows["team_code"].isin(rankings["country_code"])]
    rows = rows[rows["position_code"].astype(str).str.upper() != "GK"]
    rows["player"] = (rows["given_name"].fillna("") + " " + rows["family_name"].fillna("")).str.strip()
    rows.loc[rows["player"] == "", "player"] = rows["family_name"].fillna("")
    rows["player_key"] = rows["player"].astype(str).str.lower().str.strip()
    rows["country_code"] = rows["team_code"]
    rows["country"] = rows["team_name"]
    rows["position"] = rows["position_code"]

    # Count actual goals scored at the 2018 World Cup (excluding own goals)
    goal_counts = (
        goals[(goals["tournament_id"] == "WC-2018") & (goals["own_goal"] == 0)]
        .groupby("player_id")
        .size()
        .rename("target_goals")
        .reset_index()
    )
    rows = rows.merge(players[["player_id", "birth_date"]], on="player_id", how="left")
    rows = rows.merge(goal_counts, on="player_id", how="left")

    # Merge pre-tournament club stats from the 2017-18 season
    rows = rows.merge(
        load_club_2017_features()[[
            "player_key",
            "country_code",
            "club_minutes",
            "club_goals",
            "club_non_penalty_goals",
            "club_penalty_goals",
            "club_assists",
            "club_goals_per90",
            "league_goals_rank_pct",
            "league_non_penalty_goals_rank_pct",
            "team_goal_share",
        ]],
        on=["player_key", "country_code"],
        how="left",
    )
    rows["age"] = (
        pd.Timestamp("2018-06-14") - pd.to_datetime(rows["birth_date"], errors="coerce")
    ).dt.days / 365.25
    rows["fifa_rank"] = rows["country_code"].map(rankings.set_index("country_code")["rank"])
    rows["target_goals"] = rows["target_goals"].fillna(0.0)
    rows[[
        "club_minutes",
        "club_goals",
        "club_non_penalty_goals",
        "club_penalty_goals",
        "club_assists",
        "club_goals_per90",
        "league_goals_rank_pct",
        "league_non_penalty_goals_rank_pct",
        "team_goal_share",
    ]] = rows[[
        "club_minutes",
        "club_goals",
        "club_non_penalty_goals",
        "club_penalty_goals",
        "club_assists",
        "club_goals_per90",
        "league_goals_rank_pct",
        "league_non_penalty_goals_rank_pct",
        "team_goal_share",
    ]].fillna(0.0)
    rows["source_tournament"] = "2018 World Cup"
    rows = add_position_features(rows)
    return rows[
        [
            "source_tournament",
            "player",
            "country",
            "country_code",
            "position",
            "target_goals",
            *FEATURE_COLUMNS,
        ]
    ]


def build_2022_training_rows() -> pd.DataFrame:
    """Build training rows from the 2022 World Cup with real goal labels."""
    rankings = load_fifa_snapshot("2022-11-20").head(30)
    club = load_club_2021_features()[
        [
            "player_key",
            "country_code",
            "club_minutes",
            "club_goals",
            "club_non_penalty_goals",
            "club_penalty_goals",
            "club_assists",
            "club_goals_per90",
            "league_goals_rank_pct",
            "league_non_penalty_goals_rank_pct",
            "team_goal_share",
        ]
    ]
    rows = pd.read_csv(dataset_dir("world_cup_2022") / "player_stats.csv")
    rows["country_key"] = rows["team"].astype(str).str.lower().str.strip().replace(COUNTRY_FIXES)
    rows = rows.merge(rankings[["country_key", "country_code", "rank"]], on="country_key", how="inner")
    rows = rows[rows["position"].astype(str).str.upper() != "GK"]
    rows["player_key"] = rows["player"].astype(str).str.lower().str.strip()
    rows = rows.merge(club, on=["player_key", "country_code"], how="left")
    rows["age"] = rows["age"].astype(str).str.split("-").str[0]
    rows["fifa_rank"] = rows["rank"]
    rows["country"] = rows["team"]
    rows["target_goals"] = rows["goals"].fillna(0.0)
    rows[[
        "club_minutes",
        "club_goals",
        "club_non_penalty_goals",
        "club_penalty_goals",
        "club_assists",
        "club_goals_per90",
        "league_goals_rank_pct",
        "league_non_penalty_goals_rank_pct",
        "team_goal_share",
    ]] = rows[[
        "club_minutes",
        "club_goals",
        "club_non_penalty_goals",
        "club_penalty_goals",
        "club_assists",
        "club_goals_per90",
        "league_goals_rank_pct",
        "league_non_penalty_goals_rank_pct",
        "team_goal_share",
    ]].fillna(0.0)
    rows["source_tournament"] = "2022 World Cup"
    rows = add_position_features(rows)
    return rows[
        [
            "source_tournament",
            "player",
            "country",
            "country_code",
            "position",
            "target_goals",
            *FEATURE_COLUMNS,
        ]
    ]

#concat all input training data
def build_training_table() -> pd.DataFrame:
    """Combine 2018 and 2022 World Cup data into one training set."""
    df1, df2 = build_2018_training_rows(), build_2022_training_rows()
    df = pd.concat([df1,df2], ignore_index=True)
    return df
#Inference for 2026 world cup (we are calling our model on these)

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
