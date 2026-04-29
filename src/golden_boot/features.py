from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, KAGGLE_DATASETS, RAW_DIR


CLUB_STAT_COLUMNS = [
    "club_games",
    "club_minutes",
    "club_goals",
    "club_assists",
    "club_goals_per90",
    "club_assists_per90",
    "club_shots",
    "club_shots_per90",
    "club_crosses",
]


def dataset_dir(dataset_key: str) -> Path:
    return RAW_DIR / KAGGLE_DATASETS[dataset_key]["folder"]


def normalize_text(value: object) -> str:
    """Normalize names/countries enough for simple joins across Kaggle files."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace(".", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    return pd.to_numeric(text, errors="coerce")


def parse_age(value: object) -> float:
    """Handle ages like 25, 25.0, or World Cup style '32-094'."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if "-" in text:
        years, days = text.split("-", 1)
        return float(years) + float(days[:3]) / 365.25
    return float(pd.to_numeric(text, errors="coerce"))


def parse_nation_code(value: object) -> str:
    tokens = re.findall(r"[A-Z]{3}", str(value).upper())
    return tokens[-1] if tokens else ""


def name_from_parts(given_name: object, family_name: object) -> str:
    given = "" if pd.isna(given_name) else str(given_name).strip()
    family = "" if pd.isna(family_name) else str(family_name).strip()
    if normalize_text(given) in {"", "not applicable", "unknown"}:
        return family
    return f"{given} {family}".strip()


def age_on_date(birth_date: object, event_date: str) -> float:
    birth = pd.to_datetime(birth_date, errors="coerce")
    if pd.isna(birth):
        return np.nan
    event = pd.Timestamp(event_date)
    return (event - birth).days / 365.25


def country_aliases() -> dict[str, str]:
    return {
        "england": "ENG",
        "scotland": "SCO",
        "wales": "WAL",
        "northern ireland": "NIR",
        "united states": "USA",
        "usa": "USA",
        "iran": "IRN",
        "ir iran": "IRN",
        "south korea": "KOR",
        "korea republic": "KOR",
        "turkey": "TUR",
        "turkiye": "TUR",
        "russia": "RUS",
    }


def fifa_rankings_for_date(event_date: str) -> pd.DataFrame:
    folder = dataset_dir("fifa_rankings")
    files = sorted(folder.glob("fifa_ranking-*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No FIFA ranking CSVs found in {folder}. Run with --download first."
        )

    rankings = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    rankings = rankings.drop_duplicates()
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"], errors="coerce")
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    rankings["total_points"] = pd.to_numeric(rankings["total_points"], errors="coerce")
    rankings["country_code"] = rankings["country_abrv"].astype(str).str.upper()
    rankings["country_key"] = rankings["country_full"].map(normalize_text)

    cutoff = pd.Timestamp(event_date)
    snapshot_date = rankings.loc[rankings["rank_date"] <= cutoff, "rank_date"].max()
    if pd.isna(snapshot_date):
        raise ValueError(f"No FIFA ranking snapshot found on or before {event_date}.")

    snapshot = rankings[rankings["rank_date"] == snapshot_date].copy()
    snapshot = snapshot.sort_values(["rank", "country_full"])
    return snapshot.drop_duplicates("country_code")


def country_code_lookup() -> dict[str, str]:
    latest = fifa_rankings_for_date("2100-01-01")
    lookup = (
        latest.drop_duplicates("country_key")
        .set_index("country_key")["country_code"]
        .to_dict()
    )
    lookup.update(country_aliases())
    return lookup


def current_top30_rankings() -> pd.DataFrame:
    """Use the Kaggle January 2026 top-30 snapshot when present."""
    lookup = country_code_lookup()
    ranking_path = dataset_dir("current_rankings_2026") / "fifa_world_rankings_jan_2026.csv"

    if ranking_path.exists():
        df = pd.read_csv(ranking_path)
        out = pd.DataFrame(
            {
                "rank": pd.to_numeric(df["Rank"], errors="coerce"),
                "country_full": df["Country"],
                "country_key": df["Country"].map(normalize_text),
                "total_points": pd.to_numeric(df["Points"], errors="coerce"),
            }
        )
        out["country_code"] = out["country_key"].map(lookup)
        return out.dropna(subset=["country_code"]).sort_values("rank").head(30)

    latest = fifa_rankings_for_date("2100-01-01")
    return latest.rename(columns={"country_abrv": "country_code"}).head(30)


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _combine_positions(values: pd.Series) -> str:
    parts: list[str] = []
    for value in values.dropna().astype(str):
        parts.extend(piece.strip() for piece in value.split(","))
    unique_parts = sorted({piece for piece in parts if piece})
    return ",".join(unique_parts)


def _aggregate_club_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    numeric_sum_cols = [
        "club_games",
        "club_minutes",
        "club_goals",
        "club_assists",
        "club_shots",
        "club_crosses",
    ]
    for col in numeric_sum_cols + ["age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby(["name_key", "country_code"], dropna=False)
        .agg(
            player=("player", "first"),
            position=("position", _combine_positions),
            age=("age", "max"),
            club_games=("club_games", "sum"),
            club_minutes=("club_minutes", "sum"),
            club_goals=("club_goals", "sum"),
            club_assists=("club_assists", "sum"),
            club_shots=("club_shots", "sum"),
            club_crosses=("club_crosses", "sum"),
        )
        .reset_index()
    )

    nineties = grouped["club_minutes"].replace(0, np.nan) / 90.0
    grouped["club_goals_per90"] = (grouped["club_goals"] / nineties).fillna(0)
    grouped["club_assists_per90"] = (grouped["club_assists"] / nineties).fillna(0)
    grouped["club_shots_per90"] = (grouped["club_shots"] / nineties).fillna(0)
    grouped["has_club_stats"] = 1.0
    return grouped


def load_club_2017_features() -> pd.DataFrame:
    """2017/18 club form used before the 2018 World Cup."""
    folder = dataset_dir("club_2017_2018") / "Football_Players"
    files = sorted(folder.glob("2017*.json"))
    if not files:
        raise FileNotFoundError(f"No 2017 club JSON files found in {folder}.")

    lookup = country_code_lookup()
    rows: list[dict] = []

    for path in files:
        frame = pd.read_json(path, lines=True)
        for _, row in frame.iterrows():
            general = _as_dict(row.get("general_stats"))
            offensive = _as_dict(row.get("offensive_stats"))
            passing = _as_dict(row.get("passing_stats"))
            minutes = float(general.get("time") or 0)
            games = float(general.get("games") or 0)

            rows.append(
                {
                    "player": row.get("name"),
                    "name_key": normalize_text(row.get("name")),
                    "country_code": lookup.get(normalize_text(row.get("nationality"))),
                    "position": row.get("position"),
                    "age": parse_age(row.get("age")),
                    "club_games": games,
                    "club_minutes": minutes,
                    "club_goals": float(offensive.get("goals") or 0),
                    "club_assists": float(offensive.get("assists") or 0),
                    "club_shots": float(offensive.get("shots") or 0),
                    "club_crosses": float(passing.get("CrsPA") or 0),
                }
            )

    return _aggregate_club_rows(pd.DataFrame(rows))


def load_club_2021_features() -> pd.DataFrame:
    """2021/22 club form used before the 2022 World Cup."""
    path = dataset_dir("club_2021_2022") / "2021-2022 Football Player Stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run with --download first.")

    df = pd.read_csv(path, encoding="latin1", sep=";")
    nineties = pd.to_numeric(df["90s"], errors="coerce").fillna(0)
    goals_per90 = pd.to_numeric(df["Goals"], errors="coerce").fillna(0)
    assists_per90 = pd.to_numeric(df["Assists"], errors="coerce").fillna(0)
    shots_per90 = pd.to_numeric(df["Shots"], errors="coerce").fillna(0)
    crosses_per90 = pd.to_numeric(df["Crs"], errors="coerce").fillna(0)

    out = pd.DataFrame(
        {
            "player": df["Player"],
            "name_key": df["Player"].map(normalize_text),
            "country_code": df["Nation"].map(parse_nation_code),
            "position": df["Pos"],
            "age": df["Age"].map(parse_age),
            "club_games": pd.to_numeric(df["MP"], errors="coerce").fillna(0),
            "club_minutes": pd.to_numeric(df["Min"], errors="coerce").fillna(0),
            "club_goals": goals_per90 * nineties,
            "club_assists": assists_per90 * nineties,
            "club_shots": shots_per90 * nineties,
            "club_crosses": crosses_per90 * nineties,
        }
    )
    return _aggregate_club_rows(out)


def load_current_2026_features() -> pd.DataFrame:
    """Current 2025/26 player form for inference candidates."""
    folder = dataset_dir("current_2025_2026")
    path = folder / "players_data-2025_2026.csv"
    if not path.exists():
        path = folder / "players_data_light-2025_2026.csv"
    if not path.exists():
        raise FileNotFoundError(f"No current player stats CSV found in {folder}.")

    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "player": df["Player"],
            "name_key": df["Player"].map(normalize_text),
            "country_code": df["Nation"].map(parse_nation_code),
            "position": df["Pos"],
            "age": df["Age"].map(parse_age),
            "club_games": pd.to_numeric(df["MP"], errors="coerce").fillna(0),
            "club_minutes": pd.to_numeric(df["Min"], errors="coerce").fillna(0),
            "club_goals": pd.to_numeric(df["Gls"], errors="coerce").fillna(0),
            "club_assists": pd.to_numeric(df["Ast"], errors="coerce").fillna(0),
            "club_shots": pd.to_numeric(df.get("Sh", 0), errors="coerce").fillna(0),
            "club_crosses": pd.to_numeric(df.get("Crs", 0), errors="coerce").fillna(0),
        }
    )
    return _aggregate_club_rows(out)


def is_forward_position(position: object) -> bool:
    text = str(position).upper()
    return bool(re.search(r"\b(FW|ST|CF|SS|LW|RW)\b", text.replace(",", " ")))


def is_midfield_position(position: object) -> bool:
    text = str(position).upper()
    return bool(re.search(r"\b(MF|AM|LM|RM)\b", text.replace(",", " ")))


def add_position_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_forward"] = df["position"].map(is_forward_position).astype(float)
    df["is_midfielder"] = df["position"].map(is_midfield_position).astype(float)
    return df


def attacking_role_mask(df: pd.DataFrame) -> pd.Series:
    """Keep forwards, plus midfielders with a clear attacking profile."""
    forward = df["position"].map(is_forward_position)
    midfielder = df["position"].map(is_midfield_position)
    direct_output = df["club_goals"].fillna(0) + df["club_assists"].fillna(0)
    per90_output = df["club_goals_per90"].fillna(0) + df["club_assists_per90"].fillna(0)
    shot_volume = df["club_shots_per90"].fillna(0)
    attacking_mid = midfielder & (
        (direct_output >= 3.0) | (per90_output >= 0.15) | (shot_volume >= 0.75)
    )
    return forward | attacking_mid


def _rank_maps(ranking_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    ranking_df = ranking_df.drop_duplicates("country_code")
    rank_map = ranking_df.set_index("country_code")["rank"].to_dict()
    points_map = ranking_df.set_index("country_code")["total_points"].to_dict()
    return rank_map, points_map


def _clean_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CLUB_STAT_COLUMNS:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in FEATURE_COLUMNS:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["has_club_stats"] = df["has_club_stats"].fillna(0.0)
    df["is_forward"] = df["is_forward"].fillna(0.0)
    df["is_midfielder"] = df["is_midfielder"].fillna(0.0)
    return df


def build_2018_training_rows() -> pd.DataFrame:
    event_date = "2018-06-14"
    rankings = fifa_rankings_for_date(event_date)
    top_codes = set(rankings.head(30)["country_code"])
    rank_map, points_map = _rank_maps(rankings)

    roster = pd.read_csv(dataset_dir("world_cup_database") / "squads.csv")
    roster = roster[roster["tournament_id"].eq("WC-2018")].copy()
    roster = roster[roster["team_code"].isin(top_codes)]
    roster = roster[roster["position_code"].isin(["FW", "MF"])]
    roster["player"] = [
        name_from_parts(given, family)
        for given, family in zip(roster["given_name"], roster["family_name"])
    ]
    roster["name_key"] = roster["player"].map(normalize_text)

    players = pd.read_csv(dataset_dir("world_cup_database") / "players.csv")
    roster = roster.merge(players[["player_id", "birth_date"]], on="player_id", how="left")
    roster["age_at_tournament"] = roster["birth_date"].map(
        lambda value: age_on_date(value, event_date)
    )

    goals = pd.read_csv(dataset_dir("world_cup_database") / "goals.csv")
    goal_counts = (
        goals[
            goals["tournament_id"].eq("WC-2018")
            & pd.to_numeric(goals["own_goal"], errors="coerce").eq(0)
        ]
        .groupby("player_id")
        .size()
        .rename("target_goals")
    )
    roster = roster.merge(goal_counts, on="player_id", how="left")
    roster["target_goals"] = roster["target_goals"].fillna(0.0)

    club = load_club_2017_features().rename(
        columns={
            "country_code": "club_country_code",
            "player": "club_player",
            "position": "club_position",
            "age": "club_age",
        }
    )
    rows = roster.merge(club, on="name_key", how="left")
    rows = rows[rows["club_country_code"].isna() | rows["club_country_code"].eq(rows["team_code"])]

    rows["source_tournament"] = "2018 World Cup"
    rows["country"] = rows["team_name"]
    rows["country_code"] = rows["team_code"]
    rows["position"] = rows["position_code"]
    rows["age"] = rows["club_age"].fillna(rows["age_at_tournament"])
    rows["fifa_rank"] = rows["country_code"].map(rank_map)
    rows["fifa_points"] = rows["country_code"].map(points_map)
    rows["has_club_stats"] = rows["has_club_stats"].fillna(0.0)
    rows = _clean_feature_columns(add_position_flags(rows))
    return rows[attacking_role_mask(rows)].copy()


def build_2022_training_rows() -> pd.DataFrame:
    event_date = "2022-11-20"
    rankings = fifa_rankings_for_date(event_date)
    top_codes = set(rankings.head(30)["country_code"])
    rank_map, points_map = _rank_maps(rankings)
    lookup = country_code_lookup()

    rows = pd.read_csv(dataset_dir("world_cup_2022") / "player_stats.csv")
    rows["country_code"] = rows["team"].map(lambda value: lookup.get(normalize_text(value)))
    rows = rows[rows["country_code"].isin(top_codes)].copy()
    rows = rows[rows["position"].map(is_forward_position) | rows["position"].map(is_midfield_position)]
    rows["name_key"] = rows["player"].map(normalize_text)
    rows["age_at_tournament"] = rows["age"].map(parse_age)

    club = load_club_2021_features().rename(
        columns={
            "country_code": "club_country_code",
            "player": "club_player",
            "position": "club_position",
            "age": "club_age",
        }
    )
    rows = rows.merge(club, on="name_key", how="left")
    rows = rows[rows["club_country_code"].isna() | rows["club_country_code"].eq(rows["country_code"])]

    rows["source_tournament"] = "2022 World Cup"
    rows["target_goals"] = pd.to_numeric(rows["goals"], errors="coerce").fillna(0.0)
    rows["country"] = rows["team"]
    rows["age"] = rows["club_age"].fillna(rows["age_at_tournament"])
    rows["fifa_rank"] = rows["country_code"].map(rank_map)
    rows["fifa_points"] = rows["country_code"].map(points_map)
    rows["has_club_stats"] = rows["has_club_stats"].fillna(0.0)
    rows = _clean_feature_columns(add_position_flags(rows))
    return rows[attacking_role_mask(rows)].copy()


def build_training_table() -> pd.DataFrame:
    rows = pd.concat(
        [build_2018_training_rows(), build_2022_training_rows()],
        ignore_index=True,
    )
    rows["target_goals"] = pd.to_numeric(rows["target_goals"], errors="coerce").fillna(0.0)
    keep_cols = [
        "source_tournament",
        "player",
        "country",
        "country_code",
        "position",
        "target_goals",
        *FEATURE_COLUMNS,
    ]
    return rows[keep_cols].sort_values(["source_tournament", "country", "player"])


def build_current_candidates(
    min_minutes: int = 300,
    max_per_country: int = 10,
) -> pd.DataFrame:
    rankings = current_top30_rankings()
    top_codes = set(rankings["country_code"])
    rank_map, points_map = _rank_maps(rankings)
    country_map = rankings.drop_duplicates("country_code").set_index("country_code")[
        "country_full"
    ].to_dict()

    rows = load_current_2026_features()
    rows = rows[rows["country_code"].isin(top_codes)].copy()
    rows["country"] = rows["country_code"].map(country_map)
    rows["fifa_rank"] = rows["country_code"].map(rank_map)
    rows["fifa_points"] = rows["country_code"].map(points_map)
    rows = _clean_feature_columns(add_position_flags(rows))
    rows = rows[rows["club_minutes"] >= float(min_minutes)]
    rows = rows[attacking_role_mask(rows)].copy()

    rows["candidate_score"] = (
        rows["club_goals"]
        + 0.60 * rows["club_assists"]
        + 0.05 * rows["club_shots"]
        + 0.02 * rows["club_crosses"]
    )
    if max_per_country > 0:
        rows = (
            rows.sort_values(["country_code", "candidate_score"], ascending=[True, False])
            .groupby("country_code", group_keys=False)
            .head(max_per_country)
        )

    keep_cols = ["player", "country", "country_code", "position", *FEATURE_COLUMNS]
    return rows[keep_cols].sort_values(["country", "player"])
