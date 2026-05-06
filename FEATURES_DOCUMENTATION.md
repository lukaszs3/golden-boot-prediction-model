# Features.py Documentation

This module is responsible for loading, transforming, and aggregating player statistics from various data sources to prepare data for the Golden Boot prediction model. It handles feature engineering for both training data (historical World Cup tournaments) and inference data (current 2026 candidates).

## Table of Contents
1. [Constants](#constants)
2. [Utility Functions](#utility-functions)
3. [Data Loading Functions](#data-loading-functions)
4. [Feature Engineering Functions](#feature-engineering-functions)
5. [Training Data Functions](#training-data-functions)
6. [Inference Data Functions](#inference-data-functions)

---

## Constants

### `COUNTRY_FIXES`
```python
COUNTRY_FIXES = {
    "usa": "united states",
    "ir iran": "iran",
    "korea republic": "south korea",
    "turkiye": "turkey",
}
```
A dictionary mapping inconsistent country name formats to standardized names. Used when normalizing country data from different sources that may use different naming conventions.

---

## Utility Functions

### `dataset_dir(dataset_key: str) -> Path`
**Purpose:** Constructs the file path to a specific dataset directory.

**Parameters:**
- `dataset_key` (str): A key from the `KAGGLE_DATASETS` config dictionary

**Returns:** Path object pointing to the dataset folder

**Example Usage:**
```python
fifa_rankings_path = dataset_dir("fifa_rankings")
```

---

## Data Loading Functions

### `load_fifa_snapshot(event_date: str) -> pd.DataFrame`
**Purpose:** Loads FIFA world rankings for a specific event date by retrieving the most recent ranking snapshot on or before that date.

**Parameters:**
- `event_date` (str): ISO format date string (e.g., "2018-06-14")

**Returns:** DataFrame with columns:
- `country_full`: Full country name
- `country_key`: Lowercase, standardized country name
- `country_code`: 3-letter country code
- `rank`: FIFA ranking position

**Process:**
1. Reads all FIFA ranking CSV files from the rankings dataset directory
2. Combines them into a single DataFrame
3. Converts `rank_date` to datetime format
4. Finds the most recent ranking on or before the specified event date
5. Filters to that snapshot date
6. Standardizes country names using `COUNTRY_FIXES`
7. Returns deduplicated data by country code, sorted by rank

**Example Usage:**
```python
rankings_2018 = load_fifa_snapshot("2018-06-14")  # Get 2018 WC rankings
```

---

### `load_current_rankings() -> pd.DataFrame`
**Purpose:** Loads the most recent FIFA world rankings as of January 2026, limited to top 30 teams.

**Returns:** DataFrame with columns:
- `country`: Full country name
- `country_code`: 3-letter country code
- `rank`: FIFA ranking position

**Process:**
1. Attempts to load the 2026 current rankings CSV file
2. If it exists:
   - Loads the current rankings data
   - Normalizes country names to match FIFA snapshot format
   - Merges with latest FIFA snapshot to get country codes
   - Converts rank to numeric format
3. If it doesn't exist:
   - Falls back to loading the latest FIFA snapshot
   - Renames columns appropriately
4. Returns top 30 ranked countries

**Example Usage:**
```python
current_top_teams = load_current_rankings()
```

---

### `load_club_2017_features() -> pd.DataFrame`
**Purpose:** Loads and aggregates club statistics from the 2017-18 season (used as features for 2018 World Cup training data).

**Returns:** Aggregated DataFrame with one row per player per country containing:
- `player_key`, `country_key`, `country_code`
- `club_minutes`, `club_goals`, `club_non_penalty_goals`, `club_penalty_goals`, `club_assists`
- `club_goals_per90`, `league_goals_rank_pct`, `league_non_penalty_goals_rank_pct`, `team_goal_share`
- `age` (max age if player has multiple entries)

**Process:**
1. Loads 2017-18 season FIFA rankings
2. Iterates through all 2017 JSON files in the club stats dataset
3. For each file:
   - Extracts general stats (minutes, age, team)
   - Extracts offensive stats (goals, non-penalty goals, assists)
   - Creates standardized DataFrame
4. Concatenates all data
5. Merges with country codes from FIFA rankings
6. Finishes player data (calculates goals_per90)
7. Adds competition context features (league rankings, team goal share)
8. Aggregates multiple player entries using weighted averages
9. Labels all as forwards (position = "FW")

**Example Usage:**
```python
club_2017 = load_club_2017_features()
```

---

### `load_club_2021_features() -> pd.DataFrame`
**Purpose:** Loads and aggregates club statistics from the 2021-22 season (used as features for 2022 World Cup training data).

**Returns:** Similar structure to `load_club_2017_features()`, containing aggregated player club statistics.

**Process:**
1. Reads 2021-22 season player stats CSV with Latin-1 encoding
2. Extracts key columns: player, nation, squad, position, age, minutes, goals, assists
3. Converts nation codes to 3-letter country codes (last 3 characters)
4. Calculates annualized stats by multiplying by 90s ratio
5. Applies data finishing (calculates non-penalty goals, penalties, goals_per90)
6. Adds competition context features
7. Aggregates multiple entries per player per country
8. Labels all as forwards (position = "FW")

**Example Usage:**
```python
club_2021 = load_club_2021_features()
```

---

## Feature Engineering Functions

### `finish_players(df: pd.DataFrame) -> pd.DataFrame`
**Purpose:** Calculates derived goal-related statistics for player data.

**Parameters:**
- `df` (pd.DataFrame): Player statistics DataFrame

**Returns:** DataFrame with new/modified columns:
- `club_non_penalty_goals`: Non-penalty goals (or all goals if unavailable)
- `club_penalty_goals`: Penalty goals (total - non-penalty)
- `club_goals_per90`: Goals per 90 minutes played

**Process:**
1. Converts goal columns to numeric, filling NaN with 0
2. Calculates penalty goals as: total goals - non-penalty goals, clipped to minimum 0
3. Calculates goals per 90 as: goals / (minutes / 90), handling zero minutes by returning 0

**Example Usage:**
```python
df = finish_players(player_df)
```

---

### `add_competition_context_features(df: pd.DataFrame) -> pd.DataFrame`
**Purpose:** Adds league-level and team-level relative performance features to position players within their competitive context.

**Parameters:**
- `df` (pd.DataFrame): Player statistics DataFrame with league and team information

**Returns:** DataFrame with new columns:
- `league_goals_rank_pct`: Percentile rank of player's goal scoring within their league (0-1 scale, 1 = best)
- `league_non_penalty_goals_rank_pct`: Percentile rank of non-penalty goals within league
- `team_goal_share`: Proportion of their team's goals scored by this player

**Process:**
1. Fills missing league and team names with "unknown"
2. For each league:
   - Counts total players
   - Ranks players by goals and non-penalty goals
   - Converts ranks to percentile (1 - normalized rank)
3. For each team:
   - Sums total goals
   - Calculates each player's share (player_goals / team_goals)

**Example Usage:**
```python
df = add_competition_context_features(df)
```

---

### `add_position_features(df: pd.DataFrame) -> pd.DataFrame`
**Purpose:** Creates binary features indicating player position type using position codes.

**Parameters:**
- `df` (pd.DataFrame): Player data with `position` column

**Returns:** DataFrame with new columns (0 or 1):
- `position_is_defender`: True if position contains "DF"
- `position_is_midfielder`: True if position contains "MF"
- `position_is_forward`: True if position contains "FW", "ST", "CF", "SS", "LW", or "RW"

**Process:**
1. Converts position text to uppercase
2. Uses regex patterns to detect position types
3. Creates binary indicator columns

**Example Usage:**
```python
df = add_position_features(df)
```

---

### `aggregate_player_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame`
**Purpose:** Consolidates multiple rows per player (e.g., multiple clubs in a season) into single summary rows using aggregation logic.

**Parameters:**
- `df` (pd.DataFrame): Player data with multiple rows per player
- `group_cols` (list[str]): Columns to group by (e.g., ["player_key", "country_code"])

**Returns:** Aggregated DataFrame with one row per grouping

**Aggregation Logic:**
- `age`: Maximum value across entries
- `club_minutes`, `club_goals`, `club_non_penalty_goals`, `club_penalty_goals`, `club_assists`: Sum
- `league_goals_rank_pct`, `league_non_penalty_goals_rank_pct`, `team_goal_share`: Weighted average using `club_minutes` as weights
- Other columns: Preserved from first row

**Process:**
1. Defines weighted average helper function (uses club_minutes as weights)
2. Iterates through groups
3. For each group:
   - Preserves grouping column values
   - Aggregates numeric columns appropriately
   - Appends to result list
4. Converts to DataFrame and applies `finish_players()`

**Example Usage:**
```python
aggregated = aggregate_player_rows(player_df, ["player_key", "country_code"])
```

---

## Training Data Functions

### `build_2018_training_rows() -> pd.DataFrame`
**Purpose:** Constructs training data from the 2018 World Cup with actual goal labels, combining squad data, player information, and pre-tournament club statistics.

**Returns:** DataFrame with one row per non-goalkeeper player from the 2018 World Cup with columns:
- `source_tournament`: "2018 World Cup"
- `player`, `country`, `country_code`, `position`
- `target_goals`: Actual goals scored in the tournament
- All feature columns (see FEATURE_COLUMNS in config)

**Process:**
1. Loads 2018 FIFA rankings (top 30)
2. Loads 2018 World Cup squads, filtering to top 30 teams and non-goalkeepers
3. Constructs player names from given/family names
4. Creates player_key for matching
5. Counts actual goals scored in the tournament (excluding own goals)
6. Loads pre-tournament club stats (2017-18 season)
7. Merges with club statistics
8. Calculates player age as of June 14, 2018
9. Maps FIFA rankings by country
10. Fills missing values with 0 for club statistics
11. Adds position features
12. Returns selected columns

**Example Usage:**
```python
training_2018 = build_2018_training_rows()
```

---

### `build_2022_training_rows() -> pd.DataFrame`
**Purpose:** Constructs training data from the 2022 World Cup with actual goal labels using a different data source format than 2018.

**Returns:** Similar structure to `build_2018_training_rows()` but for 2022 tournament.

**Process:**
1. Loads 2022 FIFA rankings
2. Loads pre-tournament club stats (2021-22 season)
3. Reads 2022 World Cup player stats
4. Normalizes country names and merges with rankings
5. Filters to non-goalkeepers and top 30 teams
6. Merges with club statistics
7. Extracts age from age string format
8. Maps FIFA rankings
9. Uses actual tournament goals as target_goals
10. Fills missing club statistics with 0
11. Adds position features
12. Returns selected columns

**Example Usage:**
```python
training_2022 = build_2022_training_rows()
```

---

### `build_training_table() -> pd.DataFrame`
**Purpose:** Combines 2018 and 2022 World Cup training data into a single comprehensive training set.

**Returns:** Concatenated DataFrame containing all players from both tournaments.

**Process:**
1. Calls `build_2018_training_rows()`
2. Calls `build_2022_training_rows()`
3. Concatenates both DataFrames
4. Returns combined training set

**Example Usage:**
```python
training_data = build_training_table()
```

---

## Inference Data Functions

### `load_current_2026_features() -> pd.DataFrame`
**Purpose:** Loads and aggregates club statistics from the 2025-26 season for current player candidates in the 2026 World Cup prediction.

**Returns:** Aggregated DataFrame with one row per player per country containing club statistics.

**Process:**
1. Determines which current season dataset file exists (full or light version)
2. Reads the CSV file
3. Constructs DataFrame with standardized column names
4. Extracts country code from nation field (last 3 characters)
5. Finishes player data (calculates derived stats)
6. Adds competition context features
7. Aggregates multiple entries per player
8. Returns aggregated data with player, position, and club statistics

**Example Usage:**
```python
current_candidates = load_current_2026_features()
```

---

### `build_current_candidates(min_minutes: int = 300, max_per_country: int = 10) -> pd.DataFrame`
**Purpose:** Builds and filters the final set of 2026 World Cup candidates for scoring, applying multiple selection criteria.

**Parameters:**
- `min_minutes` (int, default=300): Minimum club minutes played in 2025-26 season
- `max_per_country` (int, default=10): Maximum candidates per country (0 = no limit)

**Returns:** Final candidate DataFrame with columns:
- `player`, `country`, `country_code`, `position`
- All feature columns

**Selection Criteria:**
1. Must be in a top 30 FIFA-ranked country
2. Must be in the allowed countries list (ALLOWED_CANDIDATE_COUNTRY_CODES config)
3. Must have played minimum club minutes
4. Must play a forward/attacking position (FW, ST, CF, SS, LW, RW)
5. Per-country limit: Top candidates by goals, then minutes

**Process:**
1. Loads current FIFA rankings
2. Loads current 2025-26 season statistics
3. Merges with rankings
4. Filters by country code, minutes, and position
5. Maps FIFA ranking to feature
6. Adds position features
7. If max_per_country > 0:
   - Sorts by country, goals (desc), minutes (desc)
   - Groups by country
   - Keeps top N per country
8. Selects and returns final columns sorted by country and player name

**Example Usage:**
```python
candidates = build_current_candidates(min_minutes=300, max_per_country=10)
```

---

## Data Flow Summary

### Training Pipeline
```
2018 World Cup:
  - Load squads → Load club 2017-18 stats
  - Count actual goals → Merge with rankings
  - build_2018_training_rows()

2022 World Cup:
  - Load stats → Load club 2021-22 stats
  - Count actual goals → Merge with rankings
  - build_2022_training_rows()

Combined:
  - build_training_table() → Training dataset
```

### Inference Pipeline
```
Load 2025-26 stats:
  - load_current_2026_features()
  - add_competition_context_features()

Load rankings:
  - load_current_rankings()

Build candidates:
  - Filter by criteria
  - build_current_candidates()
  - Score with model
```

---

## Key Design Patterns

1. **Feature Consistency**: Both training and inference paths normalize country codes and player names to ensure matching
2. **Weighted Aggregation**: When players have multiple club entries, statistics are combined using minutes-weighted averages for rates and sums for totals
3. **Context Features**: Players are positioned relative to their league and team peers, not just absolute statistics
4. **Forward Focus**: The model specifically targets forward/attacking positions where Golden Boot goals are scored
5. **Modular Design**: Functions are composed to build complex pipelines from simple building blocks

