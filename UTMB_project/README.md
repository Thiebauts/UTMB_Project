# UTMB Project - Race Data Pipeline

A Python pipeline to scrape race results from UTMB World Series events, extract UTMB scores, and combine data for analysis.

## Comparison: UTMB_project vs UTMB_project_2

| Feature | UTMB_project | UTMB_project_2 |
|---------|--------------|----------------|
| **Speed** | ~5-10 min per race | ~2-5 seconds per race |
| **Data source** | live.utmb.world (runner pages) | Direct API |
| **Finisher data** | ✓ | ✓ |
| **DNF data** | ✓ | ✗ |
| **Bib numbers** | ✓ | ✗ |
| **Checkpoint times** | ✓ (planned) | ✗ |
| **UTMB scores** | ✓ (with auth) | ✓ (with auth) |
| **Use case** | Detailed data with checkpoints | Fast bulk downloads |

**Use UTMB_project** when you need:
- Complete data including DNFs and checkpoints
- Bib numbers
- Detailed race analysis
- Checkpoint timing data (planned feature)

**Use UTMB_project_2** when you need:
- Fast bulk downloads of finisher data
- Quick UTMB score retrieval
- Less detailed data is acceptable

## Key Features

- **50+ UTMB World Series events**: Complete list of all known events built-in
- **Year-specific race data**: Each event/year combination is stored separately, handling course changes (distance, elevation) between years
- **Automatic metadata extraction**: Distance, elevation gain (D+), and race date are fetched from live.utmb.world
- **Two-step scraping**: Step 1 scrapes results (no login), Step 2 adds UTMB scores (requires cookie)
- **Master file generation**: Combine all races into a single CSV for analysis
- **Checkpoint data**: Planned feature for detailed checkpoint timing analysis

## Quick Start

### 1. Add Event Data to Database

Before scraping, populate `events.json` with race data:

```bash
cd UTMB_project

# See all 50+ known UTMB World Series events
python3 update_race_data.py --list-known

# Add a specific event for a specific year
python3 update_race_data.py --event kullamannen --year 2025

# Add all available years for an event (2018-present)
python3 update_race_data.py --event kullamannen --all-years

# Fetch ALL known UTMB World Series events (all years) - takes a while!
python3 update_race_data.py --all-events

# Fetch all events, but skip ones already in database
python3 update_race_data.py --all-events --skip-existing

# List events in database
python3 update_race_data.py --list

# Show races for a specific event/year
python3 update_race_data.py --show kullamannen 2025
```

### 2. Scrape Race Results (No login needed)

```bash
python3 utmb_scraper.py --step 1 --tenant kullamannen --year 2025 --race 100M
```

### 3. Add UTMB Scores (Login required)

1. Log in to [utmb.world](https://utmb.world)
2. Open Developer Tools (F12) → Network tab
3. Click any request to `utmb.world` and copy the Cookie value

```bash
python3 utmb_scraper.py --step 2 --tenant kullamannen --year 2025 --race 100M --cookie "YOUR_COOKIE"
```

### 4. Merge All Results

```bash
python3 merge_results.py
```

## Project Structure

```
UTMB_project/
├── events.json           # Race data by event and year
├── update_race_data.py   # Script to populate/update events.json
├── utmb_scraper.py       # Main scraper (Step 1 + Step 2)
├── merge_results.py      # Combine all CSVs into master file
├── data/
│   ├── kullamannen_2025_100M.csv
│   └── ...
├── master_results.csv    # Combined results from all races
└── README.md
```

## events.json Structure

Race data is stored per event and per year, handling year-to-year changes:

```json
{
  "events": {
    "kullamannen": {
      "event_name": "Kullamannen by UTMB®",
      "years": {
        "2025": {
          "event_info": {
            "country": "Sweden",
            "country_code": "SE",
            "date_start": "2025-10-31"
          },
          "races": {
            "100M": {
              "name": "Ultra 100 Miles",
              "distance_km": 173,
              "elevation_gain": 2300,
              "elevation_loss": 2300,
              "utmb_category": "100m",
              "running_stones": 4,
              "start_date": "2025-10-31T17:00:11.000Z",
              "start_place": "Hoganas"
            },
            "100k": {
              "name": "Sprint Ultra 100km",
              "distance_km": 108,
              "elevation_gain": 749,
              "start_date": "2025-10-31T21:00:08.000Z",
              "start_place": "Hoganas"
            }
          }
        }
      }
    }
  }
}
```

**Race fields include:**
- `distance_km`, `elevation_gain`, `elevation_loss` - Course metrics
- `start_date` - Race start date/time in ISO format (e.g., `2025-10-31T17:00:11.000Z`)
- `start_place` - Start location (e.g., `Hoganas`)
- `utmb_category` - UTMB category (`100m`, `100k`, `50k`, etc.)
- `running_stones` - UTMB Running Stones points

## Command-Line Reference

Both `UTMB_project` and `UTMB_project_2` use the same CLI keywords for consistency:
- `--step 1` or `--step 2`
- `--tenant` (event name)
- `--year`
- `--race`
- `--all` (batch mode)
- `--skip-existing`
- `--cookie` (for authentication)

### update_race_data.py

Manages `events.json` with race data from live.utmb.world. Includes a built-in list of 50+ UTMB World Series events.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--event`, `-e` | str | - | Event tenant name (e.g., `kullamannen`, `montblanc`) |
| `--year`, `-y` | int | - | Specific year to fetch |
| `--all-years`, `-a` | flag | - | Fetch all available years (2018-present) for `--event` |
| `--all-events` | flag | - | Fetch ALL known UTMB World Series events (all years) |
| `--skip-existing` | flag | False | With `--all-events`: skip events already in database |
| `--list`, `-l` | flag | - | List all events in database |
| `--list-known` | flag | - | List all 50+ known UTMB World Series event tenants |
| `--show`, `-s` | str str | - | Show races for EVENT YEAR |

**Examples:**

```bash
# List all known UTMB World Series events
python3 update_race_data.py --list-known

# Add kullamannen 2025
python3 update_race_data.py --event kullamannen --year 2025

# Add all years for UTMB Mont-Blanc
python3 update_race_data.py --event montblanc --all-years

# Fetch ALL events (50+ events, all years) - takes ~30 min
python3 update_race_data.py --all-events

# Fetch all events, skip ones already fetched
python3 update_race_data.py --all-events --skip-existing

# List events in database
python3 update_race_data.py --list

# Show alsacegrandest 2023 races
python3 update_race_data.py --show alsacegrandest 2023
```

**Known Event Tenants (examples):**

| Region | Tenants |
|--------|---------|
| Europe | `utmb`, `kullamannen`, `lavaredo`, `eigerultratrail`, `alsacegrandest`, `tenerifebyutmb`, `mozart100`... |
| Oceania | `tarawera`, `uta`, `kosciuszko` |
| North America | `canyons`, `grindstone`, `kodiak`, `whistler`, `puertovallarta`... |
| South America | `paraty`, `quitobyutmb`, `valholl`... |
| Asia | `chiangmai`, `translantau`, `transjeju`, `malaysia`... |
| Africa | `mut` |

Use `--list-known` to see the complete list with regions.

> **Note on Tenant Names**: The tenant names on `live.utmb.world` are **inconsistent**. Some use short names (`kullamannen`), some add "byutmb" (`tenerifebyutmb`), and some use full event names (`eigerultratrail`, `100milesofistria`). The `UTMB_EVENTS` list in `update_race_data.py` contains verified tenant names.

### utmb_scraper.py

Scrapes race results and UTMB scores. Supports single race mode and batch mode.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--step` | int | **required** | `1` = scrape results, `2` = fetch UTMB scores |
| `--tenant` | str | `kullamannen` | Event tenant name (filter in batch mode) |
| `--year` | int | `2025` | Event year (filter in batch mode) |
| `--race` | str | `100M` | Race ID (single race mode only) |
| `--all` | flag | - | **Batch mode**: scrape all races from `events.json` |
| `--skip-existing` | flag | False | With `--all`: skip races that already have CSV files |
| `--dry-run` | flag | False | With `--all`: show what would be scraped without actually scraping |
| `--bib-min` | int | auto | Minimum bib number |
| `--bib-max` | int | auto | Maximum bib number |
| `--max-search` | int | `10000` | Max bib for auto-discovery |
| `--cookie` | str | - | Browser cookie (**required for step 2**) |

**Single Race Mode:**

```bash
# Scrape with defaults (Kullamannen 2025 100M)
python3 utmb_scraper.py --step 1

# Scrape UTMB Mont-Blanc 2024 CCC
python3 utmb_scraper.py --step 1 --tenant montblanc --year 2024 --race CCC

# Add UTMB scores
python3 utmb_scraper.py --step 2 --tenant kullamannen --year 2025 --race 100M --cookie "access_token=xxx..."
```

**Batch Mode (scrape from events.json):**

```bash
# Preview what would be scraped (dry run)
python3 utmb_scraper.py --step 1 --all --dry-run

# Scrape ALL races from ALL events in events.json
python3 utmb_scraper.py --step 1 --all

# Scrape all races, but skip ones that already have CSV files
python3 utmb_scraper.py --step 1 --all --skip-existing

# Scrape all races for a specific event (all years)
python3 utmb_scraper.py --step 1 --all --tenant kullamannen

# Scrape all races for a specific event and year
python3 utmb_scraper.py --step 1 --all --tenant kullamannen --year 2025

# Add UTMB scores to all existing CSV files
python3 utmb_scraper.py --step 2 --all --cookie "access_token=xxx..."

# Add UTMB scores only for kullamannen races
python3 utmb_scraper.py --step 2 --all --tenant kullamannen --cookie "access_token=xxx..."
```

**Workflow for scraping all data:**

```bash
# 1. First, populate events.json with race metadata
python3 update_race_data.py --all-events

# 2. Then scrape all race results (this takes a long time!)
python3 utmb_scraper.py --step 1 --all

# 3. Finally, add UTMB scores (requires login cookie)
python3 utmb_scraper.py --step 2 --all --cookie "YOUR_COOKIE"

# 4. Merge all results into master file
python3 merge_results.py
```

### merge_results.py

Combines all race CSV files into a master file.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | `data/` | Directory with CSV files |
| `--output` | str | `master_results.csv` | Output file |
| `--finishers-only` | flag | False | Exclude DNF/DNS |

**Examples:**

```bash
python3 merge_results.py
python3 merge_results.py --finishers-only
python3 merge_results.py --output analysis/all_races.csv
```

## CSV File Format

Each race CSV includes metadata headers:

```csv
# Race: Kullamannen by UTMB 2025 - Ultra 100 Miles
# Distance: 173 km
# D+: 2300 m
# D-: 2300 m
# Date: 2025-10-31
# URL: https://live.utmb.world/kullamannen/2025
bib,name,country,age,sex,category,club,utmb_profile_url,utmb_index,rank_scratch,...,utmb_score
```

## Data Columns

| Column | Description |
|--------|-------------|
| `bib` | Runner's bib number |
| `name` | Full name |
| `country` | Country code (SE, FR, etc.) |
| `age`, `sex`, `category` | Demographics |
| `club` | Running club |
| `utmb_profile_url` | Link to UTMB profile |
| `utmb_index` | Overall UTMB Performance Index |
| `rank_scratch`, `rank_sex`, `rank_category` | Rankings |
| `race_time` | Finish time (HH:MM:SS) |
| `race_time_seconds` | Finish time in seconds |
| `is_finisher` | True/False |
| `status` | Race status (f=finished) |
| `utmb_score` | UTMB score for this race (Step 2) |

## Requirements

```bash
pip install requests pandas tqdm
```

## Notes

- **Year-to-year changes**: Races can change distance, elevation, or be added/removed between years. The `events.json` database stores each year separately.
- **D- (elevation loss)**: Not always provided by the API. Defaults to D+ value.
- **Cookie expiration**: Access token expires after ~30 minutes. Get a fresh cookie if Step 2 fails.
- **Bib range**: Auto-discovered by default.
- **Finding race IDs**: Check the live results page (e.g., [live.utmb.world/kullamannen/2025](https://live.utmb.world/kullamannen/2025)) to see available race IDs.
