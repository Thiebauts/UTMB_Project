# UTMB Race Data Scraper v2 - Fast API-Based

A fast Python scraper for fetching race results from UTMB World Series events using the direct API endpoint.

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

**Use UTMB_project_2** when you need:
- Fast bulk downloads of finisher data
- Quick UTMB score retrieval
- Less detailed data is acceptable

## Installation

```bash
pip install requests pandas tqdm
```

## Usage

Both projects use the same CLI keywords for consistency:
- `--step 1` or `--step 2`
- `--tenant` (event name)
- `--year`
- `--race`
- `--all` (batch mode)
- `--skip-existing`
- `--cookie` (for authentication)

### Step 1: Scrape Race Results (No Auth Required)

#### List Available Races

```bash
python3 utmb_api_scraper.py --step 1 --tenant kullamannen --list
```

Output:
```
============================================================
Available Races for: kullamannen
============================================================

  2025:
    ✓ Ultra 100 Miles
    ✓ Sprint Ultra 100 km
    ✓ Seventh Seal
    ✓ North Shore

  2024:
    ✓ Ultra 100 Miles
    ...
```

#### Scrape a Single Race

**Note:** Race names with spaces must be quoted.

```bash
# Partial match works - just use part of the race name
python3 utmb_api_scraper.py --step 1 --tenant kullamannen --year 2025 --race "Ultra 100"

# Or use the full race name (must be quoted if it contains spaces)
python3 utmb_api_scraper.py --step 1 --tenant kullamannen --year 2025 --race "Ultra 100 Miles"
```

#### Scrape All Races for a Year

```bash
python3 utmb_api_scraper.py --step 1 --tenant kullamannen --year 2025 --all
```

#### Scrape All Historical Data

```bash
python3 utmb_api_scraper.py --step 1 --tenant kullamannen --all
```

#### Skip Existing Files

```bash
python3 utmb_api_scraper.py --step 1 --tenant kullamannen --all --skip-existing
```

### Step 2: Add UTMB Scores (Auth Required)

UTMB scores require authentication. You need to get a cookie from your browser after logging in to utmb.world.

#### How to Get Your Cookie

1. Log in to [utmb.world](https://utmb.world) in your browser
2. Open Developer Tools (F12)
3. Go to Network tab
4. Refresh the page
5. Click on any request to utmb.world
6. Find "Cookie:" in Request Headers
7. Copy the entire cookie value

#### Add Scores to All CSV Files

```bash
python3 utmb_api_scraper.py --step 2 --cookie "YOUR_COOKIE"
```

#### Add Scores with Filters (tenant/year)

```bash
# Only process files for a specific tenant
python3 utmb_api_scraper.py --step 2 --cookie "YOUR_COOKIE" --tenant kullamannen

# Only process files for a specific year
python3 utmb_api_scraper.py --step 2 --cookie "YOUR_COOKIE" --year 2025

# Combine filters
python3 utmb_api_scraper.py --step 2 --cookie "YOUR_COOKIE" --tenant kullamannen --year 2025
```

#### Skip Files That Already Have Scores

```bash
python3 utmb_api_scraper.py --step 2 --cookie "YOUR_COOKIE" --skip-existing
```

## Output Format

CSV files are saved to `data/` with metadata headers:

```csv
# Race: Kullamannen 2025 - Ultra 100 Miles
# Tenant: kullamannen
# Year: 2025
# Race URI: 10675.kullamannenbyutmbultra100miles.2025
# Total Finishers: 650
# Scraped: 2025-12-03T23:19:28.162259
rank_scratch,name,race_time,race_time_seconds,country,country_code,age_group,sex,utmb_index,utmb_score,runner_uri,utmb_profile_url
1.0,Christian MALMSTROM,16:18:22,58702.0,Sweden,SE,45-49,H,,850,1243141.christian.malmstrom,https://utmb.world/runner/1243141.christian.malmstrom
2.0,Alexandre BOUCHEIX,16:34:44,59684.0,France,FR,20-34,H,,823,758585.alexandre.boucheix,https://utmb.world/runner/758585.alexandre.boucheix
...
```

## Finding Event Tenant Names

The tenant name is the subdomain of the UTMB event website. For example:
- `kullamannen.utmb.world` → tenant is `kullamannen`
- `lavaredo.utmb.world` → tenant is `lavaredo`

## API Details

This scraper uses the UTMB public API endpoints:
- Results: `https://api.utmb.world/races/{race_uri}/results`
- Runner History (for scores): `https://api.utmb.world/runners/{runner_uri}/results`

The race URI is obtained from the event's results page at `https://{tenant}.utmb.world/runners/results`.
