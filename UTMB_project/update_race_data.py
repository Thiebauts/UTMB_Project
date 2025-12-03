#!/usr/bin/env python3
"""
=============================================================================
UTMB Race Data Updater
=============================================================================

Updates events.json with race data from live.utmb.world.

USAGE:
------
# Update a specific event for a specific year
python3 update_race_data.py --event kullamannen --year 2025

# Update a specific event for all available years (scans 2018-current)
python3 update_race_data.py --event kullamannen --all-years

# Update ALL known UTMB World Series events (all years)
python3 update_race_data.py --all-events

# List all events in the database
python3 update_race_data.py --list

# List all known UTMB World Series event tenants
python3 update_race_data.py --list-known

# Show races for a specific event/year
python3 update_race_data.py --show kullamannen 2025

=============================================================================
"""

import requests
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

EVENTS_FILE = Path(__file__).parent / 'events.json'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'text/html,application/json',
}

# Year range for --all-years option
MIN_YEAR = 2018
MAX_YEAR = datetime.now().year + 1  # Include next year for upcoming events

# =============================================================================
# KNOWN UTMB WORLD SERIES EVENTS
# =============================================================================
# Complete list of all UTMB World Series event tenants (as of 2025)
# Tenant names are used in URLs: https://live.utmb.world/{tenant}/{year}
#
# NOTE: The tenant names on live.utmb.world are INCONSISTENT and don't follow
# a predictable pattern. Some use short names, some add "byutmb", some use
# full event names. These have been manually verified.

UTMB_EVENTS = {
    # =========================================================================
    # EUROPE - VERIFIED
    # =========================================================================
    'arcofattrition': 'Arc of Attrition (UK)',
    'chianti': 'Chianti Ultra Trail (Italy)',
    'tenerifebyutmb': 'Tenerife by UTMB (Spain)',  # NOT 'tenerife'
    '100milesofistria': 'Istria 100 by UTMB (Croatia)',  # NOT 'istria'
    'grandraidventoux': 'Grand Raid Ventoux (France)',  # NOT 'ventoux'
    'ohmeudeus': 'Oh Meu Deus (Portugal)',  # First event in 2026
    'alsacegrandest': 'Trail Alsace Grand Est (France)',
    'uts': 'Ultra Trail Snowdonia (UK)',  # NOT 'snowdonia'
    'mozart100': 'Mozart 100 by UTMB (Austria)',  # NOT 'mozart'
    'andorrabyutmb': 'Trail 100 Andorra by UTMB (Andorra)',  # NOT 'andorra'
    'tsj': 'Trail de Saint-Jacques (France)',  # NOT 'saint-jacques'
    'zugspitz': 'Zugspitz Ultra Trail (Germany)',  # First event in 2026
    'lavaredo': 'Lavaredo Ultra Trail (Italy)',
    'aranbyutmb': "Val d'Aran by UTMB (Spain)",  # NOT 'valdaran'
    'restonica': 'Restonica Trail (France)',
    'tvsb': 'Trail Verbier St-Bernard (Switzerland)',  # NOT 'verbier'
    'eigerultratrail': 'Eiger Ultra Trail (Switzerland)',  # NOT 'eiger'
    'mrwwbyutmb': 'Monte Rosa Walser Way (Italy)',  # NOT 'mrww'
    'bucovina': 'Bucovina Ultra Rocks (Romania)',  # First event in 2026
    'gauja': 'Gauja Trail (Latvia)',  # First event in 2026
    'kat100': 'KAT by UTMB (Austria)',  # NOT 'kat'
    'utmb': 'UTMB Mont-Blanc (France)',  # NOT 'montblanc'
    'wildstrubel': 'Wildstrubel by UTMB (Switzerland)',
    'kackar': 'Kaçkar by UTMB (Turkey)',
    'julianalps': 'Julian Alps Trail Run (Slovenia)',  # NOT 'julianalpls'
    'nicebyutmb': 'Nice Côte d\'Azur by UTMB (France)',  # NOT 'nice'
    'kullamannen': 'Trail Kullamannen (Sweden)',
    'puglia': 'Puglia by UTMB (Italy)',
    'mallorcabyutmb': 'Mallorca by UTMB (Spain)',  # NOT 'mallorca'
    
    # =========================================================================
    # OCEANIA - VERIFIED
    # =========================================================================
    'tarawera': 'Tarawera Ultra (New Zealand)',
    'uta': 'Ultra-Trail Australia (Australia)',
    'kosciuszko': 'Ultra-Trail Kosciuszko (Australia)',
    
    # =========================================================================
    # NORTH AMERICA - VERIFIED
    # =========================================================================
    'puertovallarta': 'Puerto Vallarta by UTMB (Mexico)',  # NOT 'puerto-vallarta'
    'desertrats': 'Desert Rats by UTMB (USA)',
    'canyons': 'Canyons by UTMB (USA)',
    'rothrock': 'Rothrock by UTMB (USA)',  # First event in 2026
    'speedgoatmountainraces': 'Speedgoat by UTMB (USA)',  # NOT 'speedgoat'
    'borealys': 'Boréalys by UTMB (Canada)',  # First event in 2026
    'grindstone': 'Grindstone by UTMB (USA)',
    'kodiak': 'Kodiak by UTMB (USA)',
    'pacifictrails': 'Pacific Trails by UTMB (USA)',
    'chihuahuabyutmb': 'Chihuahua by UTMB (Mexico)',  # NOT 'chihuahua'
    'whistler': 'Whistler by UTMB (Canada)',
    
    # =========================================================================
    # SOUTH AMERICA - VERIFIED
    # =========================================================================
    'valholl': 'Ushuaia by UTMB (Argentina)',
    'torrencialbyutmb': 'Torrencial by UTMB (Chile)',  # NOT 'torrencial'
    'quitobyutmb': 'Quito by UTMB (Ecuador)',  # NOT 'quito'
    'paraty': 'Paraty by UTMB (Brazil)',
    'barilochebyutmb': 'Bariloche by UTMB (Argentina)',  # NOT 'bariloche'
    
    # =========================================================================
    # ASIA - VERIFIED
    # =========================================================================
    'xtrail': 'X-Trail by UTMB (Philippines)',  # First event in 2026
    'amazean': 'Amazean by UTMB (Indonesia)',  # First event in 2026
    'kagapa': 'Kagapa by UTMB (Philippines)',  # First event in 2026
    'malaysia': 'Malaysia Ultra Trail (Malaysia)',
    'oman': 'Oman by UTMB (Oman)',  # First event in 2026
    'chiangmai': 'Chiang Mai Thailand by UTMB (Thailand)',  # NOT 'thailand'
    'translantau': 'Translantau by UTMB (Hong Kong)',
    'shudao': 'Shudao by UTMB (China)',  # First event in 2026
    'transjeju': 'Trans Jeju by UTMB (South Korea)',
    'laketoba': 'Lake Toba by UTMB (Indonesia)',
    'dajingmen': 'Da Jing Men by UTMB (China)',
    'mountyun': 'Mount Yun by UTMB (China)',  # NOT 'mount-yun'
    'xiamen': 'Xiamen by UTMB (China)',
    
    # =========================================================================
    # AFRICA - VERIFIED
    # =========================================================================
    'mut': 'MUT by UTMB (South Africa)',
}


# =============================================================================
# DATA LOADING/SAVING
# =============================================================================

def load_events_data() -> dict:
    """Load events data from JSON file."""
    if EVENTS_FILE.exists():
        with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "_metadata": {
            "description": "UTMB World Series race data by event and year",
            "last_updated": None,
            "source": "live.utmb.world"
        },
        "events": {}
    }


def save_events_data(data: dict):
    """Save events data to JSON file."""
    data["_metadata"]["last_updated"] = datetime.now().isoformat()
    with open(EVENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to {EVENTS_FILE.name}")


# =============================================================================
# FETCHING FROM LIVE.UTMB.WORLD
# =============================================================================

def fetch_event_year(event: str, year: int) -> dict:
    """
    Fetch race data for a specific event and year from live.utmb.world.
    
    Args:
        event: Event tenant name (e.g., 'kullamannen', 'utmb')
        year: Event year
    
    Returns:
        dict with event info and races, or None if not found/error
    """
    url = f"https://live.utmb.world/{event}/{year}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            return None
        
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            response.text,
            re.DOTALL
        )
        if not match:
            return None
        
        data = json.loads(match.group(1))
        page_props = data.get('props', {}).get('pageProps', {})
        
        # Handle cases where these might be False/None instead of dict
        current_event = page_props.get('currentEvent')
        homepage = page_props.get('homepage')
        
        # Ensure they are dicts (some pages return False for these)
        if not isinstance(current_event, dict):
            current_event = {}
        if not isinstance(homepage, dict):
            homepage = {}
        
        # Get event name
        event_name = homepage.get('eventName') or current_event.get('title') or event.title()
        
        # Get event metadata
        event_info = {
            'country': current_event.get('country'),
            'country_code': current_event.get('countryCode'),
            'continent': current_event.get('continent'),
            'date_start': current_event.get('dateStart'),
            'date_end': current_event.get('dateEnd'),
        }
        
        # Build races dict
        races = {}
        for race in homepage.get('races', []):
            race_id = race.get('raceId')
            distance = race.get('distance')
            
            if not race_id or not distance:
                continue
            
            # Skip non-competitive races
            race_name = race.get('name') or ''
            if any(skip in race_name.lower() for skip in ['family', 'youth', 'promo', 'kids', 'enfant']):
                continue
            
            races[race_id] = {
                'name': race_name,
                'distance_km': distance,
                'elevation_gain': race.get('elevationGain'),
                'elevation_loss': race.get('elevationLoss') or race.get('elevationGain'),
                'utmb_category': race.get('category'),
                'running_stones': race.get('stones'),
                'start_date': race.get('startDate'),  # e.g., "2025-10-31T17:00:11.000Z"
                'start_place': race.get('startPlace'),  # e.g., "Hoganas"
            }
        
        if not races:
            return None
        
        return {
            'event_name': event_name,
            'event_info': event_info,
            'races': races,
            'fetched_at': datetime.now().isoformat(),
        }
        
    except Exception as e:
        print(f"    Error fetching {event}/{year}: {e}")
        return None


def update_event_year(event: str, year: int, save: bool = True) -> bool:
    """
    Update events.json with data for a specific event and year.
    
    Args:
        event: Event tenant name
        year: Event year
        save: Whether to save to file
    
    Returns:
        True if successful, False otherwise
    """
    print(f"  Fetching {event}/{year}...")
    
    year_data = fetch_event_year(event, year)
    
    if not year_data:
        print(f"    No data found for {event}/{year}")
        return False
    
    # Load existing data
    data = load_events_data()
    
    # Initialize event if needed
    if event not in data["events"]:
        data["events"][event] = {
            "event_name": year_data['event_name'],
            "years": {}
        }
    
    # Update event name (use most recent)
    data["events"][event]["event_name"] = year_data['event_name']
    
    # Add year data
    data["events"][event]["years"][str(year)] = {
        "event_info": year_data['event_info'],
        "races": year_data['races'],
        "fetched_at": year_data['fetched_at'],
    }
    
    # Print summary
    races = year_data['races']
    print(f"    ✓ Found {len(races)} races:")
    for race_id, race in sorted(races.items()):
        print(f"      - {race_id}: {race['name']} ({race['distance_km']}km, D+{race['elevation_gain']}m)")
    
    # Save if requested
    if save:
        save_events_data(data)
    
    return True


def update_event_all_years(event: str, save: bool = True) -> dict:
    """
    Update events.json with data for all available years of an event.
    
    Args:
        event: Event tenant name
        save: Whether to save after completion
    
    Returns:
        dict with years found
    """
    print(f"\nScanning {event} for years {MIN_YEAR}-{MAX_YEAR}...")
    
    data = load_events_data()
    years_found = {}
    
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        year_data = fetch_event_year(event, year)
        
        if year_data:
            years_found[year] = len(year_data['races'])
            
            # Initialize event if needed
            if event not in data["events"]:
                data["events"][event] = {
                    "event_name": year_data['event_name'],
                    "years": {}
                }
            
            # Update event name
            data["events"][event]["event_name"] = year_data['event_name']
            
            # Add year data
            data["events"][event]["years"][str(year)] = {
                "event_info": year_data['event_info'],
                "races": year_data['races'],
                "fetched_at": year_data['fetched_at'],
            }
            
            print(f"  ✓ {year}: {len(year_data['races'])} races")
    
    if years_found:
        if save:
            save_events_data(data)
        print(f"\n✓ Found data for {len(years_found)} years: {list(years_found.keys())}")
    else:
        print(f"\n✗ No data found for {event}")
    
    return years_found


def update_all_events(skip_existing: bool = False):
    """
    Update events.json with data for ALL known UTMB World Series events.
    
    Args:
        skip_existing: If True, skip events already in database
    """
    print(f"\n{'=' * 70}")
    print(f"FETCHING ALL UTMB WORLD SERIES EVENTS ({len(UTMB_EVENTS)} events)")
    print('=' * 70)
    
    data = load_events_data()
    results = {
        'success': [],
        'failed': [],
        'skipped': [],
    }
    
    for i, (tenant, name) in enumerate(UTMB_EVENTS.items(), 1):
        print(f"\n[{i}/{len(UTMB_EVENTS)}] {name}")
        print("-" * 50)
        
        # Check if should skip
        if skip_existing and tenant in data.get("events", {}):
            existing_years = list(data["events"][tenant].get("years", {}).keys())
            print(f"  Skipping (already in database with years: {existing_years})")
            results['skipped'].append(tenant)
            continue
        
        # Fetch all years for this event
        years_found = {}
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            year_data = fetch_event_year(tenant, year)
            
            if year_data:
                years_found[year] = len(year_data['races'])
                
                # Initialize event if needed
                if tenant not in data["events"]:
                    data["events"][tenant] = {
                        "event_name": year_data['event_name'],
                        "years": {}
                    }
                
                # Update event name
                data["events"][tenant]["event_name"] = year_data['event_name']
                
                # Add year data
                data["events"][tenant]["years"][str(year)] = {
                    "event_info": year_data['event_info'],
                    "races": year_data['races'],
                    "fetched_at": year_data['fetched_at'],
                }
                
                print(f"  ✓ {year}: {len(year_data['races'])} races")
        
        if years_found:
            results['success'].append((tenant, list(years_found.keys())))
        else:
            results['failed'].append(tenant)
        
        # Save after each event (in case of interruption)
        save_events_data(data)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print('=' * 70)
    print(f"\n✓ Successfully fetched: {len(results['success'])} events")
    for tenant, years in results['success']:
        print(f"    {tenant}: {years}")
    
    if results['skipped']:
        print(f"\n⊘ Skipped (already in DB): {len(results['skipped'])} events")
        for tenant in results['skipped']:
            print(f"    {tenant}")
    
    if results['failed']:
        print(f"\n✗ Failed (no data found): {len(results['failed'])} events")
        for tenant in results['failed']:
            print(f"    {tenant}")
    
    print()


# =============================================================================
# QUERY FUNCTIONS
# =============================================================================

def list_events():
    """List all events in the database."""
    data = load_events_data()
    events = data.get("events", {})
    
    if not events:
        print("\nNo events in database. Use --event <name> --year <year> to add one.")
        print(f"Or use --all-events to fetch all {len(UTMB_EVENTS)} known UTMB World Series events.")
        return
    
    print(f"\nEvents in database ({len(events)} events):")
    print("-" * 80)
    
    for event_id, event_data in sorted(events.items()):
        event_name = event_data.get("event_name", event_id)
        years = sorted(event_data.get("years", {}).keys())
        
        if years:
            year_range = f"{years[0]}-{years[-1]}" if len(years) > 1 else years[0]
            total_races = sum(
                len(event_data["years"][y].get("races", {})) 
                for y in years
            )
            print(f"  {event_id:<20} | {event_name:<35} | Years: {year_range:<10} | {total_races} races")
        else:
            print(f"  {event_id:<20} | {event_name:<35} | No years")
    
    print()
    print(f"Tip: Use --list-known to see all {len(UTMB_EVENTS)} known UTMB World Series events.")


def list_known_events():
    """List all known UTMB World Series event tenants."""
    print(f"\nKnown UTMB World Series Events ({len(UTMB_EVENTS)} events):")
    print("=" * 80)
    
    # Group by region
    regions = {
        'EUROPE': [],
        'OCEANIA': [],
        'NORTH AMERICA': [],
        'SOUTH AMERICA': [],
        'ASIA': [],
        'AFRICA': [],
    }
    
    # Categorize based on the order in UTMB_EVENTS
    current_region = 'EUROPE'
    for tenant, name in UTMB_EVENTS.items():
        if tenant == 'tarawera':
            current_region = 'OCEANIA'
        elif tenant == 'puerto-vallarta':
            current_region = 'NORTH AMERICA'
        elif tenant == 'valholl':
            current_region = 'SOUTH AMERICA'
        elif tenant == 'xtrail':
            current_region = 'ASIA'
        elif tenant == 'mut':
            current_region = 'AFRICA'
        regions[current_region].append((tenant, name))
    
    # Load database to check which are already fetched
    data = load_events_data()
    db_events = data.get("events", {})
    
    for region, events in regions.items():
        if events:
            print(f"\n{region}:")
            print("-" * 70)
            for tenant, name in events:
                in_db = "✓" if tenant in db_events else " "
                years_info = ""
                if tenant in db_events:
                    years = sorted(db_events[tenant].get("years", {}).keys())
                    if years:
                        years_info = f" [{years[0]}-{years[-1]}]" if len(years) > 1 else f" [{years[0]}]"
                print(f"  [{in_db}] {tenant:<20} - {name}{years_info}")
    
    print()
    print("Legend: [✓] = in database, [ ] = not yet fetched")
    print(f"\nUse --all-events to fetch all events, or --event <tenant> --all-years for a specific one.")


def show_event_year(event: str, year: int):
    """Show races for a specific event and year."""
    data = load_events_data()
    
    if event not in data.get("events", {}):
        print(f"\nEvent '{event}' not found. Use --event {event} --year {year} to fetch it.")
        return
    
    event_data = data["events"][event]
    year_str = str(year)
    
    if year_str not in event_data.get("years", {}):
        available_years = list(event_data.get("years", {}).keys())
        print(f"\nYear {year} not found for {event}.")
        print(f"Available years: {available_years}")
        return
    
    year_data = event_data["years"][year_str]
    races = year_data.get("races", {})
    event_info = year_data.get("event_info", {})
    
    print(f"\n{event_data['event_name']} - {year}")
    print("=" * 60)
    
    if event_info.get('country'):
        print(f"Location: {event_info['country']} ({event_info.get('country_code', '')})")
    if event_info.get('date_start'):
        print(f"Date: {event_info['date_start']}")
    
    print(f"\nRaces ({len(races)}):")
    print("-" * 100)
    print(f"{'ID':<10} | {'Name':<25} | {'Distance':>10} | {'D+':>8} | {'Start Date/Time':<20} | {'Start Place':<15}")
    print("-" * 100)
    
    for race_id, race in sorted(races.items(), key=lambda x: x[1].get('distance_km', 0), reverse=True):
        dist = f"{race['distance_km']} km"
        elev = f"{race['elevation_gain']} m" if race.get('elevation_gain') else "N/A"
        
        # Format start date/time
        start_dt = race.get('start_date')
        if start_dt:
            # Parse ISO format and display nicely
            try:
                from datetime import datetime as dt
                parsed = dt.fromisoformat(start_dt.replace('Z', '+00:00'))
                start_str = parsed.strftime('%Y-%m-%d %H:%M')
            except:
                start_str = start_dt[:16] if len(start_dt) > 16 else start_dt
        else:
            start_str = 'N/A'
        
        start_place = race.get('start_place') or 'N/A'
        print(f"{race_id:<10} | {race['name']:<25} | {dist:>10} | {elev:>8} | {start_str:<20} | {start_place:<15}")
    
    print()


def get_race_metadata(event: str, year: int, race_id: str) -> dict:
    """
    Get metadata for a specific race.
    
    This is the main function used by utmb_scraper.py.
    
    Args:
        event: Event tenant name
        year: Event year
        race_id: Race ID
    
    Returns:
        dict with race metadata or None if not found
    """
    data = load_events_data()
    
    event_data = data.get("events", {}).get(event, {})
    year_data = event_data.get("years", {}).get(str(year), {})
    races = year_data.get("races", {})
    
    if race_id not in races:
        return None
    
    race = races[race_id]
    event_info = year_data.get("event_info", {})
    
    return {
        'event_name': event_data.get('event_name', event),
        'race_name': f"{event_data.get('event_name', event)} {year} - {race['name']}",
        'distance_km': race.get('distance_km'),
        'elevation_gain': race.get('elevation_gain'),
        'elevation_loss': race.get('elevation_loss'),
        'utmb_category': race.get('utmb_category'),
        'date': event_info.get('date_start'),
        'start_date': race.get('start_date'),  # Race-specific start datetime (ISO format)
        'start_place': race.get('start_place'),  # Race start location
        'country': event_info.get('country'),
        'url': f"https://live.utmb.world/{event}/{year}",
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UTMB Race Data Updater',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--event', '-e', type=str,
                        help='Event tenant name (e.g., kullamannen, utmb)')
    parser.add_argument('--year', '-y', type=int,
                        help='Specific year to fetch')
    parser.add_argument('--all-years', '-a', action='store_true',
                        help='Fetch all available years for the event')
    parser.add_argument('--all-events', action='store_true',
                        help='Fetch ALL known UTMB World Series events (all years)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='With --all-events: skip events already in database')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all events in database')
    parser.add_argument('--list-known', action='store_true',
                        help='List all known UTMB World Series event tenants')
    parser.add_argument('--show', '-s', nargs=2, metavar=('EVENT', 'YEAR'),
                        help='Show races for a specific event and year')
    
    args = parser.parse_args()
    
    if args.list:
        list_events()
    
    elif args.list_known:
        list_known_events()
    
    elif args.show:
        event, year = args.show
        show_event_year(event, int(year))
    
    elif args.all_events:
        update_all_events(skip_existing=args.skip_existing)
    
    elif args.event:
        if args.all_years:
            print(f"\n{'=' * 60}")
            print(f"Updating {args.event} - All Years")
            print('=' * 60)
            update_event_all_years(args.event)
        
        elif args.year:
            print(f"\n{'=' * 60}")
            print(f"Updating {args.event}/{args.year}")
            print('=' * 60)
            update_event_year(args.event, args.year)
        
        else:
            print("Error: --event requires either --year or --all-years")
            parser.print_help()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

