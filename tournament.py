#!/usr/bin/env python3
'''
Tournament runner for IIT Pokerbots.

Usage:
  a) Round-robin (all pairs):
       python tournament.py --mode all --matches 3

  b) One bot vs all others:
       python tournament.py --mode one --bot mahoraga.py --matches 3

Options:
  --mode       "all" for round-robin, "one" for one-vs-all
  --bot        Bot filename (inside bots/) to test against all others (required if mode=one)
  --matches    Number of matches per pair (default: 3)
  --bots-dir   Directory containing bots (default: ./bots)
  --output     CSV output file (default: tournament_results.csv)
'''

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path


# ─── Constants ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_PATH = SCRIPT_DIR / 'engine.py'
CONFIG_PATH = SCRIPT_DIR / 'config.py'
DEFAULT_BOTS_DIR = SCRIPT_DIR / 'bots'


# ─── Helpers ────────────────────────────────────────────────────────────────────

def discover_bots(bots_dir: Path) -> list[tuple[str, str]]:
    """Return list of (bot_name, bot_filepath) for every .py file in bots_dir."""
    bots = []
    for f in sorted(bots_dir.iterdir()):
        if f.suffix == '.py' and f.is_file():
            name = f.stem  # filename without .py
            bots.append((name, str(f.resolve())))
    return bots


def write_temp_config(bot1_name, bot1_file, bot2_name, bot2_file, log_folder):
    """Write a temporary config.py that the engine will import."""
    # Use forward slashes to avoid unescaped backslash syntax errors on Windows
    py_cmd    = sys.executable.replace('\\', '/')
    b1_file   = bot1_file.replace('\\', '/')
    b2_file   = bot2_file.replace('\\', '/')
    log_dir   = log_folder.replace('\\', '/') if isinstance(log_folder, str) else str(log_folder).replace('\\', '/')
    content = textwrap.dedent(f"""\
        PYTHON_CMD = "{py_cmd}"
        BOT_1_NAME = '{bot1_name}'
        BOT_1_FILE = '{b1_file}'
        BOT_2_NAME = '{bot2_name}'
        BOT_2_FILE = '{b2_file}'
        GAME_LOG_FOLDER = '{log_dir}'
    """)
    CONFIG_PATH.write_text(content)


def parse_engine_output(stdout: str) -> dict:
    """
    Parse the engine's stdout and return a dict with stats for both bots.
    Returns: {
        'bot1': { 'name', 'bankroll', 'win_rate', 'avg_payoff', 'auction_win_rate',
                  'avg_bid', 'bid_var', 'avg_query_time', 'avg_hand_time', 'max_time' },
        'bot2': { ... },
        'match_time': float,
    }
    """
    result = {}
    # Find all bot stat blocks
    bot_blocks = re.findall(
        r'Stats for (.+?):\s*\n'
        r'\s*Total Bankroll:\s*([-\d]+)\s*\n'
        r'.*?\n'
        r'\s*Win Rate:\s*([\d.]+)%\s*\n'
        r'\s*Avg Payoff/Hand:\s*([-\d.]+)\s*\n'
        r'.*?\n'
        r'\s*Auction Win Rate:\s*([\d.]+)%\s*\n'
        r'\s*Avg Bid Amount \(Mean, Var\):\s*\(([-\d.]+),\s*([-\d.]+)\)\s*\n'
        r'.*?\n'
        r'\s*Avg Response Time \(Query\):\s*([-\d.]+)s\s*\n'
        r'\s*Avg Response Time \(Hand\):\s*([-\d.]+)s\s*\n'
        r'\s*Max Response Time:\s*([-\d.]+)s',
        stdout
    )

    keys = ['bot1', 'bot2']
    for i, block in enumerate(bot_blocks[:2]):
        name, bankroll, win_rate, avg_payoff, auction_wr, avg_bid, bid_var, avg_q, avg_h, max_t = block
        result[keys[i]] = {
            'name': name,
            'bankroll': int(bankroll),
            'win_rate': float(win_rate),
            'avg_payoff': float(avg_payoff),
            'auction_win_rate': float(auction_wr),
            'avg_bid': float(avg_bid),
            'bid_var': float(bid_var),
            'avg_query_time': float(avg_q),
            'avg_hand_time': float(avg_h),
            'max_time': float(max_t),
        }

    match_time_m = re.search(r'Total Match Time:\s*([\d.]+)s', stdout)
    result['match_time'] = float(match_time_m.group(1)) if match_time_m else 0.0

    return result


def run_single_match(bot1_name, bot1_file, bot2_name, bot2_file, match_num, log_folder) -> dict | None:
    """Run one match between two bots and return parsed stats."""
    write_temp_config(bot1_name, bot1_file, bot2_name, bot2_file, log_folder)
    # Set PYTHONPATH so bots in subdirectories can still import pkbot
    env = os.environ.copy()
    project_root = str(SCRIPT_DIR)
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    try:
        proc = subprocess.run(
            [sys.executable, str(ENGINE_PATH), '--small_log'],
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if proc.returncode != 0:
            print(f"    ⚠ Match {match_num} FAILED (exit code {proc.returncode})")
            if proc.stderr:
                print(f"      stderr: {proc.stderr[:300]}")
            return None
        return parse_engine_output(proc.stdout)
    except subprocess.TimeoutExpired:
        print(f"    ⚠ Match {match_num} TIMED OUT")
        return None
    except Exception as e:
        print(f"    ⚠ Match {match_num} ERROR: {e}")
        return None


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Tournament runner for IIT Pokerbots')
    parser.add_argument('--mode', choices=['all', 'one'], required=True,
                        help='"all" for round-robin, "one" for one-vs-all')
    parser.add_argument('--bot', type=str, default=None,
                        help='Bot filename (e.g. mahoraga.py) to test against all others (mode=one)')
    parser.add_argument('--matches', type=int, default=3,
                        help='Number of matches per pair (default: 3)')
    parser.add_argument('--bots-dir', type=str, default=str(DEFAULT_BOTS_DIR),
                        help='Directory containing bot .py files (default: ./bots)')
    parser.add_argument('--output', type=str, default='tournament_results.csv',
                        help='CSV output file (default: tournament_results.csv)')
    args = parser.parse_args()

    bots_dir = Path(args.bots_dir).resolve()
    if not bots_dir.is_dir():
        print(f"Error: bots directory '{bots_dir}' does not exist.")
        sys.exit(1)

    bots = discover_bots(bots_dir)
    if len(bots) < 2 and args.mode == 'all':
        print(f"Error: need at least 2 bots in '{bots_dir}', found {len(bots)}.")
        sys.exit(1)

    bot_names = {name for name, _ in bots}
    print(f"Discovered {len(bots)} bot(s): {', '.join(n for n, _ in bots)}")

    # Build matchups
    if args.mode == 'all':
        matchups = list(itertools.combinations(bots, 2))
    else:
        if args.bot is None:
            print("Error: --bot is required in mode 'one'.")
            sys.exit(1)
        target_stem = Path(args.bot).stem
        target = None
        opponents = []
        for name, path in bots:
            if name == target_stem:
                target = (name, path)
            else:
                opponents.append((name, path))
        if target is None:
            print(f"Error: bot '{args.bot}' not found in '{bots_dir}'.")
            sys.exit(1)
        if not opponents:
            print(f"Error: no opponents found for '{target_stem}' in '{bots_dir}'.")
            sys.exit(1)
        matchups = [(target, opp) for opp in opponents]

    # Backup original config
    original_config = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else None

    # Log folder for tournament
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_folder = str(SCRIPT_DIR / 'logs' / f'tournament_{timestamp}')
    os.makedirs(log_folder, exist_ok=True)

    total_matches = len(matchups) * args.matches
    print(f"\nRunning {total_matches} matches ({len(matchups)} pairs × {args.matches} each)\n")
    print("=" * 70)

    # Collect all results
    all_results = []
    aggregated = {}  # (bot1, bot2) -> list of result dicts

    for pair_idx, ((b1_name, b1_file), (b2_name, b2_file)) in enumerate(matchups, 1):
        pair_key = (b1_name, b2_name)
        aggregated[pair_key] = []
        print(f"\n[{pair_idx}/{len(matchups)}] {b1_name} vs {b2_name}")

        for m in range(1, args.matches + 1):
            print(f"  Match {m}/{args.matches}...", end=' ', flush=True)
            stats = run_single_match(b1_name, b1_file, b2_name, b2_file, m, log_folder)
            if stats and 'bot1' in stats and 'bot2' in stats:
                b1 = stats['bot1']
                b2 = stats['bot2']
                winner = b1_name if b1['bankroll'] > b2['bankroll'] else (b2_name if b2['bankroll'] > b1['bankroll'] else 'TIE')
                row = {
                    'pair': f"{b1_name} vs {b2_name}",
                    'match': m,
                    'bot1': b1_name, 'bot2': b2_name,
                    'bot1_bankroll': b1['bankroll'], 'bot2_bankroll': b2['bankroll'],
                    'bot1_win_rate': b1['win_rate'], 'bot2_win_rate': b2['win_rate'],
                    'bot1_avg_payoff': b1['avg_payoff'], 'bot2_avg_payoff': b2['avg_payoff'],
                    'bot1_auction_wr': b1['auction_win_rate'], 'bot2_auction_wr': b2['auction_win_rate'],
                    'bot1_avg_bid': b1['avg_bid'], 'bot2_avg_bid': b2['avg_bid'],
                    'bot1_avg_query_time': b1['avg_query_time'], 'bot2_avg_query_time': b2['avg_query_time'],
                    'bot1_max_time': b1['max_time'], 'bot2_max_time': b2['max_time'],
                    'match_time': stats['match_time'],
                    'winner': winner,
                }
                all_results.append(row)
                aggregated[pair_key].append(row)
                margin = abs(b1['bankroll'])
                print(f"✓  {winner} wins (bankroll: {b1['bankroll']:+d} / {b2['bankroll']:+d}, margin: {margin})")
            else:
                print("✗  (no valid result)")

    # Restore original config
    if original_config is not None:
        CONFIG_PATH.write_text(original_config)

    # ─── Write CSV ──────────────────────────────────────────────────────────
    csv_path = SCRIPT_DIR / args.output
    if all_results:
        fieldnames = all_results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✓ Results saved to {csv_path}")

    # ─── Pretty-print summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TOURNAMENT SUMMARY")
    print("=" * 70)

    # Per-pair summary
    for (b1_name, b2_name), results in aggregated.items():
        if not results:
            continue
        n = len(results)
        b1_wins = sum(1 for r in results if r['winner'] == b1_name)
        b2_wins = sum(1 for r in results if r['winner'] == b2_name)
        ties = n - b1_wins - b2_wins
        avg_b1_bankroll = sum(r['bot1_bankroll'] for r in results) / n
        avg_b2_bankroll = sum(r['bot2_bankroll'] for r in results) / n
        avg_b1_wr = sum(r['bot1_win_rate'] for r in results) / n
        avg_b2_wr = sum(r['bot2_win_rate'] for r in results) / n
        avg_b1_auction = sum(r['bot1_auction_wr'] for r in results) / n
        avg_b2_auction = sum(r['bot2_auction_wr'] for r in results) / n

        print(f"\n┌─ {b1_name} vs {b2_name}  ({n} matches)")
        print(f"│")
        print(f"│  Series Score: {b1_name} {b1_wins} - {b2_wins} {b2_name}" + (f" ({ties} tie{'s' if ties > 1 else ''})" if ties else ""))
        print(f"│")
        print(f"│  {'Metric':<25} {'':>3} {b1_name:>15} {b2_name:>15}")
        print(f"│  {'─'*25} {'':>3} {'─'*15} {'─'*15}")
        print(f"│  {'Avg Bankroll':<25} {'':>3} {avg_b1_bankroll:>+15.0f} {avg_b2_bankroll:>+15.0f}")
        print(f"│  {'Avg Win Rate':<25} {'':>3} {avg_b1_wr:>14.1f}% {avg_b2_wr:>14.1f}%")
        print(f"│  {'Avg Auction Win Rate':<25} {'':>3} {avg_b1_auction:>14.1f}% {avg_b2_auction:>14.1f}%")

        # Per-match breakdown
        print(f"│")
        print(f"│  Match-by-match:")
        for r in results:
            m = r['match']
            w = r['winner']
            print(f"│    Match {m}: {r['bot1_bankroll']:>+7d} / {r['bot2_bankroll']:>+7d}  → {w}")
        print(f"└{'─'*69}")

    # ─── Leaderboard ────────────────────────────────────────────────────────
    if len(aggregated) > 1 or args.mode == 'one':
        # Aggregate per-bot stats
        bot_stats = {}  # name -> { 'wins', 'losses', 'ties', 'total_bankroll', 'matches' }
        for (b1_name, b2_name), results in aggregated.items():
            for name in [b1_name, b2_name]:
                if name not in bot_stats:
                    bot_stats[name] = {'wins': 0, 'losses': 0, 'ties': 0, 'total_bankroll': 0, 'matches': 0}
            for r in results:
                bot_stats[b1_name]['matches'] += 1
                bot_stats[b2_name]['matches'] += 1
                bot_stats[b1_name]['total_bankroll'] += r['bot1_bankroll']
                bot_stats[b2_name]['total_bankroll'] += r['bot2_bankroll']
                if r['winner'] == b1_name:
                    bot_stats[b1_name]['wins'] += 1
                    bot_stats[b2_name]['losses'] += 1
                elif r['winner'] == b2_name:
                    bot_stats[b2_name]['wins'] += 1
                    bot_stats[b1_name]['losses'] += 1
                else:
                    bot_stats[b1_name]['ties'] += 1
                    bot_stats[b2_name]['ties'] += 1

        # Sort by total bankroll
        leaderboard = sorted(bot_stats.items(), key=lambda x: x[1]['total_bankroll'], reverse=True)

        print(f"\n{'=' * 70}")
        print("  LEADERBOARD")
        print(f"{'=' * 70}")
        print(f"  {'#':<4} {'Bot':<20} {'W-L-T':<12} {'Total Bankroll':>16} {'Avg Bankroll':>14}")
        print(f"  {'─'*4} {'─'*20} {'─'*12} {'─'*16} {'─'*14}")
        for rank, (name, s) in enumerate(leaderboard, 1):
            wlt = f"{s['wins']}-{s['losses']}-{s['ties']}"
            avg = s['total_bankroll'] / s['matches'] if s['matches'] else 0
            print(f"  {rank:<4} {name:<20} {wlt:<12} {s['total_bankroll']:>+16d} {avg:>+14.0f}")
        print()

    if not all_results:
        print("\n  No valid results were recorded.\n")


if __name__ == '__main__':
    main()
