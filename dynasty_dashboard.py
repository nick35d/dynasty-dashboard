import streamlit as st
import requests
import pandas as pd
from functools import lru_cache

# -----------------------------
# Config
# -----------------------------

st.set_page_config(page_title="Dynasty League Dashboard", layout="wide")

SLEEPER_BASE = "https://api.sleeper.app/v1"
LEAGUE_ID_CURRENT = "917164805700575232"  # your current league id


# -----------------------------
# Sleeper API helpers
# -----------------------------

@lru_cache(maxsize=None)
def get_league(league_id: str):
    return requests.get(f"{SLEEPER_BASE}/league/{league_id}").json()


@lru_cache(maxsize=None)
def get_league_rosters(league_id: str):
    return requests.get(f"{SLEEPER_BASE}/league/{league_id}/rosters").json()


@lru_cache(maxsize=None)
def get_league_users(league_id: str):
    return requests.get(f"{SLEEPER_BASE}/league/{league_id}/users").json()


@lru_cache(maxsize=None)
def get_league_matchups(league_id: str, week: int):
    return requests.get(f"{SLEEPER_BASE}/league/{league_id}/matchups/{week}").json()


@lru_cache(maxsize=None)
def get_league_drafts(league_id: str):
    return requests.get(f"{SLEEPER_BASE}/league/{league_id}/drafts").json()


@lru_cache(maxsize=None)
def get_draft_picks(draft_id: str):
    return requests.get(f"{SLEEPER_BASE}/draft/{draft_id}/picks").json()


@lru_cache(maxsize=None)
def get_draft_traded_picks(draft_id: str):
    return requests.get(f"{SLEEPER_BASE}/draft/{draft_id}/traded_picks").json()


@lru_cache(maxsize=None)
def get_league_transactions(league_id: str, week: int):
    return requests.get(f"{SLEEPER_BASE}/league/{league_id}/transactions/{week}").json()


# -----------------------------
# League chain + mapping helpers
# -----------------------------

def walk_league_chain(latest_league_id: str):
    leagues = []
    current = latest_league_id
    while current:
        league = get_league(current)
        leagues.append(league)
        current = league.get("previous_league_id")
    leagues.reverse()
    return leagues


def build_roster_user_maps(league):
    rosters = get_league_rosters(league["league_id"])
    users = get_league_users(league["league_id"])

    user_map = {u["user_id"]: u.get("display_name", "") for u in users}
    roster_to_user = {}
    for r in rosters:
        owner_id = r.get("owner_id")
        roster_id = r["roster_id"]
        roster_to_user[roster_id] = user_map.get(owner_id, f"Roster {roster_id}")
    return roster_to_user


# -----------------------------
# Playoff parsing
# -----------------------------

def detect_playoff_weeks_for_season(league):
    """
    Automatic playoff-week detection with safe fallback:
    - Try to infer from settings
    - Fallback to:
        2020: weeks 14–16
        >=2021: weeks 15–17
    """
    season = int(league["season"])
    settings = league.get("settings", {}) or {}
    playoff_week_start = settings.get("playoff_week_start")
    playoff_teams = settings.get("playoff_teams")

    if playoff_week_start and playoff_teams:
        start = int(playoff_week_start)
        teams = int(playoff_teams)
        weeks = 3 if teams == 6 else 2
        end = start + weeks - 1
        return list(range(start, end + 1))

    if season == 2020:
        return [14, 15, 16]
    else:
        return [15, 16, 17]


def get_playoff_teams_for_season(league):
    """Seeds 1–6 are playoff teams."""
    rosters = get_league_rosters(league["league_id"])
    playoff_teams = set()
    for r in rosters:
        seed = r.get("settings", {}).get("seed")
        if seed and 1 <= seed <= 6:
            playoff_teams.add(r["roster_id"])
    return playoff_teams


def parse_playoff_games_for_season(league):
    """
    Return DataFrame of winners-bracket playoff games only,
    plus metadata to identify rounds.
    """
    season = int(league["season"])
    playoff_weeks = detect_playoff_weeks_for_season(league)
    playoff_teams = get_playoff_teams_for_season(league)

    games = []

    for week in playoff_weeks:
        matchups = get_league_matchups(league["league_id"], week)
        by_matchup = {}
        for m in matchups:
            mid = m.get("matchup_id")
            if mid is None:
                continue
            by_matchup.setdefault(mid, []).append(m)

        for mid, pair in by_matchup.items():
            if len(pair) != 2:
                continue
            m1, m2 = pair
            r1, r2 = m1["roster_id"], m2["roster_id"]
            if r1 not in playoff_teams or r2 not in playoff_teams:
                continue

            pts1 = m1.get("points", 0)
            pts2 = m2.get("points", 0)
            if pts1 > pts2:
                winner, loser = r1, r2
                w_pts, l_pts = pts1, pts2
            else:
                winner, loser = r2, r1
                w_pts, l_pts = pts2, pts1

            games.append({
                "season": season,
                "week": week,
                "matchup_id": mid,
                "roster1": r1,
                "roster2": r2,
                "winner": winner,
                "loser": loser,
                "winner_points": w_pts,
                "loser_points": l_pts
            })

    df = pd.DataFrame(games)
    if df.empty:
        return df

    playoff_weeks_sorted = sorted(playoff_weeks)
    first_week = playoff_weeks_sorted[0]
    last_week = playoff_weeks_sorted[-1]

    def classify_round(row):
        if row["week"] == first_week:
            return "Quarterfinal"
        elif row["week"] == last_week:
            return "Final"
        else:
            return "Semifinal"

    df["round"] = df.apply(classify_round, axis=1)
    return df


def compute_playoff_records(leagues):
    records = {}

    for league in leagues:
        roster_to_user = build_roster_user_maps(league)
        df = parse_playoff_games_for_season(league)
        if df.empty:
            continue

        for _, row in df.iterrows():
            winner = row["winner"]
            loser = row["loser"]

            for rid, result in [(winner, "W"), (loser, "L")]:
                name = roster_to_user.get(rid, f"Roster {rid}")
                if name not in records:
                    records[name] = {"wins": 0, "losses": 0}
                if result == "W":
                    records[name]["wins"] += 1
                else:
                    records[name]["losses"] += 1

    rows = []
    for name, rec in records.items():
        w, l = rec["wins"], rec["losses"]
        total = w + l
        pct = w / total if total > 0 else 0
        rows.append({
            "Manager": name,
            "Playoff Wins": w,
            "Playoff Losses": l,
            "Playoff Win %": round(pct, 3)
        })

    return pd.DataFrame(rows).sort_values(["Playoff Win %", "Playoff Wins"], ascending=[False, False])


def compute_champions(leagues):
    rows = []
    for league in leagues:
        season = int(league["season"])
        roster_to_user = build_roster_user_maps(league)
        df = parse_playoff_games_for_season(league)
        if df.empty:
            continue
        last_week = df["week"].max()
        finals = df[df["week"] == last_week]
        if finals.empty:
            continue
        finals = finals.copy()
        finals["total_points"] = finals["winner_points"] + finals["loser_points"]
        champ_game = finals.sort_values("total_points", ascending=False).iloc[0]
        champ_roster = champ_game["winner"]
        runner_roster = champ_game["loser"]
        rows.append({
            "Season": season,
            "Champion": roster_to_user.get(champ_roster, f"Roster {champ_roster}"),
            "Runner-Up": roster_to_user.get(runner_roster, f"Roster {runner_roster}")
        })

    return pd.DataFrame(rows).sort_values("Season")


# -----------------------------
# Draft pick / trade mapping
# -----------------------------

def build_draft_order_from_league(league):
    settings = league.get("settings", {}) or {}
    draft_order = settings.get("draft_order") or {}

    if draft_order:
        return {str(k): v for k, v in draft_order.items()}

    rosters = get_league_rosters(league["league_id"])
    tmp = []
    for r in rosters:
        seed = r.get("settings", {}).get("seed")
        if seed is not None:
            tmp.append((r["roster_id"], seed))
    if not tmp:
        return {}
    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    return {str(rid): i + 1 for i, (rid, _) in enumerate(tmp)}


def map_traded_picks_to_players(draft_picks, traded_picks, draft_order):
    drafted_lookup = {}
    for p in draft_picks:
        season = int(p.get("season", 0) or 0)
        round_ = p.get("round")
        draft_slot = p.get("draft_slot")
        if season and round_ and draft_slot:
            drafted_lookup[(season, round_, draft_slot)] = p

    results = []

    for tp in traded_picks:
        season = int(tp["season"])
        round_ = tp["round"]
        roster_id = tp["roster_id"]

        draft_slot = draft_order.get(str(roster_id)) or draft_order.get(roster_id)
        if draft_slot is None:
            results.append({
                "season": season,
                "round": round_,
                "roster_id": roster_id,
                "draft_slot": None,
                "player": None,
                "error": "No draft_slot found for roster_id"
            })
            continue

        player = drafted_lookup.get((season, round_, draft_slot))

        results.append({
            "season": season,
            "round": round_,
            "roster_id": roster_id,
            "draft_slot": draft_slot,
            "player": player
        })

    return results


def traded_pick_table_for_season(league):
    drafts = get_league_drafts(league["league_id"])
    if not drafts:
        return pd.DataFrame()

    draft = drafts[0]
    draft_id = draft["draft_id"]
    draft_picks = get_draft_picks(draft_id)
    traded_picks = get_draft_traded_picks(draft_id)
    draft_order = build_draft_order_from_league(league)
    roster_to_user = build_roster_user_maps(league)

    mapped = map_traded_picks_to_players(draft_picks, traded_picks, draft_order)

    rows = []
    for m in mapped:
        p = m["player"]
        if p is None:
            player_name = "Unknown"
            pick_overall = None
        else:
            meta = p.get("metadata", {}) or {}
            player_name = (meta.get("first_name", "") + " " + meta.get("last_name", "")).strip()
            pick_overall = p.get("pick_no")

        rows.append({
            "Season": m["season"],
            "Round": m["round"],
            "Draft Slot": m["draft_slot"],
            "Original Roster": roster_to_user.get(m["roster_id"], f"Roster {m['roster_id']}"),
            "Player": player_name,
            "Overall Pick": pick_overall
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Pick Label"] = df.apply(
        lambda r: f"{r['Round']}.{str(r['Draft Slot']).zfill(2)}" if pd.notnull(r["Draft Slot"]) else "",
        axis=1
    )
    return df.sort_values(["Season", "Round", "Draft Slot"])


# -----------------------------
# Trade trees (table-based)
# -----------------------------

def build_trade_events_for_league(league):
    season = int(league["season"])
    events = []
    for week in range(1, 19):
        txs = get_league_transactions(league["league_id"], week)
        for tx in txs:
            if tx.get("type") != "trade":
                continue
            tid = tx["transaction_id"]
            adds = tx.get("adds", {}) or {}
            drops = tx.get("drops", {}) or {}
            draft_picks = tx.get("draft_picks", []) or []

            for player_id, roster_id in adds.items():
                events.append({
                    "season": season,
                    "week": week,
                    "transaction_id": tid,
                    "asset_type": "player",
                    "asset_id": player_id,
                    "to_roster": roster_id,
                    "direction": "in"
                })
            for player_id, roster_id in drops.items():
                events.append({
                    "season": season,
                    "week": week,
                    "transaction_id": tid,
                    "asset_type": "player",
                    "asset_id": player_id,
                    "from_roster": roster_id,
                    "direction": "out"
                })

            for dp in draft_picks:
                events.append({
                    "season": season,
                    "week": week,
                    "transaction_id": tid,
                    "asset_type": "pick",
                    "round": dp.get("round"),
                    "pick_season": int(dp.get("season")),
                    "to_roster": dp.get("owner_id"),
                    "from_roster": dp.get("previous_owner_id"),
                    "direction": "pick_trade"
                })

    return pd.DataFrame(events)


def build_pick_lineage(leagues, target_season, target_round):
    rows = []
    rows.append({
        "Step": 0,
        "Description": f"Original pick: {target_season} Round {target_round}",
        "From": "",
        "To": "",
        "Season": target_season
    })

    step = 1
    for league in leagues:
        season = int(league["season"])
        events = build_trade_events_for_league(league)
        if events.empty:
            continue
        mask = (events["asset_type"] == "pick") & \
               (events["pick_season"] == target_season) & \
               (events["round"] == target_round)
        dfp = events[mask]
        roster_to_user = build_roster_user_maps(league)
        for _, row in dfp.iterrows():
            rows.append({
                "Step": step,
                "Description": f"Pick traded in {season}, Week {row['week']}",
                "From": roster_to_user.get(row.get("from_roster"), ""),
                "To": roster_to_user.get(row.get("to_roster"), ""),
                "Season": season
            })
            step += 1

    return pd.DataFrame(rows)


def build_player_lineage(leagues, player_id):
    rows = []
    step = 0
    for league in leagues:
        season = int(league["season"])
        events = build_trade_events_for_league(league)
        if events.empty:
            continue
        mask = (events["asset_type"] == "player") & (events["asset_id"] == player_id)
        dfp = events[mask]
        roster_to_user = build_roster_user_maps(league)
        for _, row in dfp.iterrows():
            step += 1
            direction = row["direction"]
            if direction == "in":
                desc = f"Acquired in {season}, Week {row['week']}"
                frm = ""
                to = roster_to_user.get(row.get("to_roster"), "")
            else:
                desc = f"Sent away in {season}, Week {row['week']}"
                frm = roster_to_user.get(row.get("from_roster"), "")
                to = ""
            rows.append({
                "Step": step,
                "Description": desc,
                "From": frm,
                "To": to,
                "Season": season
            })

    return pd.DataFrame(rows)


# -----------------------------
# Data bootstrap
# -----------------------------

@st.cache_data(show_spinner=False)
def load_all_leagues():
    return walk_league_chain(LEAGUE_ID_CURRENT)


leagues = load_all_leagues()
season_options = [int(l["season"]) for l in leagues]
season_options.sort()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Dynasty League Dashboard")

selected_tab = st.sidebar.radio(
    "Select View",
    [
        "Head-to-Head",
        "Top Rivalries",
        "Playoff History",
        "Team Transaction Profiles",
        "Manager Tendencies",
        "1st-Round Pick Trade Explorer",
        "Trade Trees"
    ]
)

selected_season = st.sidebar.selectbox(
    "Season",
    options=season_options,
    index=len(season_options) - 1
)

current_league = next(l for l in leagues if int(l["season"]) == selected_season)


# -----------------------------
# Tab: Head-to-Head (simple but correct)
# -----------------------------

if selected_tab == "Head-to-Head":
    st.header("Head-to-Head")

    roster_to_user = build_roster_user_maps(current_league)
    rows = []
    for week in range(1, 19):
        matchups = get_league_matchups(current_league["league_id"], week)
        by_matchup = {}
        for m in matchups:
            mid = m.get("matchup_id")
            if mid is None:
                continue
            by_matchup.setdefault(mid, []).append(m)
        for mid, pair in by_matchup.items():
            if len(pair) != 2:
                continue
            m1, m2 = pair
            r1, r2 = m1["roster_id"], m2["roster_id"]
            pts1, pts2 = m1.get("points", 0), m2.get("points", 0)
            if pts1 > pts2:
                winner, loser = r1, r2
            else:
                winner, loser = r2, r1
            rows.append({
                "Winner": roster_to_user.get(winner, f"Roster {winner}"),
                "Loser": roster_to_user.get(loser, f"Roster {loser}")
            })

    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby(["Winner", "Loser"]).size().reset_index(name="Wins")
        st.dataframe(summary.sort_values("Wins", ascending=False), use_container_width=True)
    else:
        st.info("No matchup data for this season.")


# -----------------------------
# Tab: Top Rivalries
# -----------------------------

elif selected_tab == "Top Rivalries":
    st.header("Top Rivalries")

    roster_to_user = build_roster_user_maps(current_league)
    rows = []
    for week in range(1, 19):
        matchups = get_league_matchups(current_league["league_id"], week)
        by_matchup = {}
        for m in matchups:
            mid = m.get("matchup_id")
            if mid is None:
                continue
            by_matchup.setdefault(mid, []).append(m)
        for mid, pair in by_matchup.items():
            if len(pair) != 2:
                continue
            m1, m2 = pair
            r1, r2 = m1["roster_id"], m2["roster_id"]
            rows.append({"r1": r1, "r2": r2})

    if rows:
        df = pd.DataFrame(rows)
        df["pair"] = df.apply(lambda r: tuple(sorted([r["r1"], r["r2"]])), axis=1)
        summary = df.groupby("pair").size().reset_index(name="Games")
        summary["Manager A"] = summary["pair"].apply(lambda p: roster_to_user.get(p[0], f"Roster {p[0]}"))
        summary["Manager B"] = summary["pair"].apply(lambda p: roster_to_user.get(p[1], f"Roster {p[1]}"))
        st.dataframe(
            summary[["Manager A", "Manager B", "Games"]].sort_values("Games", ascending=False),
            use_container_width=True
        )
    else:
        st.info("No rivalry data for this season.")


# -----------------------------
# Tab: Playoff History
# -----------------------------

elif selected_tab == "Playoff History":
    st.header("Playoff History")

    champs_df = compute_champions(leagues)
    st.subheader("Champions by Season")
    if champs_df.empty:
        st.info("No playoff data available.")
    else:
        st.dataframe(champs_df, use_container_width=True)

    st.subheader("All-Time Playoff Records")
    records_df = compute_playoff_records(leagues)
    if records_df.empty:
        st.info("No playoff records available.")
    else:
        st.dataframe(records_df, use_container_width=True)

    st.subheader(f"Playoff Bracket Games – {selected_season}")
    bracket_df = parse_playoff_games_for_season(current_league)
    if bracket_df.empty:
        st.info("No playoff games detected for this season.")
    else:
        roster_to_user = build_roster_user_maps(current_league)
        bracket_df["Winner Name"] = bracket_df["winner"].map(lambda r: roster_to_user.get(r, f"Roster {r}"))
        bracket_df["Loser Name"] = bracket_df["loser"].map(lambda r: roster_to_user.get(r, f"Roster {r}"))
        st.dataframe(
            bracket_df[["season", "week", "round", "Winner Name", "Loser Name", "winner_points", "loser_points"]],
            use_container_width=True
        )


# -----------------------------
# Tab: Team Transaction Profiles (simple placeholder)
# -----------------------------

elif selected_tab == "Team Transaction Profiles":
    st.header("Team Transaction Profiles")
    st.write("This tab can be expanded with your original transaction profile logic.")
    st.info("Currently not implemented in detail in this version.")


# -----------------------------
# Tab: Manager Tendencies (simple placeholder)
# -----------------------------

elif selected_tab == "Manager Tendencies":
    st.header("Manager Tendencies")
    st.write("This tab can be expanded with your original tendencies logic.")
    st.info("Currently not implemented in detail in this version.")


# -----------------------------
# Tab: 1st-Round Pick Trade Explorer
# -----------------------------

elif selected_tab == "1st-Round Pick Trade Explorer":
    st.header("1st-Round Pick Trade Explorer")

    df = traded_pick_table_for_season(current_league)
    if df.empty:
        st.info("No traded pick data for this season.")
    else:
        st.dataframe(
            df[["Season", "Pick Label", "Original Roster", "Player", "Overall Pick"]],
            use_container_width=True
        )


# -----------------------------
# Tab: Trade Trees
# -----------------------------

elif selected_tab == "Trade Trees":
    st.header("Trade Trees")

    mode = st.radio("Lineage Mode", ["Pick Lineage", "Player Lineage"], horizontal=True)

    if mode == "Pick Lineage":
        st.subheader("Pick Lineage")

        col1, col2 = st.columns(2)
        with col1:
            pick_season = st.selectbox("Pick Season", options=season_options, index=len(season_options) - 1)
        with col2:
            pick_round = st.number_input("Round", min_value=1, max_value=10, value=1, step=1)

        if st.button("Build Pick Lineage"):
            lineage_df = build_pick_lineage(leagues, pick_season, pick_round)
            if lineage_df.empty:
                st.info("No lineage events found for this pick.")
            else:
                st.dataframe(lineage_df.sort_values(["Season", "Step"]), use_container_width=True)

    else:
        st.subheader("Player Lineage")

        player_id = st.text_input("Sleeper Player ID (e.g., 6790)")
        if st.button("Build Player Lineage"):
            if not player_id:
                st.warning("Enter a player ID.")
            else:
                lineage_df = build_player_lineage(leagues, player_id)
                if lineage_df.empty:
                    st.info("No lineage events found for this player.")
                else:
                    st.dataframe(lineage_df.sort_values(["Season", "Step"]), use_container_width=True)
