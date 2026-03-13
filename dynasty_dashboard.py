import streamlit as st
import requests
import pandas as pd
import numpy as np
from functools import lru_cache

st.set_page_config(page_title="Dynasty Dashboard", layout="wide")

# ---------- CONFIG ----------

LEAGUE_IDS_BY_SEASON = {
    "2025": "1180359904212156416",
    "2024": "1048279751829368832",
    "2023": "917164805700575232",
    "2022": "784501800425377792",
    "2021": "650036126123405312",
    "2020": "543561523148333056",
}

SEASONS_ORDERED = ["2025", "2024", "2023", "2022", "2021", "2020"]


# ---------- API HELPERS ----------

BASE_URL = "https://api.sleeper.app/v1"


def _get(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=None)
def get_league(league_id: str):
    return _get(f"{BASE_URL}/league/{league_id}")


@lru_cache(maxsize=None)
def get_rosters(league_id: str):
    return _get(f"{BASE_URL}/league/{league_id}/rosters")


@lru_cache(maxsize=None)
def get_users(league_id: str):
    return _get(f"{BASE_URL}/league/{league_id}/users")


@lru_cache(maxsize=None)
def get_matchups(league_id: str, week: int):
    return _get(f"{BASE_URL}/league/{league_id}/matchups/{week}")


@lru_cache(maxsize=None)
def get_transactions(league_id: str, week: int):
    return _get(f"{BASE_URL}/league/{league_id}/transactions/{week}")


@lru_cache(maxsize=None)
def get_drafts(league_id: str):
    return _get(f"{BASE_URL}/league/{league_id}/drafts")


@lru_cache(maxsize=None)
def get_draft_picks(draft_id: str):
    return _get(f"{BASE_URL}/draft/{draft_id}/picks")


# ---------- UTILS ----------

def build_user_and_roster_maps(league_id: str):
    users = get_users(league_id)
    rosters = get_rosters(league_id)

    user_map = {u["user_id"]: u.get("display_name", u.get("username", "Unknown")) for u in users}
    roster_to_owner = {r["roster_id"]: r["owner_id"] for r in rosters}
    owner_to_roster = {v: k for k, v in roster_to_owner.items()}

    roster_name_map = {}
    for r in rosters:
        owner_id = r["owner_id"]
        name = user_map.get(owner_id, f"Roster {r['roster_id']}")
        roster_name_map[r["roster_id"]] = name

    return users, rosters, user_map, roster_to_owner, owner_to_roster, roster_name_map


def get_all_league_ids():
    return [LEAGUE_IDS_BY_SEASON[s] for s in SEASONS_ORDERED]


def get_all_league_data():
    leagues = {}
    for season, league_id in LEAGUE_IDS_BY_SEASON.items():
        leagues[season] = {
            "league": get_league(league_id),
            "rosters": get_rosters(league_id),
            "users": get_users(league_id),
        }
    return leagues


def get_all_matchups_for_league(league_id: str):
    league = get_league(league_id)
    total_weeks = league.get("settings", {}).get("playoff_week_start", 15)
    matchups_by_week = {}
    for w in range(1, total_weeks + 1):
        try:
            matchups_by_week[w] = get_matchups(league_id, w)
        except Exception:
            matchups_by_week[w] = []
    return matchups_by_week


def get_all_transactions_for_league(league_id: str):
    league = get_league(league_id)
    total_weeks = league.get("settings", {}).get("playoff_week_start", 15)
    tx_by_week = {}
    for w in range(1, total_weeks + 1):
        try:
            tx_by_week[w] = get_transactions(league_id, w)
        except Exception:
            tx_by_week[w] = []
    return tx_by_week


def get_all_draft_picks_for_league(league_id: str):
    drafts = get_drafts(league_id)
    all_picks = []
    for d in drafts:
        draft_id = d["draft_id"]
        picks = get_draft_picks(draft_id)
        for p in picks:
            p["league_id"] = league_id
            all_picks.append(p)
    return all_picks


# ---------- 1ST ROUND PICK TRADE EXPLORER ----------

def compute_first_round_trade_results(transactions, draft_picks, rosters, users):
    if not transactions:
        return pd.DataFrame(), pd.DataFrame()

    season_to_league = LEAGUE_IDS_BY_SEASON.copy()

    user_map = {u["user_id"]: u.get("display_name", u.get("username", "Unknown")) for u in users}
    roster_owner = {r["roster_id"]: r["owner_id"] for r in rosters}

    df_tx = pd.DataFrame(transactions)
    df_dp = pd.DataFrame(draft_picks) if draft_picks else pd.DataFrame()

    if not df_dp.empty:
        if "league_id" in df_dp.columns:
            df_dp["league_id"] = df_dp["league_id"].astype(str)
        if "round" in df_dp.columns:
            df_dp["round"] = pd.to_numeric(df_dp["round"], errors="coerce").fillna(0).astype(int)

    results = []

    if df_tx.empty:
        return pd.DataFrame(), pd.DataFrame()

    trade_rows = df_tx[df_tx["type"] == "trade"]
    for _, trade in trade_rows.iterrows():
        picks = trade.get("draft_picks") or []
        for pick in picks:
            if pick.get("round") != 1:
                continue

            season = str(pick.get("season"))
            pick_roster_id = pick.get("roster_id")
            prev_rid = pick.get("previous_owner_id")
            curr_rid = pick.get("owner_id")

            pick_league_id = pick.get("league_id")
            if not pick_league_id and season in season_to_league:
                pick_league_id = season_to_league[season]

            original_owner = user_map.get(roster_owner.get(prev_rid), f"Roster {prev_rid}")
            current_owner = user_map.get(roster_owner.get(curr_rid), f"Roster {curr_rid}")

            player_name = None
            pick_no = None

            if pick_league_id and not df_dp.empty:
                mask = (
                    (df_dp["league_id"].astype(str) == str(pick_league_id)) &
                    (df_dp["round"] == 1) &
                    (df_dp["roster_id"] == pick_roster_id)
                )
                dp = df_dp[mask]
                if not dp.empty:
                    dp_row = dp.iloc[0]
                    meta = dp_row.get("metadata") or {}
                    player_name = meta.get("player_name") or meta.get("full_name")
                    pick_no = dp_row.get("pick_no")

            results.append({
                "season": season,
                "original_owner": original_owner,
                "current_owner": current_owner,
                "player_selected": player_name,
                "pick_no": pick_no,
                "transaction_id": trade.get("transaction_id"),
                "trade_draft_picks": trade.get("draft_picks"),
                "pick_league_id": pick_league_id,
                "pick_roster_id": pick_roster_id,
            })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res, df_res

    df_res = df_res.drop_duplicates(
        subset=["transaction_id", "season", "pick_league_id", "pick_roster_id"]
    )

    past = df_res[df_res["player_selected"].notna()].reset_index(drop=True)
    future = df_res[df_res["player_selected"].isna()].reset_index(drop=True)
    return past, future


def build_first_round_trade_view():
    st.header("1st-Round Pick Trade Explorer")

    all_tx = []
    all_dp = []
    all_rosters = []
    all_users = []

    for season, league_id in LEAGUE_IDS_BY_SEASON.items():
        tx_by_week = get_all_transactions_for_league(league_id)
        for w, txs in tx_by_week.items():
            for t in txs:
                t["season"] = season
                t["league_id"] = league_id
                all_tx.append(t)

        picks = get_all_draft_picks_for_league(league_id)
        for p in picks:
            p["season"] = season
            all_dp.append(p)

        rosters = get_rosters(league_id)
        users = get_users(league_id)
        for r in rosters:
            r["season"] = season
            r["league_id"] = league_id
            all_rosters.append(r)
        for u in users:
            u["season"] = season
            u["league_id"] = league_id
            all_users.append(u)

    past, future = compute_first_round_trade_results(all_tx, all_dp, all_rosters, all_users)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Past 1st-Round Picks (Already Drafted)")
        if past.empty:
            st.info("No past 1st-round pick trades have been mapped to drafted players yet.")
        else:
            display_cols = [
                "season",
                "original_owner",
                "current_owner",
                "player_selected",
                "pick_no",
                "transaction_id",
            ]
            st.dataframe(past[display_cols].sort_values(["season", "pick_no"]))

    with col2:
        st.subheader("Future 1st-Round Picks (Not Yet Drafted)")
        if future.empty:
            st.info("No future 1st-round pick trades found.")
        else:
            display_cols = [
                "season",
                "original_owner",
                "current_owner",
                "transaction_id",
                "trade_draft_picks",
            ]
            st.dataframe(future[display_cols].sort_values(["season", "transaction_id"]))


# ---------- PLAYOFF HISTORY ----------

def classify_playoff_round(week, playoff_start_week):
    if week < playoff_start_week:
        return None
    # Simple mapping: first playoff week = quarter/semis depending on league size
    offset = week - playoff_start_week
    if offset == 0:
        return "Quarterfinal / Semifinal"
    elif offset == 1:
        return "Semifinal / Final"
    else:
        return "Championship / Consolation"


def build_playoff_history_view():
    st.header("Playoff History")

    rows = []

    for season, league_id in LEAGUE_IDS_BY_SEASON.items():
        league = get_league(league_id)
        users, rosters, user_map, roster_owner, owner_to_roster, roster_name_map = build_user_and_roster_maps(league_id)

        playoff_start = league.get("settings", {}).get("playoff_week_start", 15)
        total_weeks = league.get("settings", {}).get("playoff_week_start", 15) + 3

        for week in range(1, total_weeks + 1):
            matchups = get_matchups(league_id, week)
            if not matchups:
                continue

            # Try to detect playoff via matchup_type; fallback to week >= playoff_start
            is_playoff_week = any(m.get("matchup_type") == "playoff" for m in matchups)
            if not is_playoff_week and week < playoff_start:
                continue

            round_label = classify_playoff_round(week, playoff_start)

            # Group by matchup_id
            df_m = pd.DataFrame(matchups)
            if "matchup_id" not in df_m.columns:
                continue

            for mid, g in df_m.groupby("matchup_id"):
                if len(g) < 2:
                    continue
                g = g.sort_values("roster_id")
                team_a = g.iloc[0]
                team_b = g.iloc[1]

                for team, opp in [(team_a, team_b), (team_b, team_a)]:
                    rid = team["roster_id"]
                    opp_rid = opp["roster_id"]
                    pts = team.get("points", 0)
                    opp_pts = opp.get("points", 0)

                    owner_id = roster_owner.get(rid)
                    opp_owner_id = roster_owner.get(opp_rid)

                    rows.append({
                        "season": season,
                        "league_id": league_id,
                        "week": week,
                        "round": round_label,
                        "roster_id": rid,
                        "team_name": roster_name_map.get(rid, f"Roster {rid}"),
                        "owner_id": owner_id,
                        "opp_roster_id": opp_rid,
                        "opp_team_name": roster_name_map.get(opp_rid, f"Roster {opp_rid}"),
                        "opp_owner_id": opp_owner_id,
                        "points": pts,
                        "opp_points": opp_pts,
                        "win": pts > opp_pts,
                    })

    if not rows:
        st.info("No playoff data found across configured seasons.")
        return

    df = pd.DataFrame(rows)

    # Aggregate playoff performance
    agg = df.groupby(["team_name", "owner_id"]).agg(
        playoff_games=("win", "count"),
        playoff_wins=("win", "sum"),
        playoff_losses=("win", lambda x: (~x).sum()),
    ).reset_index()

    agg["playoff_win_pct"] = np.where(
        agg["playoff_games"] > 0,
        agg["playoff_wins"] / agg["playoff_games"],
        np.nan,
    )

    # Titles: championship wins = last playoff round, win == True
    # Approx: highest week per season where team played & won
    titles_rows = []
    for (season, team_name), g in df.groupby(["season", "team_name"]):
        max_week = g["week"].max()
        g_last = g[g["week"] == max_week]
        # If they won in last week they played, treat as title
        if any(g_last["win"]):
            titles_rows.append({"season": season, "team_name": team_name})

    df_titles = pd.DataFrame(titles_rows)
    title_counts = df_titles.groupby("team_name").size().reset_index(name="titles")

    agg = agg.merge(title_counts, on="team_name", how="left")
    agg["titles"] = agg["titles"].fillna(0).astype(int)

    st.subheader("Playoff Summary by Team")
    display_cols = [
        "team_name",
        "playoff_games",
        "playoff_wins",
        "playoff_losses",
        "playoff_win_pct",
        "titles",
    ]
    st.dataframe(
        agg[display_cols].sort_values(
            ["titles", "playoff_win_pct", "playoff_games"], ascending=[False, False, False]
        )
    )

    st.subheader("Raw Playoff Matchups")
    st.dataframe(
        df.sort_values(["season", "week", "team_name"])[
            ["season", "week", "round", "team_name", "points", "opp_team_name", "opp_points", "win"]
        ]
    )


# ---------- MANAGER TENDENCIES ----------

def build_manager_tendencies_view():
    st.header("Manager Tendencies & Behavior")

    rows = []

    for season, league_id in LEAGUE_IDS_BY_SEASON.items():
        league = get_league(league_id)
        users, rosters, user_map, roster_owner, owner_to_roster, roster_name_map = build_user_and_roster_maps(league_id)

        playoff_start = league.get("settings", {}).get("playoff_week_start", 15)
        total_weeks = playoff_start + 3

        for week in range(1, total_weeks + 1):
            matchups = get_matchups(league_id, week)
            if not matchups:
                continue

            df_m = pd.DataFrame(matchups)
            if "matchup_id" not in df_m.columns:
                continue

            for mid, g in df_m.groupby("matchup_id"):
                if len(g) < 2:
                    continue
                g = g.sort_values("roster_id")
                team_a = g.iloc[0]
                team_b = g.iloc[1]

                for team, opp in [(team_a, team_b), (team_b, team_a)]:
                    rid = team["roster_id"]
                    opp_rid = opp["roster_id"]
                    pts = team.get("points", 0)
                    opp_pts = opp.get("points", 0)
                    proj = team.get("projected_points", 0)
                    opp_proj = opp.get("projected_points", 0)

                    owner_id = roster_owner.get(rid)
                    opp_owner_id = roster_owner.get(opp_rid)

                    margin = pts - opp_pts
                    proj_margin = proj - opp_proj

                    win = pts > opp_pts
                    upset_win = win and proj < opp_proj
                    upset_loss = (not win) and proj > opp_proj

                    blowout_win = win and margin >= 30
                    close_win = win and 0 < margin < 10
                    blowout_loss = (not win) and margin <= -30
                    heartbreaker_loss = (not win) and -10 < margin < 0

                    rows.append({
                        "season": season,
                        "league_id": league_id,
                        "week": week,
                        "roster_id": rid,
                        "team_name": roster_name_map.get(rid, f"Roster {rid}"),
                        "owner_id": owner_id,
                        "points": pts,
                        "opp_points": opp_pts,
                        "projected_points": proj,
                        "opp_projected_points": opp_proj,
                        "margin": margin,
                        "proj_margin": proj_margin,
                        "win": win,
                        "upset_win": upset_win,
                        "upset_loss": upset_loss,
                        "blowout_win": blowout_win,
                        "close_win": close_win,
                        "blowout_loss": blowout_loss,
                        "heartbreaker_loss": heartbreaker_loss,
                    })

    if not rows:
        st.info("No matchup data found across configured seasons.")
        return

    df = pd.DataFrame(rows)

    agg = df.groupby(["team_name", "owner_id"]).agg(
        games=("win", "count"),
        wins=("win", "sum"),
        losses=("win", lambda x: (~x).sum()),
        avg_margin=("margin", "mean"),
        avg_proj_margin=("proj_margin", "mean"),
        blowout_wins=("blowout_win", "sum"),
        close_wins=("close_win", "sum"),
        blowout_losses=("blowout_loss", "sum"),
        heartbreaker_losses=("heartbreaker_loss", "sum"),
        upset_wins=("upset_win", "sum"),
        upset_losses=("upset_loss", "sum"),
    ).reset_index()

    agg["win_pct"] = np.where(
        agg["games"] > 0,
        agg["wins"] / agg["games"],
        np.nan,
    )

    st.subheader("Manager Profiles")
    display_cols = [
        "team_name",
        "games",
        "wins",
        "losses",
        "win_pct",
        "avg_margin",
        "avg_proj_margin",
        "blowout_wins",
        "close_wins",
        "blowout_losses",
        "heartbreaker_losses",
        "upset_wins",
        "upset_losses",
    ]
    st.dataframe(
        agg[display_cols].sort_values(
            ["win_pct", "games"], ascending=[False, False]
        )
    )

    st.subheader("Raw Matchup Tendencies")
    st.dataframe(
        df.sort_values(["season", "week", "team_name"])[
            [
                "season",
                "week",
                "team_name",
                "points",
                "opp_points",
                "projected_points",
                "opp_projected_points",
                "margin",
                "proj_margin",
                "win",
                "upset_win",
                "upset_loss",
                "blowout_win",
                "close_win",
                "blowout_loss",
                "heartbreaker_loss",
            ]
        ]
    )


# ---------- SIMPLE PLACEHOLDER SECTIONS (OVERVIEW / H2H / RIVALRIES / TRANSACTIONS) ----------

def build_league_overview_view():
    st.header("League Overview")

    leagues = get_all_league_data()
    rows = []
    for season, data in leagues.items():
        league = data["league"]
        rosters = data["rosters"]
        users = data["users"]
        rows.append({
            "season": season,
            "league_name": league.get("name", ""),
            "num_teams": len(rosters),
            "num_users": len(users),
        })

    df = pd.DataFrame(rows)
    st.subheader("Basic League Snapshot")
    st.dataframe(df.sort_values("season", ascending=False))


def build_head_to_head_view():
    st.header("Head-to-Head (Simple Summary)")

    rows = []

    for season, league_id in LEAGUE_IDS_BY_SEASON.items():
        users, rosters, user_map, roster_owner, owner_to_roster, roster_name_map = build_user_and_roster_maps(league_id)
        matchups_by_week = get_all_matchups_for_league(league_id)

        for week, matchups in matchups_by_week.items():
            if not matchups:
                continue
            df_m = pd.DataFrame(matchups)
            if "matchup_id" not in df_m.columns:
                continue

            for mid, g in df_m.groupby("matchup_id"):
                if len(g) < 2:
                    continue
                g = g.sort_values("roster_id")
                team_a = g.iloc[0]
                team_b = g.iloc[1]

                for team, opp in [(team_a, team_b), (team_b, team_a)]:
                    rid = team["roster_id"]
                    opp_rid = opp["roster_id"]
                    pts = team.get("points", 0)
                    opp_pts = opp.get("points", 0)
                    owner_id = roster_owner.get(rid)
                    opp_owner_id = roster_owner.get(opp_rid)

                    rows.append({
                        "season": season,
                        "week": week,
                        "team_name": roster_name_map.get(rid, f"Roster {rid}"),
                        "opp_team_name": roster_name_map.get(opp_rid, f"Roster {opp_rid}"),
                        "owner_id": owner_id,
                        "opp_owner_id": opp_owner_id,
                        "points": pts,
                        "opp_points": opp_pts,
                        "win": pts > opp_pts,
                    })

    if not rows:
        st.info("No head-to-head data found.")
        return

    df = pd.DataFrame(rows)

    agg = df.groupby(["team_name"]).agg(
        games=("win", "count"),
        wins=("win", "sum"),
        losses=("win", lambda x: (~x).sum()),
        points_for=("points", "sum"),
        points_against=("opp_points", "sum"),
    ).reset_index()

    agg["win_pct"] = np.where(
        agg["games"] > 0,
        agg["wins"] / agg["games"],
        np.nan,
    )

    st.subheader("Head-to-Head Summary")
    st.dataframe(
        agg.sort_values(["win_pct", "games"], ascending=[False, False])
    )


def build_rivalries_view():
    st.header("Rivalries (Simple View)")

    rows = []

    for season, league_id in LEAGUE_IDS_BY_SEASON.items():
        users, rosters, user_map, roster_owner, owner_to_roster, roster_name_map = build_user_and_roster_maps(league_id)
        matchups_by_week = get_all_matchups_for_league(league_id)

        for week, matchups in matchups_by_week.items():
            if not matchups:
                continue
            df_m = pd.DataFrame(matchups)
            if "matchup_id" not in df_m.columns:
                continue

            for mid, g in df_m.groupby("matchup_id"):
                if len(g) < 2:
                    continue
                g = g.sort_values("roster_id")
                team_a = g.iloc[0]
                team_b = g.iloc[1]

                rid_a = team_a["roster_id"]
                rid_b = team_b["roster_id"]
                pts_a = team_a.get("points", 0)
                pts_b = team_b.get("points", 0)

                name_a = roster_name_map.get(rid_a, f"Roster {rid_a}")
                name_b = roster_name_map.get(rid_b, f"Roster {rid_b}")

                rows.append({
                    "season": season,
                    "team_a": name_a,
                    "team_b": name_b,
                    "points_a": pts_a,
                    "points_b": pts_b,
                    "winner": name_a if pts_a > pts_b else (name_b if pts_b > pts_a else "Tie"),
                })

    if not rows:
        st.info("No rivalry data found.")
        return

    df = pd.DataFrame(rows)

    # Normalize pair key
    df["pair"] = df.apply(
        lambda r: " vs ".join(sorted([r["team_a"], r["team_b"]])),
        axis=1,
    )

    agg = df.groupby("pair").agg(
        games=("winner", "count"),
        wins_a=("winner", lambda x: sum(x == x.index[0])),
    ).reset_index()

    st.subheader("Rivalry Pairs (by games played)")
    st.dataframe(
        df[["season", "team_a", "team_b", "points_a", "points_b", "winner"]].sort_values(
            ["season", "team_a", "team_b"]
        )
    )


def build_transactions_view():
    st.header("Transactions & Archetypes (Simple View)")

    rows = []

    for season, league_id in LEAGUE_IDS_BY_SEASON.items():
        tx_by_week = get_all_transactions_for_league(league_id)
        for week, txs in tx_by_week.items():
            for t in txs:
                rows.append({
                    "season": season,
                    "league_id": league_id,
                    "week": week,
                    "type": t.get("type"),
                    "status": t.get("status"),
                    "transaction_id": t.get("transaction_id"),
                })

    if not rows:
        st.info("No transaction data found.")
        return

    df = pd.DataFrame(rows)

    st.subheader("Transaction Types by Season")
    st.dataframe(
        df.groupby(["season", "type"]).size().reset_index(name="count").sort_values(
            ["season", "count"], ascending=[False, False]
        )
    )

    st.subheader("Raw Transactions")
    st.dataframe(df.sort_values(["season", "week", "transaction_id"]))


# ---------- MAIN APP ----------

def main():
    st.title("Dynasty Dashboard")

    st.sidebar.title("Navigation")
    section = st.sidebar.selectbox(
        "Go to section",
        [
            "League Overview",
            "Head-to-Head",
            "Rivalries",
            "Transactions & Archetypes",
            "1st-Round Pick Trade Explorer",
            "Playoff History",
            "Manager Tendencies",
        ],
    )

    if section == "League Overview":
        build_league_overview_view()
    elif section == "Head-to-Head":
        build_head_to_head_view()
    elif section == "Rivalries":
        build_rivalries_view()
    elif section == "Transactions & Archetypes":
        build_transactions_view()
    elif section == "1st-Round Pick Trade Explorer":
        build_first_round_trade_view()
    elif section == "Playoff History":
        build_playoff_history_view()
    elif section == "Manager Tendencies":
        build_manager_tendencies_view()


if __name__ == "__main__":
    main()
