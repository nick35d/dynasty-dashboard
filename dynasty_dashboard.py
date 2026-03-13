import requests
import pandas as pd
import numpy as np
from collections import defaultdict

import streamlit as st
import plotly.express as px

BASE = "https://api.sleeper.app/v1"

LEAGUE_IDS = [
    "1180359904212156416",  # 2025
    "1048279751829368832",  # 2024
    "917164805700575232",   # 2023
    "784501800425377792",   # 2022
    "650036126123405312",   # 2021
    "543561523148333056"    # 2020
]

# -----------------------------
# Data loading helpers
# -----------------------------

def api_get(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def get_league_data(league_id):
    league = api_get(f"{BASE}/league/{league_id}")
    users = api_get(f"{BASE}/league/{league_id}/users")
    rosters = api_get(f"{BASE}/league/{league_id}/rosters")

    matchups = []
    for week in range(1, 19):
        try:
            m = api_get(f"{BASE}/league/{league_id}/matchups/{week}")
            for row in m:
                row["week"] = week
                row["league_id"] = league_id
                matchups.append(row)
        except:
            pass

    transactions = []
    for week in range(1, 19):
        try:
            t = api_get(f"{BASE}/league/{league_id}/transactions/{week}")
            for row in t:
                row["week"] = week
                row["league_id"] = league_id
                transactions.append(row)
        except:
            pass

    draft_picks = []
    draft_id = league.get("draft_id")
    if draft_id:
        try:
            dp = api_get(f"{BASE}/draft/{draft_id}/picks")
            for p in dp:
                p["league_id"] = league_id
            draft_picks = dp
        except:
            pass

    return league, users, rosters, matchups, transactions, draft_picks

# -----------------------------
# Analytics functions
# -----------------------------

def compute_head_to_head(matchups, rosters, users):
    owner_map = {r["roster_id"]: r["owner_id"] for r in rosters}
    user_map = {u["user_id"]: u["display_name"] for u in users}

    h2h = defaultdict(lambda: {"wins": 0, "losses": 0})
    df = pd.DataFrame(matchups)
    if df.empty:
        return pd.DataFrame()

    for _, group in df.groupby(["league_id", "week", "matchup_id"]):
        if len(group) != 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        owner_a = owner_map.get(a["roster_id"])
        owner_b = owner_map.get(b["roster_id"])
        if a["points"] > b["points"]:
            h2h[(owner_a, owner_b)]["wins"] += 1
            h2h[(owner_b, owner_a)]["losses"] += 1
        elif b["points"] > a["points"]:
            h2h[(owner_b, owner_a)]["wins"] += 1
            h2h[(owner_a, owner_b)]["losses"] += 1

    rows = []
    for (o1, o2), rec in h2h.items():
        rows.append({
            "owner_1": user_map.get(o1, o1),
            "owner_2": user_map.get(o2, o2),
            "wins": rec["wins"],
            "losses": rec["losses"]
        })
    return pd.DataFrame(rows)

def compute_transaction_stats(transactions, rosters, users):
    user_map = {u["user_id"]: u["display_name"] for u in users}
    roster_owner = {r["roster_id"]: r["owner_id"] for r in rosters}

    stats = defaultdict(lambda: {"adds": 0, "drops": 0, "trades": 0, "fab_spent": 0})

    for t in transactions:
        t_type = t.get("type")
        adds = t.get("adds") or {}
        drops = t.get("drops") or {}

        if t_type in ("waiver", "free_agent"):
            for _, rid in adds.items():
                stats[rid]["adds"] += 1
            for _, rid in drops.items():
                stats[rid]["drops"] += 1
            if t.get("settings") and "waiver_bid" in t["settings"]:
                if adds:
                    rid = list(adds.values())[0]
                    stats[rid]["fab_spent"] += t["settings"]["waiver_bid"]

        if t_type == "trade":
            for rid in t.get("roster_ids", []):
                stats[rid]["trades"] += 1

    rows = []
    for rid, s in stats.items():
        owner_id = roster_owner.get(rid)
        display = user_map.get(owner_id, f"Roster {rid}")
        rows.append({
            "roster_id": rid,
            "user": display,
            **s
        })
    return pd.DataFrame(rows)

def compute_all_play(matchups, rosters, users):
    owner_map = {r["roster_id"]: r["owner_id"] for r in rosters}
    user_map = {u["user_id"]: u["display_name"] for u in users}

    df = pd.DataFrame(matchups)
    if df.empty:
        return pd.DataFrame()

    all_play = defaultdict(lambda: {"wins": 0, "losses": 0})

    for (_, week_group) in df.groupby(["league_id", "week"]):
        scores = week_group[["roster_id", "points"]].copy()
        scores["owner_id"] = scores["roster_id"].map(owner_map)
        for _, row in scores.iterrows():
            wins = (scores["points"] < row["points"]).sum()
            losses = (scores["points"] > row["points"]).sum()
            all_play[row["owner_id"]]["wins"] += wins
            all_play[row["owner_id"]]["losses"] += losses

    rows = []
    for uid, rec in all_play.items():
        total = rec["wins"] + rec["losses"]
        pct = rec["wins"] / total if total > 0 else np.nan
        rows.append({
            "user": user_map.get(uid, uid),
            "all_play_wins": rec["wins"],
            "all_play_losses": rec["losses"],
            "all_play_pct": pct
        })
    return pd.DataFrame(rows)

def compute_rivalry_index(df_h2h):
    if df_h2h.empty:
        return df_h2h
    df = df_h2h.copy()
    df["games"] = df["wins"] + df["losses"]
    df = df[df["games"] > 0]
    df["win_pct"] = df["wins"] / df["games"]
    df["rivalry_score"] = df["games"] * (df["win_pct"] - 0.5).abs()
    return df.sort_values("rivalry_score", ascending=False)

def compute_first_round_trade_results(transactions, draft_picks, rosters, users):
    if not transactions or not draft_picks:
        return pd.DataFrame(), pd.DataFrame()

    user_map = {u["user_id"]: u["display_name"] for u in users}
    roster_owner = {r["roster_id"]: r["owner_id"] for r in rosters}

    df_tx = pd.DataFrame(transactions)
    df_dp = pd.DataFrame(draft_picks)

    # Ensure types
    if "league_id" in df_dp.columns:
        df_dp["league_id"] = df_dp["league_id"].astype(str)
    if "round" in df_dp.columns:
        df_dp["round"] = df_dp["round"].astype(int, errors="ignore")

    results = []

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

            original_owner = user_map.get(roster_owner.get(prev_rid), f"Roster {prev_rid}")
            current_owner = user_map.get(roster_owner.get(curr_rid), f"Roster {curr_rid}")

            player_name = None
            pick_no = None

            if pick_league_id is not None:
                # Past pick: league_id should match a real draft
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
                "trade_adds": trade.get("adds"),
                "trade_drops": trade.get("drops"),
                "trade_draft_picks": trade.get("draft_picks"),
                "pick_league_id": pick_league_id,
                "pick_roster_id": pick_roster_id
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

# -----------------------------
# Caching data
# -----------------------------

@st.cache_data(show_spinner=True)
def load_all_data():
    all_leagues = []
    all_users = []
    all_rosters = []
    all_matchups = []
    all_transactions = []
    all_draft_picks = []

    for lid in LEAGUE_IDS:
        league, users, rosters, matchups, transactions, draft_picks = get_league_data(lid)
        all_leagues.append(league)
        all_users.extend(users)
        all_rosters.extend(rosters)
        all_matchups.extend(matchups)
        all_transactions.extend(transactions)
        all_draft_picks.extend(draft_picks)

    df_users = pd.DataFrame(all_users).drop_duplicates("user_id")
    df_rosters = pd.DataFrame(all_rosters)
    df_matchups = pd.DataFrame(all_matchups)
    df_transactions = pd.DataFrame(all_transactions)
    df_draft = pd.DataFrame(all_draft_picks)

    df_h2h = compute_head_to_head(all_matchups, all_rosters, all_users)
    df_txn_stats = compute_transaction_stats(all_transactions, all_rosters, all_users)
    df_all_play = compute_all_play(all_matchups, all_rosters, all_users)
    df_rivalry = compute_rivalry_index(df_h2h)
    df_past_picks, df_future_picks = compute_first_round_trade_results(
        all_transactions, all_draft_picks, all_rosters, all_users
    )

    return {
        "users": df_users,
        "rosters": df_rosters,
        "matchups": df_matchups,
        "transactions": df_transactions,
        "draft": df_draft,
        "h2h": df_h2h,
        "txn_stats": df_txn_stats,
        "all_play": df_all_play,
        "rivalry": df_rivalry,
        "past_picks": df_past_picks,
        "future_picks": df_future_picks,
    }

# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(
    page_title="Dynasty Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Dynasty Analytics Dashboard")

st.caption("Multi-year Sleeper dynasty analytics — rivalries, luck, transactions, and 1st-round pick trades.")

with st.spinner("Loading Sleeper data and computing analytics..."):
    data = load_all_data()

df_users = data["users"]
df_rosters = data["rosters"]
df_matchups = data["matchups"]
df_transactions = data["transactions"]
df_draft = data["draft"]
df_h2h = data["h2h"]
df_txn_stats = data["txn_stats"]
df_all_play = data["all_play"]
df_rivalry = data["rivalry"]
df_past_picks = data["past_picks"]
df_future_picks = data["future_picks"]

section = st.sidebar.radio(
    "Sections",
    [
        "Rivalries",
        "All-Play & Luck",
        "Transactions & Archetypes",
        "Head-to-Head",
        "1st-Round Pick Trades",
        "Best/Worst Seasons",
    ]
)

# -----------------------------
# Section: Rivalries
# -----------------------------
if section == "Rivalries":
    st.header("Rivalry Index")

    st.write("Top rivalries by intensity (games × distance from .500).")
    top_n = st.slider("Show top N rivalries", 5, 50, 20)
    st.dataframe(df_rivalry.head(top_n), use_container_width=True)

    if not df_rivalry.empty:
        fig = px.scatter(
            df_rivalry,
            x="games",
            y="win_pct",
            size="rivalry_score",
            color="owner_1",
            hover_data=["owner_1", "owner_2", "wins", "losses"],
            title="Rivalry Landscape"
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Section: All-Play & Luck
# -----------------------------
elif section == "All-Play & Luck":
    st.header("All-Play Standings & Luck Index")

    st.subheader("All-Play Standings")
    st.dataframe(
        df_all_play.sort_values("all_play_pct", ascending=False),
        use_container_width=True
    )

    st.subheader("Luck Index (All-Play vs Actual)")

    # Approximate actual wins by counting matchup wins
    if not df_matchups.empty:
        df = df_matchups.copy()
        # Need roster_id -> user
        roster_owner = {r["roster_id"]: r["owner_id"] for _, r in df_rosters.iterrows()}
        user_map = {u["user_id"]: u["display_name"] for _, u in df_users.iterrows()}

        # Compute actual wins per roster
        wins = defaultdict(int)
        for (_, group) in df.groupby(["league_id", "week", "matchup_id"]):
            if len(group) != 2:
                continue
            a, b = group.iloc[0], group.iloc[1]
            if a["points"] > b["points"]:
                wins[a["roster_id"]] += 1
            elif b["points"] > a["points"]:
                wins[b["roster_id"]] += 1

        rows = []
        for rid, w in wins.items():
            owner_id = roster_owner.get(rid)
            user = user_map.get(owner_id, f"Roster {rid}")
            rows.append({"roster_id": rid, "user": user, "actual_wins": w})
        df_actual = pd.DataFrame(rows)

        luck = df_all_play.merge(df_actual, on="user", how="left")
        luck["games_all_play"] = luck["all_play_wins"] + luck["all_play_losses"]
        luck["expected_wins"] = luck["all_play_pct"] * luck["games_all_play"]
        luck["luck_score"] = luck["actual_wins"] - luck["expected_wins"]

        st.dataframe(
            luck.sort_values("luck_score", ascending=False),
            use_container_width=True
        )

        fig = px.scatter(
            luck,
            x="expected_wins",
            y="actual_wins",
            text="user",
            color="luck_score",
            color_continuous_scale="RdBu",
            title="Luck Index: Actual Wins vs Expected (All-Play)"
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No matchup data available to compute luck index.")

# -----------------------------
# Section: Transactions & Archetypes
# -----------------------------
elif section == "Transactions & Archetypes":
    st.header("Transaction Profiles")

    st.subheader("Raw Transaction Stats")
    st.dataframe(
        df_txn_stats.sort_values("trades", ascending=False),
        use_container_width=True
    )

    st.subheader("Manager Archetypes")

    arche = df_txn_stats.copy()
    arche["archetype"] = ""

    median_trades = arche["trades"].median()
    median_adds = arche["adds"].median()
    median_fab = arche["fab_spent"].median()
    median_drops = arche["drops"].median()

    arche.loc[arche["trades"] > median_trades, "archetype"] += "Trader "
    arche.loc[arche["adds"] > median_adds, "archetype"] += "Waiver Goblin "
    arche.loc[arche["fab_spent"] > median_fab, "archetype"] += "FAB Maniac "
    arche.loc[arche["drops"] < median_drops, "archetype"] += "Set-and-Forget "

    st.dataframe(arche[["user", "trades", "adds", "drops", "fab_spent", "archetype"]],
                 use_container_width=True)

# -----------------------------
# Section: Head-to-Head
# -----------------------------
elif section == "Head-to-Head":
    st.header("Lifetime Head-to-Head")

    st.write("Head-to-head records across all seasons.")
    st.dataframe(
        df_h2h.sort_values("wins", ascending=False),
        use_container_width=True
    )

    owners = sorted(set(df_h2h["owner_1"]).union(df_h2h["owner_2"]))
    selected_owner = st.selectbox("Filter by manager", ["(All)"] + owners)

    if selected_owner != "(All)":
        mask = (df_h2h["owner_1"] == selected_owner) | (df_h2h["owner_2"] == selected_owner)
        st.dataframe(df_h2h[mask].sort_values("wins", ascending=False),
                     use_container_width=True)

# -----------------------------
# Section: 1st-Round Pick Trades
# -----------------------------
elif section == "1st-Round Pick Trades":
    st.header("1st-Round Pick Trade Explorer")

    st.subheader("Past 1st-Round Picks (Already Drafted)")

    if df_past_picks.empty:
        st.info("No past 1st-round pick trades mapped to drafted players yet.")
    else:
        st.dataframe(
            df_past_picks[[
                "season", "original_owner", "current_owner",
                "player_selected", "pick_no",
                "transaction_id", "trade_adds", "trade_drops"
            ]].sort_values(["season", "pick_no"]),
            use_container_width=True
        )

        st.subheader("Filter by manager")
        managers = sorted(
            set(df_past_picks["original_owner"]).union(df_past_picks["current_owner"])
        )
        sel_mgr = st.selectbox("Manager", ["(All)"] + managers, key="past_mgr")
        if sel_mgr != "(All)":
            mask = (df_past_picks["original_owner"] == sel_mgr) | \
                   (df_past_picks["current_owner"] == sel_mgr)
            st.dataframe(
                df_past_picks[mask].sort_values(["season", "pick_no"]),
                use_container_width=True
            )

    st.subheader("Future 1st-Round Picks (Not Yet Drafted)")
    if df_future_picks.empty:
        st.info("No future 1st-round pick trades found.")
    else:
        st.dataframe(
            df_future_picks[[
                "season", "original_owner", "current_owner",
                "transaction_id", "trade_draft_picks"
            ]].sort_values("season"),
            use_container_width=True
        )

# -----------------------------
# Section: Best/Worst Seasons
# -----------------------------
elif section == "Best/Worst Seasons":
    st.header("Best & Worst Seasons (Total Points For)")

    if df_matchups.empty:
        st.info("No matchup data available.")
    else:
        roster_owner = {r["roster_id"]: r["owner_id"] for _, r in df_rosters.iterrows()}
        user_map = {u["user_id"]: u["display_name"] for _, u in df_users.iterrows()}

        df = df_matchups.copy()
        df["season"] = df["league_id"].map(
            {lid: api_get(f"{BASE}/league/{lid}")["season"] for lid in LEAGUE_IDS}
        )

        agg = df.groupby(["season", "roster_id"])["points"].sum().reset_index()
        agg["owner_id"] = agg["roster_id"].map(roster_owner)
        agg["user"] = agg["owner_id"].map(user_map)

        st.subheader("Top Seasons by Points For")
        st.dataframe(
            agg.sort_values("points", ascending=False).head(20),
            use_container_width=True
        )

        st.subheader("Bottom Seasons by Points For")
        st.dataframe(
            agg.sort_values("points", ascending=True).head(20),
            use_container_width=True
        )

        fig = px.bar(
            agg.sort_values("points", ascending=False).head(30),
            x="user",
            y="points",
            color="season",
            title="Top 30 Seasons by Points For"
        )
        st.plotly_chart(fig, use_container_width=True)
