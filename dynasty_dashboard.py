import requests
import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache

SLEEPER_BASE = "https://api.sleeper.app/v1"


# -----------------------------
# Sleeper API helpers
# -----------------------------
@lru_cache(maxsize=None)
def get_league(league_id: str) -> dict:
    r = requests.get(f"{SLEEPER_BASE}/league/{league_id}")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=None)
def get_league_rosters(league_id: str) -> list:
    r = requests.get(f"{SLEEPER_BASE}/league/{league_id}/rosters")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=None)
def get_league_users(league_id: str) -> list:
    r = requests.get(f"{SLEEPER_BASE}/league/{league_id}/users")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=None)
def get_league_matchups(league_id: str, week: int) -> list:
    r = requests.get(f"{SLEEPER_BASE}/league/{league_id}/matchups/{week}")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=None)
def get_league_transactions(league_id: str, week: int) -> list:
    r = requests.get(f"{SLEEPER_BASE}/league/{league_id}/transactions/{week}")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=None)
def get_league_drafts(league_id: str) -> list:
    r = requests.get(f"{SLEEPER_BASE}/league/{league_id}/drafts")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=None)
def get_draft_picks(draft_id: str) -> list:
    r = requests.get(f"{SLEEPER_BASE}/draft/{draft_id}/picks")
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=None)
def get_players() -> dict:
    r = requests.get(f"{SLEEPER_BASE}/players/nfl")
    r.raise_for_status()
    return r.json()


# -----------------------------
# League hierarchy helpers
# -----------------------------
def get_all_league_ids(base_league_id: str) -> list:
    """Follow previous_league_id chain backwards to get all seasons, stopping safely on '0'/None."""
    league_ids = []
    current_id = base_league_id

    while True:
        league = get_league(current_id)
        league_ids.append(current_id)

        prev = league.get("previous_league_id")

        # Sleeper uses 0 / "0" / None / "" to mean "no previous league"
        if prev in [None, "0", 0, ""]:
            break

        current_id = prev

    return league_ids


def build_roster_user_map(league_id: str) -> dict:
    rosters = get_league_rosters(league_id)
    users = get_league_users(league_id)
    user_map = {u["user_id"]: u.get("display_name", f"user_{u['user_id']}") for u in users}
    roster_map = {}
    for r in rosters:
        owner_id = r.get("owner_id")
        name = user_map.get(owner_id, f"Roster {r['roster_id']}")
        roster_map[r["roster_id"]] = name
    return roster_map


# -----------------------------
# Head-to-Head
# -----------------------------
def compute_head_to_head(league_ids: list) -> pd.DataFrame:
    records = []
    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        rosters = get_league_rosters(league_id)
        roster_id_to_name = build_roster_user_map(league_id)

        for week in range(1, 19):
            matchups = get_league_matchups(league_id, week)
            if not matchups:
                continue
            df = pd.DataFrame(matchups)
            if "matchup_id" not in df.columns:
                continue
            for _, group in df.groupby("matchup_id"):
                if len(group) != 2:
                    continue
                a, b = group.iloc[0], group.iloc[1]
                ra, rb = a["roster_id"], b["roster_id"]
                sa, sb = a.get("points", 0), b.get("points", 0)
                if sa == sb:
                    continue
                if sa > sb:
                    winner, loser = ra, rb
                else:
                    winner, loser = rb, ra
                records.append(
                    {
                        "season": season,
                        "winner_roster_id": winner,
                        "loser_roster_id": loser,
                        "winner_name": roster_id_to_name.get(winner, f"Roster {winner}"),
                        "loser_name": roster_id_to_name.get(loser, f"Roster {loser}"),
                    }
                )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    teams = sorted(set(df["winner_name"]).union(df["loser_name"]))
    matrix = pd.DataFrame(0, index=teams, columns=teams, dtype=int)

    for _, row in df.iterrows():
        w = row["winner_name"]
        l = row["loser_name"]
        matrix.loc[w, l] += 1

    return matrix


def render_head_to_head(league_ids: list):
    st.header("Head-to-Head Matrix (All-Time)")
    matrix = compute_head_to_head(league_ids)
    if matrix.empty:
        st.info("No head-to-head data found.")
        return
    st.dataframe(matrix.style.format("{:d}"))


# -----------------------------
# Top Rivalries
# -----------------------------
def compute_rivalries(league_ids: list) -> pd.DataFrame:
    records = []
    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        roster_name_map = build_roster_user_map(league_id)

        for week in range(1, 19):
            matchups = get_league_matchups(league_id, week)
            if not matchups:
                continue
            df = pd.DataFrame(matchups)
            if "matchup_id" not in df.columns:
                continue
            for _, group in df.groupby("matchup_id"):
                if len(group) != 2:
                    continue
                a, b = group.iloc[0], group.iloc[1]
                ra, rb = a["roster_id"], b["roster_id"]
                sa, sb = a.get("points", 0), b.get("points", 0)
                name_a = roster_name_map.get(ra, f"Roster {ra}")
                name_b = roster_name_map.get(rb, f"Roster {rb}")
                pair = tuple(sorted([name_a, name_b]))
                margin = sa - sb
                records.append(
                    {
                        "season": season,
                        "team_a": pair[0],
                        "team_b": pair[1],
                        "points_a": sa if name_a == pair[0] else sb,
                        "points_b": sb if name_b == pair[1] else sa,
                    }
                )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["games"] = 1
    df["diff"] = df["points_a"] - df["points_b"]

    agg = (
        df.groupby(["team_a", "team_b"])
        .agg(
            games=("games", "sum"),
            total_diff=("diff", "sum"),
            avg_margin=("diff", "mean"),
        )
        .reset_index()
    )

    agg["rivalry_score"] = agg["games"] * (1 / (1 + agg["total_diff"].abs()))
    agg = agg.sort_values("rivalry_score", ascending=False)
    return agg


def render_top_rivalries(league_ids: list):
    st.header("Top Rivalries")
    df = compute_rivalries(league_ids)
    if df.empty:
        st.info("No rivalry data found.")
        return

    top_n = st.slider("Number of rivalries to show", 5, 50, 10)
    st.subheader("Top Rivalries (Ranked)")
    st.dataframe(
        df.head(top_n)[
            ["team_a", "team_b", "games", "total_diff", "avg_margin", "rivalry_score"]
        ],
        use_container_width=True,
    )


# -----------------------------
# Playoff History
# -----------------------------
def compute_playoff_history(league_ids: list):
    records = []
    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        settings = league.get("settings", {})
        playoff_week_start = settings.get("playoff_week_start")
        if not playoff_week_start:
            continue

        roster_name_map = build_roster_user_map(league_id)

        for week in range(playoff_week_start, playoff_week_start + 4):
            matchups = get_league_matchups(league_id, week)
            if not matchups:
                continue
            df = pd.DataFrame(matchups)
            if "matchup_id" not in df.columns:
                continue
            for _, group in df.groupby("matchup_id"):
                if len(group) != 2:
                    continue
                a, b = group.iloc[0], group.iloc[1]
                ra, rb = a["roster_id"], b["roster_id"]
                sa, sb = a.get("points", 0), b.get("points", 0)
                if sa == sb:
                    continue
                if sa > sb:
                    winner, loser = ra, rb
                    wp, lp = sa, sb
                else:
                    winner, loser = rb, ra
                    wp, lp = sb, sa
                records.append(
                    {
                        "season": season,
                        "week": week,
                        "winner_roster_id": winner,
                        "loser_roster_id": loser,
                        "winner_name": roster_name_map.get(
                            winner, f"Roster {winner}"
                        ),
                        "loser_name": roster_name_map.get(loser, f"Roster {loser}"),
                        "winner_points": wp,
                        "loser_points": lp,
                    }
                )

    if not records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(records)

    titles = (
        df.sort_values(["season", "week"])
        .groupby("season")
        .tail(1)[["season", "winner_name"]]
        .rename(columns={"winner_name": "champion"})
    )

    wins = df.groupby("winner_name").size().rename("playoff_wins")
    losses = df.groupby("loser_name").size().rename("playoff_losses")
    rec = pd.concat([wins, losses], axis=1).fillna(0)
    rec["playoff_games"] = rec["playoff_wins"] + rec["playoff_losses"]
    rec["playoff_win_pct"] = np.where(
        rec["playoff_games"] > 0,
        rec["playoff_wins"] / rec["playoff_games"],
        np.nan,
    )
    rec = rec.reset_index().rename(columns={"index": "team"})

    return df, titles, rec


def render_playoff_history(league_ids: list):
    st.header("Playoff History")
    df, titles, rec = compute_playoff_history(league_ids)
    if df.empty:
        st.info("No playoff data found.")
        return

    st.subheader("Champions by Season")
    st.dataframe(titles.sort_values("season"), use_container_width=True)

    st.subheader("Playoff Records")
    st.dataframe(
        rec.sort_values("playoff_win_pct", ascending=False),
        use_container_width=True,
    )

    st.subheader("Playoff Matchups (All-Time)")
    st.dataframe(
        df[
            [
                "season",
                "week",
                "winner_name",
                "winner_points",
                "loser_name",
                "loser_points",
            ]
        ].sort_values(["season", "week"]),
        use_container_width=True,
    )


# -----------------------------
# Team Transaction Profiles + Archetypes
# -----------------------------
def compute_transaction_profiles(league_ids: list) -> pd.DataFrame:
    records = []
    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        roster_name_map = build_roster_user_map(league_id)

        for week in range(1, 19):
            txs = get_league_transactions(league_id, week)
            if not txs:
                continue
            for tx in txs:
                ttype = tx.get("type")
                status = tx.get("status")
                if status != "complete":
                    continue
                roster_ids = tx.get("roster_ids") or []
                for rid in roster_ids:
                    records.append(
                        {
                            "season": season,
                            "roster_id": rid,
                            "team": roster_name_map.get(rid, f"Roster {rid}"),
                            "type": ttype,
                            "adds": len(tx.get("adds") or {}),
                            "drops": len(tx.get("drops") or {}),
                            "waiver_bid": (tx.get("settings") or {}).get(
                                "waiver_bid", 0
                            ),
                        }
                    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    agg = (
        df.groupby(["team", "type"])
        .agg(
            adds=("adds", "sum"),
            drops=("drops", "sum"),
            faab_spent=("waiver_bid", "sum"),
            tx_count=("type", "size"),
        )
        .reset_index()
    )

    wide = agg.pivot(index="team", columns="type", values="tx_count").fillna(0)
    wide.columns = [f"{c}_count" for c in wide.columns]

    adds_drops = (
        df.groupby("team")[["adds", "drops", "waiver_bid"]].sum().rename(
            columns={"waiver_bid": "faab_spent"}
        )
    )

    out = wide.join(adds_drops, how="outer").fillna(0)

    out["trade_aggressiveness"] = out.get("trade_count", 0)
    out["waiver_aggressiveness"] = out.get("waiver_count", 0) + out.get(
        "free_agent_count", 0
    )
    out["roster_churn"] = out["adds"] + out["drops"]

    def classify(row):
        t = row.get("trade_aggressiveness", 0)
        w = row.get("waiver_aggressiveness", 0)
        churn = row.get("roster_churn", 0)

        if t > 10 and churn > 40:
            return "The Chaos Agent"
        if t > 10:
            return "The Trader"
        if churn < 10:
            return "The Hoarder"
        if w > 30:
            return "The Streamer"
        if t < 5 and churn < 25:
            return "The Sniper"
        return "Balanced"

    out["archetype"] = out.apply(classify, axis=1)
    out = out.reset_index()
    return out


def render_transaction_profiles(league_ids: list):
    st.header("Team Transaction Profiles & Archetypes")
    df = compute_transaction_profiles(league_ids)
    if df.empty:
        st.info("No transaction data found.")
        return

    cols = [
        "team",
        "trade_count",
        "waiver_count",
        "free_agent_count",
        "adds",
        "drops",
        "faab_spent",
        "roster_churn",
        "trade_aggressiveness",
        "waiver_aggressiveness",
        "archetype",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    st.dataframe(
        df[existing_cols].sort_values("trade_aggressiveness", ascending=False),
        use_container_width=True,
    )


# -----------------------------
# Manager Tendencies
# -----------------------------
def compute_manager_tendencies(league_ids: list) -> pd.DataFrame:
    h2h_matrix = compute_head_to_head(league_ids)
    if h2h_matrix.empty:
        return pd.DataFrame()

    teams = h2h_matrix.index.tolist()
    total_wins = h2h_matrix.sum(axis=1)
    total_losses = h2h_matrix.sum(axis=0)
    tendencies = pd.DataFrame(
        {
            "team": teams,
            "wins": total_wins.values,
            "losses": total_losses.values,
        }
    )
    tendencies["games"] = tendencies["wins"] + tendencies["losses"]
    tendencies["win_pct"] = np.where(
        tendencies["games"] > 0,
        tendencies["wins"] / tendencies["games"],
        np.nan,
    )

    tx = compute_transaction_profiles(league_ids)
    if not tx.empty:
        tendencies = tendencies.merge(tx[["team", "archetype"]], on="team", how="left")

    return tendencies


def render_manager_tendencies(league_ids: list):
    st.header("Manager Tendencies")
    df = compute_manager_tendencies(league_ids)
    if df.empty:
        st.info("No tendencies data found.")
        return

    st.dataframe(
        df.sort_values("win_pct", ascending=False),
        use_container_width=True,
    )


# -----------------------------
# 1st-Round Pick Trade Explorer
# -----------------------------
def compute_first_round_trades(league_ids: list):
    season_to_league = {}
    for league_id in league_ids:
        league = get_league(league_id)
        season_to_league[league.get("season")] = league_id

    season_roster_name = {}
    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        season_roster_name[season] = build_roster_user_map(league_id)

    season_round_roster_to_pick = {}
    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        drafts = get_league_drafts(league_id)
        if not drafts:
            continue
        for d in drafts:
            if d.get("season") != season:
                continue
            draft_id = d["draft_id"]
            picks = get_draft_picks(draft_id)
            for p in picks:
                rnd = p.get("round")
                rid = p.get("roster_id")
                pick_no = p.get("pick_no")
                meta = p.get("metadata") or {}
                first = meta.get("first_name", "")
                last = meta.get("last_name", "")
                full_name = f"{first} {last}".strip()
                key = (season, rnd, rid)
                season_round_roster_to_pick[key] = {
                    "pick_no": pick_no,
                    "player_name": full_name,
                }

    traded_pick_rows = []
    trade_detail_rows = []

    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        roster_name_map = season_roster_name.get(season, {})

        trades = []
        for week in range(1, 19):
            txs = get_league_transactions(league_id, week)
            if not txs:
                continue
            for tx in txs:
                if tx.get("status") != "complete":
                    continue
                if tx.get("type") != "trade":
                    continue
                trades.append(tx)

        for tx in trades:
            draft_picks = tx.get("draft_picks") or []
            roster_ids = tx.get("roster_ids") or []
            consenter_ids = tx.get("consenter_ids") or roster_ids

            sent_by_team = {rid: [] for rid in roster_ids}

            for dp in draft_picks:
                if dp.get("round") != 1:
                    continue
                pick_season = dp.get("season")
                slot_roster_id = dp.get("roster_id")
                original_owner_id = dp.get("previous_owner_id")
                new_owner_id = dp.get("owner_id")

                original_owner_name = roster_name_map.get(
                    original_owner_id, f"Roster {original_owner_id}"
                )
                new_owner_name = roster_name_map.get(
                    new_owner_id, f"Roster {new_owner_id}"
                )
                original_slot_name = roster_name_map.get(
                    slot_roster_id, f"Roster {slot_roster_id}"
                )

                key = (pick_season, 1, slot_roster_id)
                pick_info = season_round_roster_to_pick.get(key, {})
                pick_no = pick_info.get("pick_no")
                player_name = pick_info.get("player_name", "Unknown Player")

                if pick_no is not None:
                    player_display = f"{player_name} (1.{pick_no:02d})"
                else:
                    player_display = f"{player_name} (1.??)"

                traded_pick_rows.append(
                    {
                        "season": pick_season,
                        "original_owner": original_owner_name,
                        "new_owner": new_owner_name,
                        "original_slot": original_slot_name,
                        "player_selected": player_display,
                    }
                )

            for dp in draft_picks:
                rnd = dp.get("round")
                pick_season = dp.get("season")
                slot_roster_id = dp.get("roster_id")
                prev_owner = dp.get("previous_owner_id")

                slot_name = roster_name_map.get(
                    slot_roster_id, f"Roster {slot_roster_id}"
                )
                if rnd == 1:
                    desc = f"{pick_season} 1st (slot {slot_name})"
                else:
                    desc = f"{pick_season} {rnd}th (slot {slot_name})"

                if prev_owner in sent_by_team:
                    sent_by_team[prev_owner].append(desc)

            if len(consenter_ids) == 2:
                a, b = consenter_ids
                name_a = roster_name_map.get(a, f"Roster {a}")
                name_b = roster_name_map.get(b, f"Roster {b}")
                teams_involved = f"{name_a} ↔ {name_b}"

                sent_a = ", ".join(sent_by_team.get(a, [])) or "None"
                sent_b = ", ".join(sent_by_team.get(b, [])) or "None"

                detailed_breakdown = (
                    f"Team {name_a} sent: {sent_a}\n"
                    f"Team {name_b} sent: {sent_b}"
                )
            else:
                names = [roster_name_map.get(r, f"Roster {r}") for r in consenter_ids]
                teams_involved = " ↔ ".join(names)
                parts = []
                for rid in consenter_ids:
                    nm = roster_name_map.get(rid, f"Roster {rid}")
                    sent = ", ".join(sent_by_team.get(rid, [])) or "None"
                    parts.append(f"Team {nm} sent: {sent}")
                detailed_breakdown = "\n".join(parts)

            outcomes = []
            for dp in draft_picks:
                if dp.get("round") != 1:
                    continue
                pick_season = dp.get("season")
                slot_roster_id = dp.get("roster_id")
                slot_name = roster_name_map.get(
                    slot_roster_id, f"Roster {slot_roster_id}"
                )
                key = (pick_season, 1, slot_roster_id)
                pick_info = season_round_roster_to_pick.get(key, {})
                pick_no = pick_info.get("pick_no")
                player_name = pick_info.get("player_name", "Unknown Player")
                if pick_no is not None:
                    player_display = f"{player_name} (1.{pick_no:02d})"
                else:
                    player_display = f"{player_name} (1.??)"
                outcomes.append(
                    f"{pick_season} 1st (slot {slot_name}) → {player_display}"
                )

            outcomes_text = (
                "\n".join(outcomes) if outcomes else "No 1st-round picks in this trade"
            )

            trade_detail_rows.append(
                {
                    "season": season,
                    "teams_involved": teams_involved,
                    "detailed_breakdown": detailed_breakdown,
                    "first_round_outcomes": outcomes_text,
                }
            )

    traded_picks_df = pd.DataFrame(traded_pick_rows).sort_values(
        ["season", "original_owner"]
    )
    trade_details_df = pd.DataFrame(trade_detail_rows).sort_values(
        ["season", "teams_involved"]
    )

    return traded_picks_df, trade_details_df


def render_first_round_pick_explorer(league_ids: list):
    st.header("1st-Round Pick Trade Explorer")

    traded_picks_df, trade_details_df = compute_first_round_trades(league_ids)

    if traded_picks_df.empty:
        st.info("No traded 1st-round picks found.")
        return

    st.subheader("Traded 1st-Round Picks (One Row per Pick)")
    st.dataframe(
        traded_picks_df[
            ["season", "original_owner", "new_owner", "original_slot", "player_selected"]
        ],
        use_container_width=True,
    )

    st.subheader("Trade Details (One Row per Trade)")
    st.dataframe(
        trade_details_df[
            ["season", "teams_involved", "detailed_breakdown", "first_round_outcomes"]
        ],
        use_container_width=True,
    )


# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title="Dynasty League Dashboard", layout="wide")
    st.title("Dynasty League Dashboard")

    base_league_id = st.text_input(
        "Sleeper League ID (current season)",
        value="1180359904212156416",  # replace with your current league if needed
    )

    if not base_league_id:
        st.stop()

    try:
        league_ids = get_all_league_ids(base_league_id)
    except Exception as e:
        st.error(f"Error loading league hierarchy: {e}")
        st.stop()

    tab = st.sidebar.selectbox(
        "Select View",
        [
            "Head-to-Head",
            "Top Rivalries",
            "Playoff History",
            "Team Transaction Profiles",
            "Manager Tendencies",
            "1st-Round Pick Trade Explorer",
        ],
    )

    if tab == "Head-to-Head":
        render_head_to_head(league_ids)
    elif tab == "Top Rivalries":
        render_top_rivalries(league_ids)
    elif tab == "Playoff History":
        render_playoff_history(league_ids)
    elif tab == "Team Transaction Profiles":
        render_transaction_profiles(league_ids)
    elif tab == "Manager Tendencies":
        render_manager_tendencies(league_ids)
    elif tab == "1st-Round Pick Trade Explorer":
        render_first_round_pick_explorer(league_ids)


if __name__ == "__main__":
    main()
