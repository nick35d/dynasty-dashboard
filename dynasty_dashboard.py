import requests
import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache

SLEEPER_BASE = "https://api.sleeper.app/v1"


# -----------------------------
# Helpers
# -----------------------------
def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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
    league_ids = []
    current_id = base_league_id

    while True:
        league = get_league(current_id)
        league_ids.append(current_id)
        prev = league.get("previous_league_id")
        if prev in [None, "0", 0, ""]:
            break
        current_id = str(prev)

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
    win_matrix = compute_head_to_head(league_ids)
    if win_matrix.empty:
        st.info("No head-to-head data found.")
        return

    teams = win_matrix.index.tolist()
    loss_matrix = win_matrix.T

    display = pd.DataFrame(index=teams, columns=teams, dtype=object)
    for r in teams:
        for c in teams:
            if r == c:
                display.loc[r, c] = "—"
            else:
                w = int(win_matrix.loc[r, c])
                l = int(loss_matrix.loc[r, c])
                display.loc[r, c] = f"{w}-{l}"

    style_df = pd.DataFrame("", index=teams, columns=teams, dtype=object)
    for r in teams:
        for c in teams:
            if r == c:
                style_df.loc[r, c] = "color: gray"
            else:
                w = win_matrix.loc[r, c]
                l = loss_matrix.loc[r, c]
                if w > l:
                    style_df.loc[r, c] = "color: green"
                elif w < l:
                    style_df.loc[r, c] = "color: red"
                else:
                    style_df.loc[r, c] = "color: gray"

    styled = display.style.apply(lambda _: style_df, axis=None)
    st.dataframe(styled, use_container_width=True)


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
                if name_a == pair[0]:
                    pa, pb = sa, sb
                else:
                    pa, pb = sb, sa
                records.append(
                    {
                        "season": season,
                        "team_a": pair[0],
                        "team_b": pair[1],
                        "points_a": pa,
                        "points_b": pb,
                    }
                )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["games"] = 1
    df["diff"] = df["points_a"] - df["points_b"]
    df["win_a"] = (df["points_a"] > df["points_b"]).astype(int)
    df["win_b"] = (df["points_b"] > df["points_a"]).astype(int)

    agg = (
        df.groupby(["team_a", "team_b"])
        .agg(
            games=("games", "sum"),
            total_diff=("diff", "sum"),
            avg_margin=("diff", "mean"),
            wins_a=("win_a", "sum"),
            wins_b=("win_b", "sum"),
        )
        .reset_index()
    )

    agg["record"] = agg["wins_a"].astype(int).astype(str) + "-" + agg["wins_b"].astype(
        int
    ).astype(str)

    agg["closeness_score"] = 1 / (1 + agg["avg_margin"].abs())
    win_pct_a = agg["wins_a"] / agg["games"]
    win_pct_b = agg["wins_b"] / agg["games"]
    agg["competitiveness_score"] = 1 - (win_pct_a - win_pct_b).abs()
    agg["raw_score"] = (
        agg["games"] * agg["closeness_score"] * agg["competitiveness_score"]
    )

    max_raw = agg["raw_score"].max()
    if max_raw > 0:
        agg["rivalry_score"] = 100 * agg["raw_score"] / max_raw
    else:
        agg["rivalry_score"] = 0

    agg = agg.sort_values("rivalry_score", ascending=False)
    return agg


def render_top_rivalries(league_ids: list):
    st.header("Top Rivalries")
    df = compute_rivalries(league_ids)
    if df.empty:
        st.info("No rivalry data found.")
        return

    df = df.copy()
    df["rivalry_score_display"] = df["rivalry_score"].round(0).astype(int).astype(str)
    df["rivalry_score_display"] = "⭐ " + df["rivalry_score_display"]

    top_n = st.slider("Number of rivalries to show", 5, 50, 10)
    st.subheader("Top Rivalries (Ranked)")
    st.dataframe(
        df.head(top_n)[
            ["team_a", "team_b", "record", "games", "avg_margin", "rivalry_score_display"]
        ],
        use_container_width=True,
    )


# -----------------------------
# Playoff History (6-team, correct weeks, winners bracket only)
# -----------------------------
def compute_playoff_history(league_ids: list):
    records = []
    for league_id in league_ids:
        league = get_league(league_id)
        season_raw = league.get("season")
        if season_raw is None:
            continue
        season = int(season_raw)

        # Explicit playoff weeks based on your league:
        # 2020: weeks 14–16; 2021+: weeks 15–17
        if season == 2020:
            playoff_weeks = [14, 15, 16]
        else:
            playoff_weeks = [15, 16, 17]

        rosters = get_league_rosters(league_id)
        playoff_rosters = set()
        for r in rosters:
            r_settings = r.get("settings") or {}
            seed = (
                r_settings.get("seed")
                or r_settings.get("playoff_seed")
                or r_settings.get("p_standings")
                or r_settings.get("rank")
            )
            if seed is None:
                continue
            try:
                seed_int = int(seed)
            except (TypeError, ValueError):
                continue
            # 6-team playoff, seeds 1–6
            if 1 <= seed_int <= 6:
                playoff_rosters.add(r["roster_id"])

        if not playoff_rosters:
            continue

        roster_name_map = build_roster_user_map(league_id)

        for week in playoff_weeks:
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

                # Winners bracket only: both must be playoff teams
                if ra not in playoff_rosters or rb not in playoff_rosters:
                    continue

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
                        "winner_name": roster_name_map.get(winner, f"Roster {winner}"),
                        "loser_name": roster_name_map.get(loser, f"Roster {loser}"),
                        "winner_points": wp,
                        "loser_points": lp,
                    }
                )

    if not records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(records)

    # Champion logic: team with playoff wins and zero playoff losses
    champs = []
    for season, g in df.groupby("season"):
        # Compute per-team playoff record within this season
        wins = g.groupby("winner_name").size().rename("wins")
        losses = g.groupby("loser_name").size().rename("losses")
        rec = pd.concat([wins, losses], axis=1).fillna(0)
        rec["wins"] = rec["wins"].astype(int)
        rec["losses"] = rec["losses"].astype(int)

        # Candidates: at least one win, zero losses
        candidates = rec[(rec["wins"] > 0) & (rec["losses"] == 0)]
        if candidates.empty:
            continue

        # If multiple, break ties by:
        # 1) most wins
        # 2) latest playoff week they won
        # 3) highest total playoff points
        candidates = candidates.sort_values("wins", ascending=False)
        top_wins = candidates["wins"].max()
        candidates = candidates[candidates["wins"] == top_wins]

        if len(candidates) == 1:
            champ_name = candidates.index[0]
        else:
            # Tie-breaker 2: latest win week
            latest_win_week = {}
            total_points = {}
            for team in candidates.index:
                team_games = g[(g["winner_name"] == team) | (g["loser_name"] == team)]
                win_games = g[g["winner_name"] == team]
                if not win_games.empty:
                    latest_win_week[team] = win_games["week"].max()
                else:
                    latest_win_week[team] = -1
                total_points[team] = (
                    team_games["winner_points"].sum() + team_games["loser_points"].sum()
                )
            max_week = max(latest_win_week.values())
            week_candidates = [t for t, w in latest_win_week.items() if w == max_week]
            if len(week_candidates) == 1:
                champ_name = week_candidates[0]
            else:
                # Tie-breaker 3: highest total points
                champ_name = max(week_candidates, key=lambda t: total_points[t])

        champs.append({"season": season, "champion": champ_name})

    titles = pd.DataFrame(champs)

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

    st.subheader("Playoff Records (Winners Bracket Only)")
    st.dataframe(
        rec.sort_values(["playoff_win_pct", "playoff_wins"], ascending=[False, False]),
        use_container_width=True,
    )

    st.subheader("Playoff Matchups (Winners Bracket)")
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

    for col in ["trade_count", "waiver_count", "free_agent_count"]:
        if col not in out.columns:
            out[col] = 0

    out["total_moves"] = (
        out["trade_count"]
        + out["waiver_count"]
        + out["free_agent_count"]
        + out["adds"]
        + out["drops"]
    )

    trade_pct = out["trade_count"].rank(pct=True) * 100
    waiver_pct = (out["waiver_count"] + out["free_agent_count"]).rank(pct=True) * 100
    total_pct = out["total_moves"].rank(pct=True) * 100

    def classify(row):
        t = trade_pct.loc[row.name]
        w = waiver_pct.loc[row.name]
        tot = total_pct.loc[row.name]

        if t >= 90 and w >= 90:
            return "The Chaos Agent"
        if t >= 80:
            return "The Trader"
        if w >= 80:
            return "The Streamer"
        if tot <= 20:
            return "The Hoarder"
        if tot <= 40:
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
        "total_moves",
        "archetype",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    st.dataframe(
        df[existing_cols].sort_values("trade_count", ascending=False),
        use_container_width=True,
    )

    st.markdown("### Archetype Glossary")
    st.markdown(
        """
- **The Chaos Agent:** Top 10% in both trades and waiver/free-agent activity. High-volume, unpredictable, reshapes the league landscape weekly.
- **The Trader:** Top 20% in trade volume. Aggressive dealmaker who constantly reshapes their roster through trades.
- **The Streamer:** Top 20% in waiver/free-agent moves. Lives on the wire, plays matchups, and churns depth strategically.
- **The Sniper:** Bottom 40% in total moves but high precision. Makes few moves, but they’re targeted and high-impact.
- **The Hoarder:** Bottom 20% in total moves. Rarely trades or churns; prefers stability and long-term roster building.
- **Balanced:** Middle of the distribution. Healthy mix of trades and waivers without extreme tendencies.
"""
    )


# -----------------------------
# Manager Tendencies
# -----------------------------
def compute_manager_tendencies(league_ids: list) -> pd.DataFrame:
    game_records = []
    for league_id in league_ids:
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

                game_records.append(
                    {
                        "team": name_a,
                        "opponent": name_b,
                        "points_for": sa,
                        "points_against": sb,
                    }
                )
                game_records.append(
                    {
                        "team": name_b,
                        "opponent": name_a,
                        "points_for": sb,
                        "points_against": sa,
                    }
                )

    if not game_records:
        return pd.DataFrame()

    games_df = pd.DataFrame(game_records)

    teams = sorted(games_df["team"].unique())
    rows = []
    for team in teams:
        g = games_df[games_df["team"] == team]
        if g.empty:
            continue
        wins = (g["points_for"] > g["points_against"]).sum()
        losses = (g["points_for"] < g["points_against"]).sum()
        games = len(g)
        win_pct = wins / games if games > 0 else np.nan

        margin = g["points_for"] - g["points_against"]
        blowout_wins = ((g["points_for"] > g["points_against"]) & (margin >= 20)).sum()
        close_wins = ((g["points_for"] > g["points_against"]) & (margin <= 5)).sum()
        blowout_losses = (
            (g["points_for"] < g["points_against"]) & (margin <= -20)
        ).sum()
        close_losses = (
            (g["points_for"] < g["points_against"]) & (margin >= -5)
        ).sum()

        avg_pf = g["points_for"].mean()
        avg_pa = g["points_against"].mean()
        consistency = g["points_for"].std(ddof=0) if games > 1 else 0.0

        total_pf = g["points_for"].sum()
        total_pa = g["points_against"].sum()
        if total_pf + total_pa > 0:
            exp_win_pct = (total_pf ** 2) / (total_pf ** 2 + total_pa ** 2)
            expected_wins = exp_win_pct * games
        else:
            expected_wins = 0.0
        luck = wins - expected_wins

        rows.append(
            {
                "team": team,
                "wins": wins,
                "losses": losses,
                "games": games,
                "win_pct": win_pct,
                "blowout_wins": blowout_wins,
                "close_wins": close_wins,
                "blowout_losses": blowout_losses,
                "close_losses": close_losses,
                "avg_pf": avg_pf,
                "avg_pa": avg_pa,
                "consistency": consistency,
                "luck": luck,
            }
        )

    tendencies = pd.DataFrame(rows)

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

    display_cols = [
        "team",
        "wins",
        "losses",
        "games",
        "win_pct",
        "blowout_wins",
        "close_wins",
        "blowout_losses",
        "close_losses",
        "avg_pf",
        "avg_pa",
        "consistency",
        "luck",
        "archetype",
    ]
    existing = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[existing].sort_values("win_pct", ascending=False),
        use_container_width=True,
    )


# -----------------------------
# 1st-Round Pick Trade Explorer (trade-level only + team filter)
# -----------------------------
def compute_first_round_trades(league_ids: list):
    season_roster_name = {}
    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        if season is None:
            continue
        season_roster_name[season] = build_roster_user_map(league_id)

    season_round_slot_to_pick = {}
    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        if season is None:
            continue
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
                if rnd != 1:
                    continue
                draft_slot = p.get("draft_slot")
                if draft_slot is None:
                    continue
                pick_no = p.get("pick_no")
                meta = p.get("metadata") or {}
                first = meta.get("first_name", "")
                last = meta.get("last_name", "")
                full_name = f"{first} {last}".strip() or "Unknown Player"
                key = (season, int(rnd), int(draft_slot))
                season_round_slot_to_pick[key] = {
                    "pick_no": pick_no,
                    "player_name": full_name,
                }

    players_meta = get_players()

    def player_name_from_id(pid: str) -> str:
        info = players_meta.get(pid) or {}
        name = info.get("full_name") or info.get("first_name") or ""
        if not name:
            return f"Player {pid}"
        return name

    traded_pick_rows = []
    trade_detail_rows = []

    for league_id in league_ids:
        league = get_league(league_id)
        season = league.get("season")
        if season is None:
            continue
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

            first_round_picks = [dp for dp in draft_picks if dp.get("round") == 1]
            if not first_round_picks:
                continue

            for dp in first_round_picks:
                pick_season = dp.get("season")
                slot_roster_id = dp.get("roster_id")
                prev_owner_id = dp.get("previous_owner_id")
                new_owner_id = dp.get("owner_id")

                season_names = season_roster_name.get(pick_season, roster_name_map)

                original_slot_name = season_names.get(
                    slot_roster_id, f"Roster {slot_roster_id}"
                )
                original_owner_name = season_names.get(
                    prev_owner_id, f"Roster {prev_owner_id}"
                )
                new_owner_name = season_names.get(
                    new_owner_id, f"Roster {new_owner_id}"
                )

                try:
                    slot_int = int(slot_roster_id)
                except (TypeError, ValueError):
                    slot_int = None

                if slot_int is not None:
                    key = (pick_season, 1, slot_int)
                    pick_info = season_round_slot_to_pick.get(key, {})
                else:
                    pick_info = {}

                pick_no = pick_info.get("pick_no")
                player_name = pick_info.get("player_name", "Unknown Player")

                if pick_no is not None:
                    pick_label = f"1.{int(pick_no):02d}"
                else:
                    pick_label = "1.??"

                player_display = f"{player_name} ({pick_label})"

                traded_pick_rows.append(
                    {
                        "season": pick_season,
                        "pick": pick_label,
                        "original_slot": original_slot_name,
                        "original_owner": original_owner_name,
                        "new_owner": new_owner_name,
                        "player_selected": player_display,
                    }
                )

            players_for_trade = []
            for dp in first_round_picks:
                pick_season = dp.get("season")
                slot_roster_id = dp.get("roster_id")

                season_names = season_roster_name.get(pick_season, roster_name_map)
                slot_name = season_names.get(
                    slot_roster_id, f"Roster {slot_roster_id}"
                )

                try:
                    slot_int = int(slot_roster_id)
                except (TypeError, ValueError):
                    slot_int = None

                if slot_int is not None:
                    key = (pick_season, 1, slot_int)
                    pick_info = season_round_slot_to_pick.get(key, {})
                else:
                    pick_info = {}

                pick_no = pick_info.get("pick_no")
                player_name = pick_info.get("player_name", "Unknown Player")

                if pick_no is not None:
                    pick_label = f"1.{int(pick_no):02d}"
                else:
                    pick_label = "1.??"

                players_for_trade.append(
                    f"{pick_season} 1st ({pick_label}) → {player_name} (slot {slot_name})"
                )

            adds = tx.get("adds") or {}
            drops = tx.get("drops") or {}

            sent_assets_by_team = {rid: [] for rid in consenter_ids}

            for dp in draft_picks:
                pick_season = dp.get("season")
                rnd = dp.get("round")
                slot = dp.get("roster_id")
                prev_owner = dp.get("previous_owner_id")

                season_names = season_roster_name.get(pick_season, roster_name_map)
                slot_name = season_names.get(slot, f"Roster {slot}")

                label = f"{pick_season} {ordinal(rnd)} (from {slot_name})"
                if prev_owner in sent_assets_by_team:
                    sent_assets_by_team[prev_owner].append(label)

            for pid, rid in drops.items():
                name = player_name_from_id(pid)
                sender = rid
                if sender in sent_assets_by_team:
                    sent_assets_by_team[sender].append(name)

            teams_involved_names = [
                roster_name_map.get(rid, f"Roster {rid}") for rid in consenter_ids
            ]
            teams_involved = " ↔ ".join(teams_involved_names)

            sent_lines = []
            for rid in consenter_ids:
                nm = roster_name_map.get(rid, f"Roster {rid}")
                assets = sent_assets_by_team.get(rid, [])
                if assets:
                    sent_lines.append(f"{nm} sent: " + ", ".join(assets))
                else:
                    sent_lines.append(f"{nm} sent: (no assets)")

            picks_exchanged_text = "\n".join(sent_lines)
            players_selected_text = "\n".join(players_for_trade)

            trade_detail_rows.append(
                {
                    "season": season,
                    "teams_involved": teams_involved,
                    "picks_exchanged": picks_exchanged_text,
                    "players_selected": players_selected_text,
                }
            )

    traded_picks_df = pd.DataFrame(traded_pick_rows).sort_values(
        ["season", "pick", "original_slot"]
    )
    trade_details_df = pd.DataFrame(trade_detail_rows).sort_values(
        ["season", "teams_involved"]
    )

    return traded_picks_df, trade_details_df


def render_first_round_pick_explorer(league_ids: list):
    st.header("1st-Round Pick Trade Explorer")

    _, trade_details_df = compute_first_round_trades(league_ids)

    if trade_details_df.empty:
        st.info("No traded 1st-round picks found.")
        return

    # Option B: dropdown shows ALL teams in league history, even if they never traded a 1st
    all_team_names = set()
    for league_id in league_ids:
        roster_name_map = build_roster_user_map(league_id)
        all_team_names.update(roster_name_map.values())

    team_options = ["All Teams"] + sorted(all_team_names)
    selected_team = st.selectbox("Filter by team", team_options)

    if selected_team != "All Teams":
        mask = (
            trade_details_df["teams_involved"].str.contains(selected_team, na=False)
            | trade_details_df["picks_exchanged"].str.contains(selected_team, na=False)
            | trade_details_df["players_selected"].str.contains(selected_team, na=False)
        )
        filtered_df = trade_details_df[mask]
    else:
        filtered_df = trade_details_df

    st.subheader("Trade Details (One Row per Trade)")
    st.dataframe(
        filtered_df[
            ["season", "teams_involved", "picks_exchanged", "players_selected"]
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
        value="1180359904212156416",
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
