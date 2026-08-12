"""
Calcula las mismas features que usa el modelo (Elo, forma, h2h, fatiga) pero
para un partido que va a empezar AHORA MISMO entre dos jugadores concretos,
no para un partido histórico ya jugado.

Se apoya en matches_real_history.csv: cuanto más actualizado lo tengas
(reejecutando build_historical_dataset.py con frecuencia), más precisas
serán las señales en vivo.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PRIOR_WEIGHT = 4          # mismo peso de shrinkage que en build_features_real.py
INITIAL_ELO = 1500
K_FACTOR = 24
MIN_PREV_MATCHES = 5      # mínimo de partidos previos para confiar en las stats de un jugador


class LiveFeatureEngine:
    """Se construye UNA VEZ al arrancar signal_generator.py (o se refresca
    periódicamente) y luego responde rápido a cada consulta en vivo."""

    def __init__(self):
        self.matches = pd.read_csv(DATA_DIR / "matches_real_history.csv", parse_dates=["startDate"])
        self.matches = self.matches.sort_values("startDate").reset_index(drop=True)
        self._build_player_stats()
        self._build_elo()
        self._build_h2h_history()

    def _build_player_stats(self):
        """Últimos valores conocidos de cada jugador (forma, fatiga, volatilidad)."""
        long_rows = []
        for _, row in self.matches.iterrows():
            long_rows.append({"player": row["participantAName"], "startDate": row["startDate"],
                               "pts_for": row["teamAScore"], "pts_against": row["teamBScore"],
                               "won": int(row["teamAScore"] > row["teamBScore"])})
            long_rows.append({"player": row["participantBName"], "startDate": row["startDate"],
                               "pts_for": row["teamBScore"], "pts_against": row["teamAScore"],
                               "won": int(row["teamBScore"] > row["teamAScore"])})
        long_df = pd.DataFrame(long_rows).sort_values(["player", "startDate"])

        self.player_stats = {}
        for player, g in long_df.groupby("player"):
            g = g.sort_values("startDate")
            self.player_stats[player] = {
                "n_matches": len(g),
                "avg_pts_for": g["pts_for"].mean(),
                "avg_pts_against": g["pts_against"].mean(),
                "win_rate": g["won"].mean(),
                "volatility": g["pts_for"].std() if len(g) > 1 else 8.0,
                "form5": g["won"].tail(5).mean(),
                "form3": g["won"].tail(3).mean(),
                "form2": g["won"].tail(2).mean(),
                "last_match_time": g["startDate"].max(),
                "recent_match_times": g["startDate"].tail(10).tolist(),
            }

    def _build_elo(self):
        elo = {}
        for _, row in self.matches.iterrows():
            a, b = row["participantAName"], row["participantBName"]
            ea, eb = elo.get(a, INITIAL_ELO), elo.get(b, INITIAL_ELO)
            actual_a = 1.0 if row["teamAScore"] > row["teamBScore"] else 0.0
            expected_a = 1 / (1 + 10 ** ((eb - ea) / 400))
            margin = abs(row["teamAScore"] - row["teamBScore"])
            margin_mult = np.clip(np.log(margin + 1) / np.log(15), 0.5, 2.0)
            delta = K_FACTOR * margin_mult * (actual_a - expected_a)
            elo[a] = ea + delta
            elo[b] = eb - delta
        self.elo = elo

    def _build_h2h_history(self):
        history = {}
        for _, row in self.matches.iterrows():
            a, b = row["participantAName"], row["participantBName"]
            key = frozenset([a, b])
            winner = a if row["teamAScore"] > row["teamBScore"] else b
            history.setdefault(key, []).append(winner)
        self.h2h_history = history

    def get_features(self, player_a: str, player_b: str, now: datetime | None = None) -> dict | None:
        """Devuelve el diccionario de features listo para el modelo, o None
        si no hay histórico suficiente de alguno de los dos jugadores."""
        if now is None:
            now = datetime.now(timezone.utc)

        stats_a = self.player_stats.get(player_a)
        stats_b = self.player_stats.get(player_b)
        if stats_a is None or stats_b is None:
            return None
        if stats_a["n_matches"] < MIN_PREV_MATCHES or stats_b["n_matches"] < MIN_PREV_MATCHES:
            return None

        def hours_since_last(stats):
            last = stats["last_match_time"]
            if last.tzinfo is None:
                last = last.tz_localize("UTC")
            return max((now - last).total_seconds() / 3600, 0)

        def matches_in_last_3h(stats):
            cutoff = now - pd.Timedelta(hours=3)
            times = stats["recent_match_times"]
            return sum(1 for t in times if (t.tz_localize("UTC") if t.tzinfo is None else t) >= cutoff)

        elo_a = self.elo.get(player_a, INITIAL_ELO)
        elo_b = self.elo.get(player_b, INITIAL_ELO)
        elo_expected_prob_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

        key = frozenset([player_a, player_b])
        past_winners = self.h2h_history.get(key, [])
        n_meetings = len(past_winners)
        wins_a = sum(1 for w in past_winners if w == player_a)
        h2h_shrunk_full = (wins_a + PRIOR_WEIGHT * elo_expected_prob_a) / (n_meetings + PRIOR_WEIGHT)
        last5 = past_winners[-5:]
        wins_a_last5 = sum(1 for w in last5 if w == player_a)
        h2h_shrunk_last5 = (wins_a_last5 + PRIOR_WEIGHT * elo_expected_prob_a) / (len(last5) + PRIOR_WEIGHT)

        return {
            "diff_avg_pts_for": stats_a["avg_pts_for"] - stats_b["avg_pts_for"],
            "diff_win_rate": stats_a["win_rate"] - stats_b["win_rate"],
            "diff_form5": stats_a["form5"] - stats_b["form5"],
            "diff_form3": stats_a["form3"] - stats_b["form3"],
            "diff_form2": stats_a["form2"] - stats_b["form2"],
            "diff_net_rating": (stats_a["avg_pts_for"] - stats_a["avg_pts_against"]) -
                               (stats_b["avg_pts_for"] - stats_b["avg_pts_against"]),
            "diff_fatigue": matches_in_last_3h(stats_a) - matches_in_last_3h(stats_b),
            "diff_elo": elo_a - elo_b,
            "elo_expected_prob_a": elo_expected_prob_a,
            "h2h_shrunk_full": h2h_shrunk_full,
            "h2h_shrunk_last5": h2h_shrunk_last5,
            "h2h_n_meetings": n_meetings,
            "avg_pts_for_prev_a": stats_a["avg_pts_for"],
            "avg_pts_for_prev_b": stats_b["avg_pts_for"],
            "volatility_prev_a": stats_a["volatility"],
            "volatility_prev_b": stats_b["volatility"],
            "hours_since_last_a": hours_since_last(stats_a),
            "hours_since_last_b": hours_since_last(stats_b),
            "matches_last_3h_a": matches_in_last_3h(stats_a),
            "matches_last_3h_b": matches_in_last_3h(stats_b),
        }
