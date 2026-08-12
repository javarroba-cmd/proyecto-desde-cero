"""
Feature engineering sobre el dataset REAL descargado de h2hggl.com.

Diferencias clave respecto al prototipo con datos simulados:
- No tenemos "match_num_in_session" ni "hours_since_last" como columnas dadas:
  hay que CALCULARLAS a partir de los timestamps reales de cada jugador.
- No tenemos (todavía) cuotas de casas de apuestas — este script deja el
  hueco listo (market_prob_p1) para cuando integremos el odds scraper.
- Mismo principio que antes: todo cálculo de histórico usa SOLO partidos
  anteriores a la fecha actual (expanding, con shift(1)) para evitar leakage.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Carpeta data/ hermana de la carpeta donde está este script (features/)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "matches_real_history.csv", parse_dates=["startDate"])
df = df.sort_values("startDate").reset_index(drop=True)

# Ratings Elo (ejecuta build_elo_ratings.py ANTES de este script)
elo_df = pd.read_csv(DATA_DIR / "matches_with_elo.csv", parse_dates=["startDate"])
elo_df["externalId"] = elo_df["externalId"].astype(str)
df = df.sort_values("startDate").reset_index(drop=True)
df["externalId"] = df["externalId"].astype(str)

# Formato long: una fila por jugador y partido, para poder calcular históricos
def to_long(df):
    a = df.rename(columns={
        "participantAName": "player", "participantBName": "opponent",
        "teamAScore": "pts_for", "teamBScore": "pts_against",
    }).copy()
    a["is_a"] = 1

    b = df.rename(columns={
        "participantBName": "player", "participantAName": "opponent",
        "teamBScore": "pts_for", "teamAScore": "pts_against",
    }).copy()
    b["is_a"] = 0

    long_df = pd.concat([a, b], ignore_index=True)
    long_df["won"] = (long_df["pts_for"] > long_df["pts_against"]).astype(int)
    long_df = long_df.sort_values(["player", "startDate"]).reset_index(drop=True)
    return long_df

long_df = to_long(df)
grp = long_df.groupby("player")

# --- Fatiga REAL: horas desde el partido anterior de ESE jugador,
# y cuántos partidos lleva jugados en las últimas 3 horas (sesión) ---
long_df["hours_since_last"] = grp["startDate"].transform(
    lambda s: s.diff().dt.total_seconds() / 3600
)
# Si es su primer partido registrado, asumimos descanso completo (no fatiga)
long_df["hours_since_last"] = long_df["hours_since_last"].fillna(24)

def matches_in_last_3h(sub):
    times = sub["startDate"].values
    counts = []
    for i, t in enumerate(times):
        window_start = t - np.timedelta64(3, "h")
        count = ((times[:i] >= window_start) & (times[:i] < t)).sum()
        counts.append(count)
    return pd.Series(counts, index=sub.index)

long_df["matches_last_3h"] = grp.apply(
    lambda sub: matches_in_last_3h(sub)
).reset_index(level=0, drop=True)

# --- Históricos EXPANDING (solo pasado) ---
long_df["avg_pts_for_prev"] = grp["pts_for"].transform(lambda s: s.expanding().mean().shift(1))
long_df["avg_pts_against_prev"] = grp["pts_against"].transform(lambda s: s.expanding().mean().shift(1))
long_df["win_rate_prev"] = grp["won"].transform(lambda s: s.expanding().mean().shift(1))
long_df["volatility_prev"] = grp["pts_for"].transform(lambda s: s.expanding().std().shift(1))
long_df["form5_pts_for"] = grp["pts_for"].transform(lambda s: s.rolling(5, min_periods=2).mean().shift(1))
long_df["form5_win_rate"] = grp["won"].transform(lambda s: s.rolling(5, min_periods=2).mean().shift(1))
long_df["form3_win_rate"] = grp["won"].transform(lambda s: s.rolling(3, min_periods=2).mean().shift(1))
long_df["form2_win_rate"] = grp["won"].transform(lambda s: s.rolling(2, min_periods=2).mean().shift(1))
long_df["n_prev_matches"] = grp.cumcount()

# --- Volver a wide ---
stat_cols = ["avg_pts_for_prev", "avg_pts_against_prev", "win_rate_prev",
             "volatility_prev", "form5_pts_for", "form5_win_rate",
             "form3_win_rate", "form2_win_rate",
             "n_prev_matches", "hours_since_last", "matches_last_3h"]

a_stats = long_df.loc[long_df["is_a"] == 1, ["externalId", "player"] + stat_cols]
a_stats = a_stats.rename(columns={c: f"{c}_a" for c in stat_cols})
b_stats = long_df.loc[long_df["is_a"] == 0, ["externalId", "player"] + stat_cols]
b_stats = b_stats.rename(columns={c: f"{c}_b" for c in stat_cols})

merged = df.merge(a_stats, on="externalId", how="left").drop(columns=["player"]) \
           .merge(b_stats, on="externalId", how="left").drop(columns=["player"])

merged["diff_avg_pts_for"] = merged["avg_pts_for_prev_a"] - merged["avg_pts_for_prev_b"]
merged["diff_win_rate"] = merged["win_rate_prev_a"] - merged["win_rate_prev_b"]
merged["diff_form5"] = merged["form5_win_rate_a"] - merged["form5_win_rate_b"]
merged["diff_form3"] = merged["form3_win_rate_a"] - merged["form3_win_rate_b"]
merged["diff_form2"] = merged["form2_win_rate_a"] - merged["form2_win_rate_b"]
merged["net_rating_a"] = merged["avg_pts_for_prev_a"] - merged["avg_pts_against_prev_a"]
merged["net_rating_b"] = merged["avg_pts_for_prev_b"] - merged["avg_pts_against_prev_b"]
merged["diff_net_rating"] = merged["net_rating_a"] - merged["net_rating_b"]
merged["diff_fatigue"] = merged["matches_last_3h_a"] - merged["matches_last_3h_b"]

merged = merged.merge(
    elo_df[["externalId", "diff_elo", "elo_expected_prob_a"]],
    on="externalId", how="left"
)

# --- Head-to-head histórico entre ESTOS DOS jugadores concretos ---
# Para cada partido, miramos SOLO enfrentamientos previos (misma pareja,
# cualquier orden) anteriores a la fecha actual - sin fuga de datos.
#
# SHRINKAGE BAYESIANO: en vez de usar el h2h crudo (ruidoso con pocas
# muestras) o descartarlo con un umbral duro, lo mezclamos con el prior
# que ya nos da el Elo (elo_expected_prob_a) - la mejor estimación
# independiente que tenemos de la probabilidad de que gane A. La fórmula:
#
#   prob_shrunk = (wins_a + PRIOR_WEIGHT * elo_prior) / (n + PRIOR_WEIGHT)
#
# Con 0 enfrentamientos, cae exactamente en el prior del Elo (sin datos
# propios, confiamos del todo en el rating general). Con muchos
# enfrentamientos, el h2h real domina. PRIOR_WEIGHT=4 equivale a decir
# "confío en el Elo tanto como en 4 enfrentamientos reales" - mismo
# principio que la regresión a la media de Bill James en sabermetría de
# béisbol, aplicado aquí al head-to-head.
PRIOR_WEIGHT = 4

def compute_h2h_features(match_df):
    match_df = match_df.sort_values("startDate").reset_index(drop=True)
    h2h_n_meetings = []
    h2h_shrunk_full, h2h_shrunk_last5 = [], []
    history: dict[frozenset, list[str]] = {}  # key -> lista de ganadores en orden cronológico

    for _, row in match_df.iterrows():
        a, b = row["participantAName"], row["participantBName"]
        key = frozenset([a, b])
        past_winners = history.get(key, [])
        elo_prior = row["elo_expected_prob_a"]

        n_meetings = len(past_winners)
        wins_a = sum(1 for w in past_winners if w == a)
        shrunk_full = (wins_a + PRIOR_WEIGHT * elo_prior) / (n_meetings + PRIOR_WEIGHT)

        last5 = past_winners[-5:]
        wins_a_last5 = sum(1 for w in last5 if w == a)
        shrunk_last5 = (wins_a_last5 + PRIOR_WEIGHT * elo_prior) / (len(last5) + PRIOR_WEIGHT)

        h2h_n_meetings.append(n_meetings)
        h2h_shrunk_full.append(shrunk_full)
        h2h_shrunk_last5.append(shrunk_last5)

        winner = a if row["teamAScore"] > row["teamBScore"] else b
        history.setdefault(key, []).append(winner)

    match_df["h2h_n_meetings"] = h2h_n_meetings
    match_df["h2h_shrunk_full"] = h2h_shrunk_full
    match_df["h2h_shrunk_last5"] = h2h_shrunk_last5
    return match_df

merged = compute_h2h_features(merged)

# --- Features de momentum (remontada/derrumbe/cierre de cuarto) ---
# Solo disponibles para partidos donde ya tenemos match_details.csv
# descargado (build_match_details.py). Como esa descarga es progresiva
# (tarda horas), aquí puede faltar para una parte del histórico - en ese
# caso usamos la media general como valor neutro y marcamos con un flag
# si el dato es real o no, para que el modelo pueda aprender a confiar
# más o menos en él.
momentum_path = DATA_DIR / "momentum_features.csv"
if momentum_path.exists():
    momentum = pd.read_csv(momentum_path, parse_dates=["startDate"])
    mom_cols = ["comeback_rate_prev", "blown_lead_rate_prev", "q4_net_avg_prev"]
    defaults = momentum[mom_cols].mean().to_dict()

    mom_a = momentum.rename(columns={"player": "participantAName", **{c: f"{c}_a" for c in mom_cols}})
    mom_b = momentum.rename(columns={"player": "participantBName", **{c: f"{c}_b" for c in mom_cols}})

    merged = merged.merge(
        mom_a[["participantAName", "startDate"] + [f"{c}_a" for c in mom_cols]],
        on=["participantAName", "startDate"], how="left"
    )
    merged = merged.merge(
        mom_b[["participantBName", "startDate"] + [f"{c}_b" for c in mom_cols]],
        on=["participantBName", "startDate"], how="left"
    )

    merged["has_momentum_data"] = (
        merged[f"{mom_cols[0]}_a"].notna() & merged[f"{mom_cols[0]}_b"].notna()
    ).astype(int)

    for c in mom_cols:
        merged[f"{c}_a"] = merged[f"{c}_a"].fillna(defaults[c])
        merged[f"{c}_b"] = merged[f"{c}_b"].fillna(defaults[c])

    merged["diff_comeback_rate"] = merged["comeback_rate_prev_a"] - merged["comeback_rate_prev_b"]
    merged["diff_blown_lead_rate"] = merged["blown_lead_rate_prev_a"] - merged["blown_lead_rate_prev_b"]
    merged["diff_q4_net"] = merged["q4_net_avg_prev_a"] - merged["q4_net_avg_prev_b"]
    HAS_MOMENTUM = True
else:
    print("Aviso: momentum_features.csv no encontrado todavía - features de "
          "remontada/derrumbe/cierre de cuarto NO incluidas en este dataset. "
          "Ejecuta build_match_details.py y build_momentum_features.py primero.")
    HAS_MOMENTUM = False

merged["target_win_a"] = (merged["teamAScore"] > merged["teamBScore"]).astype(int)
merged["target_total"] = merged["teamAScore"] + merged["teamBScore"]

# Placeholder para cuando integremos cuotas reales de una casa de apuestas
merged["market_prob_a"] = np.nan
merged["odds_a"] = np.nan
merged["odds_b"] = np.nan

merged = merged[
    (merged["n_prev_matches_a"] >= 5) & (merged["n_prev_matches_b"] >= 5)
].reset_index(drop=True)

feature_cols = [
    "diff_avg_pts_for", "diff_win_rate", "diff_form5", "diff_form3", "diff_form2",
    "diff_net_rating", "diff_fatigue",
    "diff_elo", "elo_expected_prob_a",
    "h2h_shrunk_full", "h2h_shrunk_last5", "h2h_n_meetings",
    "avg_pts_for_prev_a", "avg_pts_for_prev_b",
    "volatility_prev_a", "volatility_prev_b",
    "hours_since_last_a", "hours_since_last_b",
    "matches_last_3h_a", "matches_last_3h_b",
]
if HAS_MOMENTUM:
    feature_cols += ["diff_comeback_rate", "diff_blown_lead_rate", "diff_q4_net", "has_momentum_data"]

out = merged[["externalId", "startDate", "participantAName", "participantBName"] +
             feature_cols + ["market_prob_a", "odds_a", "odds_b",
                              "target_win_a", "target_total"]].dropna(subset=feature_cols)

out_path = DATA_DIR / "features_real.csv"
out.to_csv(out_path, index=False)
print(f"Dataset de features REAL: {out.shape[0]} partidos, {len(feature_cols)} features.")
print(out[feature_cols].describe().T[["mean", "std", "min", "max"]])
