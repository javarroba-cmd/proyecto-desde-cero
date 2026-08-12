"""
Rating Elo dinámico por jugador para eNBA2K.

Por qué esto puede mejorar el modelo: en la validación anterior, 'diff_form5'
(forma de los últimos 5 partidos) fue con diferencia la variable más
importante — mucho más que el promedio histórico completo. Eso sugiere que
el "nivel actual" de un jugador fluctúa y un promedio simple no lo captura
bien. Un rating Elo es exactamente la herramienta diseñada para esto: sube y
baja partido a partido según si ganas/pierdes y contra quién, dando más peso
implícito a la forma reciente sin necesidad de fijar una ventana arbitraria
de "últimos N partidos".

Referencia: mismo principio que el rating FIDE de ajedrez o el Elo de 538
para NBA/NFL.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "matches_real_history.csv", parse_dates=["startDate"])
df = df.sort_values("startDate").reset_index(drop=True)

INITIAL_ELO = 1500
K_FACTOR = 24          # sensibilidad del ajuste por partido (estándar: 16-32)
MARGIN_MULTIPLIER = True  # si True, pondera el K según la diferencia de puntos (no solo win/loss)

elo_ratings: dict[str, float] = {}
elo_before_a = []
elo_before_b = []

def get_elo(player: str) -> float:
    return elo_ratings.get(player, INITIAL_ELO)

def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

for _, row in df.iterrows():
    a, b = row["participantAName"], row["participantBName"]
    elo_a, elo_b = get_elo(a), get_elo(b)
    elo_before_a.append(elo_a)
    elo_before_b.append(elo_b)

    actual_a = 1.0 if row["teamAScore"] > row["teamBScore"] else 0.0
    exp_a = expected_score(elo_a, elo_b)

    # Multiplicador por margen de victoria: ganar por mucho pesa más que ganar
    # por poco, señal extra que un Elo binario clásico ignora.
    if MARGIN_MULTIPLIER:
        margin = abs(row["teamAScore"] - row["teamBScore"])
        margin_mult = np.log(margin + 1) / np.log(15)  # normalizado, aprox 1.0 en margen típico
        margin_mult = np.clip(margin_mult, 0.5, 2.0)
    else:
        margin_mult = 1.0

    delta = K_FACTOR * margin_mult * (actual_a - exp_a)
    elo_ratings[a] = elo_a + delta
    elo_ratings[b] = elo_b - delta

df["elo_a_before"] = elo_before_a
df["elo_b_before"] = elo_before_b
df["diff_elo"] = df["elo_a_before"] - df["elo_b_before"]
df["elo_expected_prob_a"] = expected_score(df["elo_a_before"], df["elo_b_before"])

out_path = DATA_DIR / "matches_with_elo.csv"
df.to_csv(out_path, index=False)

# --- Vistazo rápido: ranking final de los jugadores con más partidos ---
final_ratings = pd.Series(elo_ratings).sort_values(ascending=False)
n_matches = pd.concat([
    df["participantAName"], df["participantBName"]
]).value_counts()

ranking = pd.DataFrame({"elo": final_ratings, "n_matches": n_matches}).dropna()
ranking = ranking[ranking["n_matches"] >= 20].sort_values("elo", ascending=False)

print(f"Guardado: {out_path}")
print(f"\nTop 10 jugadores por Elo (mínimo 20 partidos jugados):")
print(ranking.head(10).to_string())
print(f"\nBottom 5:")
print(ranking.tail(5).to_string())
