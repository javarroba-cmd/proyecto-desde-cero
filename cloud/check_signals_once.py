"""
Versión de UN SOLO DISPARO (no bucle infinito) del sistema de señales,
pensada para ejecutarse en GitHub Actions cada 1-2 minutos (disparada por
cron-job.org). Cada ejecución:

1. Carga el modelo YA ENTRENADO (data/model.joblib) - no reentrena.
2. Consulta la liga GG League en Codere UNA vez.
3. Registra las cuotas vistas (como hacía poll_odds_live.py).
4. Para partidos pre-partido con edge>umbral, mira el HISTORIAL de cuotas
   de ese partido concreto (codere_odds_log.csv) y calcula cuánto se ha
   movido la cuota del lado elegido desde la primera vez que se vio hasta
   ahora. Solo genera señal si, ADEMÁS del edge, la cuota NO ha empeorado
   (el mercado no se ha alejado de ese pick) - validado con datos reales:
   con la cuota mejorando, 57.0% de acierto (IC95% [49.4, 64.6]); con la
   cuota empeorando, solo 21.3% (IC95% [14.7, 28.7]), sobre 308 señales.
   Si un partido aún no tiene suficiente historial de cuotas (solo lo
   hemos visto una vez), se pospone la decisión al siguiente ciclo, no se
   descarta - evitando repetir señales ya enviadas (signals_log.csv como
   memoria persistente entre ejecuciones).
"""
import os
import re
import csv
import sys
import joblib
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scraping"))
sys.path.append(str(ROOT / "modeling"))

from codere_client import get_gg_league_events, flatten_event_odds
from live_features import LiveFeatureEngine

DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "model.joblib"

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

EDGE_THRESHOLD = 0.03
MIN_MINUTES_BEFORE_START = 0  # ajusta libremente, igual que en signal_generator.py
MADRID_TZ = ZoneInfo("Europe/Madrid")

# REACTIVADO (14/08/2026) con la nueva lógica de movimiento de cuota. Sigue
# siendo shadow mode (no apuesta dinero real), así que reactivar para
# probar esta versión mejorada en vivo es coherente con la metodología del
# proyecto (hipótesis -> shadow -> decisión). Si quieres volver a pausar
# la generación de señales sin tocar nada más, pon esto en True.
SIGNAL_GENERATION_PAUSED = False

MIN_SNAPSHOTS_REQUIRED = 2  # nº mínimo de capturas de cuota antes de poder decidir
MIN_OBSERVATION_MINUTES = 15  # tiempo MÍNIMO desde la 1ª captura hasta ahora,
# para que la comparación se parezca más a "movimiento real hacia el cierre"
# y menos a ruido de los primeros minutos (validado tras ver que el filtro
# con solo 2 capturas rendía peor en vivo que en el análisis retrospectivo,
# que comparaba contra la cuota de cierre real).
CODERE_DATE_RE = re.compile(r"/Date\((\d+)\)/")

ODDS_FIELDNAMES = [
    "snapshot_time", "node_id", "league_name", "is_live",
    "team_home", "player_home", "team_away", "player_away",
    "participant_home", "participant_away", "period_name",
    "result_home", "result_away", "start_date_formatted",
    "game_type_id", "game_type_name", "market_line", "outcome_name",
    "odd", "is_locked",
]
# Columnas "antiguas" (15) que puede haber en la parte más vieja del CSV,
# de antes de que añadiéramos is_live/team_home/etc. Leemos ambos formatos
# para no perder historial al calcular movimientos.
ODDS_FIELDNAMES_LEGACY = [
    "snapshot_time", "node_id", "league_name", "participant_home", "participant_away",
    "period_name", "result_home", "result_away", "start_date_formatted", "game_type_id",
    "game_type_name", "market_line", "outcome_name", "odd", "is_locked",
]

SIGNAL_FIELDNAMES = [
    "signal_time", "node_id", "player_a", "player_b",
    "odds_a", "odds_b", "model_prob_a", "market_prob_a", "edge",
    "bet_side", "odd_change_pct", "start_date_formatted",
    "result_checked", "target_win_a", "was_correct",
]


def parse_codere_date(value):
    if not value:
        return None
    m = CODERE_DATE_RE.match(value)
    if not m:
        return None
    ms = int(m.group(1))
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(MADRID_TZ)


def load_odds_history() -> dict:
    """Lee codere_odds_log.csv (tolerando la mezcla de formatos de 15 y 20
    columnas que puede tener el archivo) y devuelve, para cada (node_id,
    outcome_name), el timestamp de la PRIMERA captura y la lista de cuotas
    vistas en orden cronológico.
    Devuelve un dict: {(node_id, outcome_name): (primer_timestamp, [odd1, odd2, ...])}"""
    path = DATA_DIR / "codere_odds_log.csv"
    history = {}
    if not path.exists():
        return history

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # saltar cabecera (puede estar desactualizada)
        for row in reader:
            if len(row) == 20:
                cols = ODDS_FIELDNAMES
            elif len(row) == 15:
                cols = ODDS_FIELDNAMES_LEGACY
            else:
                continue  # fila corrupta (nombre con coma sin escapar), se ignora
            d = dict(zip(cols, row))
            if d.get("game_type_id") != "97":
                continue
            key = (d["node_id"], d["outcome_name"])
            history.setdefault(key, []).append((d["snapshot_time"], d["odd"]))

    # ordenar cronológicamente y quedarnos con el primer timestamp + lista de cuotas
    result = {}
    for key, entries in history.items():
        entries.sort(key=lambda e: e[0])
        try:
            odds_list = [float(o) for _, o in entries]
        except ValueError:
            continue
        first_timestamp = entries[0][0]
        result[key] = (first_timestamp, odds_list)
    return result


def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        data = resp.json()
        if not data.get("ok"):
            print(f"Telegram rechazó el mensaje: {data.get('description')}")
    except Exception as e:
        print(f"Error enviando a Telegram: {e}")


def append_csv(path: Path, fieldnames: list, rows: list):
    if not rows:
        return
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def load_already_signaled() -> set[str]:
    """Memoria PERSISTENTE de señales ya enviadas (lee el propio CSV), en
    vez de una lista en memoria que se perdía cada vez que el proceso se
    reiniciaba (causa del bug de señales duplicadas que tuvimos antes)."""
    path = DATA_DIR / "signals_log.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["node_id"])
    return set(df["node_id"].astype(str))


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    engine = LiveFeatureEngine()
    already_signaled = load_already_signaled()
    odds_history = load_odds_history()

    events = get_gg_league_events()
    snapshot_time = datetime.now(timezone.utc).isoformat()

    odds_rows = []
    signal_rows = []
    n_pending_movement = 0

    for event in events:
        odds_rows.extend(flatten_event_odds(event, snapshot_time))

        if SIGNAL_GENERATION_PAUSED:
            continue  # seguimos registrando cuotas (arriba), pero no generamos señales

        node_id = str(event.get("NodeId"))
        full_home = event.get("ParticipantHome", "")
        full_away = event.get("ParticipantAway", "")
        m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", full_home)
        player_home = m.group(2).strip() if m else None
        m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", full_away)
        player_away = m.group(2).strip() if m else None
        if not player_home or not player_away:
            continue

        if event.get("isLive"):
            continue  # solo señalamos partidos pre-partido, con antelación real
        if node_id in already_signaled:
            continue

        moneyline = None
        for game in event.get("Games", []):
            results = game.get("Results", [])
            if results and results[0].get("GameTypeId") == 97:
                moneyline = results
                break
        if not moneyline or len(moneyline) != 2:
            continue
        if any(r.get("Locked") for r in moneyline):
            continue

        odds_home = moneyline[0].get("Odd")
        odds_away = moneyline[1].get("Odd")
        if not odds_home or not odds_away:
            continue

        match_start = parse_codere_date(event.get("StartDate"))
        if match_start is None:
            continue
        minutes_to_start = (match_start - datetime.now(MADRID_TZ)).total_seconds() / 60
        if minutes_to_start < MIN_MINUTES_BEFORE_START:
            continue

        feats = engine.get_features(player_home, player_away)
        if feats is None:
            continue

        feat_df = pd.DataFrame([feats])[feature_cols]
        model_prob_home = model.predict_proba(feat_df)[0, 1]
        market_prob_home = (1 / odds_home) / (1 / odds_home + 1 / odds_away)
        edge_home = model_prob_home - market_prob_home
        edge_away = (1 - model_prob_home) - (1 - market_prob_home)

        bet_side, chosen_edge = None, 0.0
        if edge_home > EDGE_THRESHOLD:
            bet_side, chosen_edge = "home", edge_home
        elif edge_away > EDGE_THRESHOLD:
            bet_side, chosen_edge = "away", edge_away
        if bet_side is None:
            continue

        # FILTRO DE MOVIMIENTO DE CUOTA (añadido 14/08/2026, ajustado
        # 15/08/2026 tras ver que con solo 2 capturas rendía peor en vivo
        # -6.1% ROI- que en el análisis retrospectivo contra cuota de
        # cierre -que sugería mucho más-). Ahora exigimos además un tiempo
        # MÍNIMO de observación, no solo un mínimo de capturas, para que
        # la comparación se parezca más a "movimiento real hacia el
        # cierre" y menos a ruido de los primeros minutos.
        full_pick = full_home if bet_side == "home" else full_away
        odds_pick = odds_home if bet_side == "home" else odds_away
        key = (node_id, full_pick)
        entry = odds_history.get(key)

        if entry is None:
            n_pending_movement += 1
            continue
        first_timestamp_str, history = entry

        if len(history) < MIN_SNAPSHOTS_REQUIRED:
            n_pending_movement += 1
            continue

        first_seen_dt = datetime.fromisoformat(first_timestamp_str)
        if first_seen_dt.tzinfo is None:
            first_seen_dt = first_seen_dt.replace(tzinfo=timezone.utc)
        minutes_observed = (datetime.now(timezone.utc) - first_seen_dt).total_seconds() / 60
        if minutes_observed < MIN_OBSERVATION_MINUTES:
            # Aún no ha pasado suficiente tiempo desde que vimos este
            # partido por primera vez - esperamos al siguiente ciclo, NO
            # lo descartamos (no se añade a already_signaled).
            n_pending_movement += 1
            continue

        first_odd = history[0]
        odd_change_pct = (odds_pick - first_odd) / first_odd * 100
        if odd_change_pct > 0:
            # La cuota ha empeorado desde que la vimos por primera vez
            # (el mercado se ha alejado de este pick) - descartamos esta
            # señal, aunque el edge parezca bueno.
            already_signaled.add(node_id)  # no reintentar este partido
            continue

        already_signaled.add(node_id)
        fecha_str = match_start.strftime("%Y-%m-%d")
        hora_str = match_start.strftime("%H:%M:%S")
        torneo = event.get("LeagueName", "eBasketball")

        msg = (
            f"🆕 <b>PROYECTO DESDE CERO</b> · señal eBasket (cloud)\n"
            f"────────────────────\n"
            f"👉 <b>Pick</b>: {full_pick} @ {odds_pick}\n"
            f"🏀 <b>Partido</b>: {full_home}@{odds_home} vs {full_away}@{odds_away}\n"
            f"🏆 <b>Torneo</b>: {torneo}\n"
            f"📅 <b>Fecha</b>: {fecha_str}\n"
            f"⏰ <b>Hora</b>: {hora_str} (España)\n"
            f"⏳ <b>Antelación</b>: {minutes_to_start:.0f} min antes del inicio\n"
            f"📉 <b>Movimiento cuota</b>: {odd_change_pct:+.1f}% (mejorando)\n"
            f"────────────────────\n"
            f"Prob. modelo: {model_prob_home*100 if bet_side=='home' else (1-model_prob_home)*100:.1f}% | "
            f"Prob. mercado: {market_prob_home*100 if bet_side=='home' else (1-market_prob_home)*100:.1f}%\n"
            f"Edge: {chosen_edge*100:.1f}%"
        )
        print(msg.replace("<b>", "").replace("</b>", ""))
        send_telegram_message(msg)

        signal_rows.append({
            "signal_time": datetime.now(timezone.utc).isoformat(),
            "node_id": node_id, "player_a": player_home, "player_b": player_away,
            "odds_a": odds_home, "odds_b": odds_away,
            "model_prob_a": model_prob_home, "market_prob_a": market_prob_home,
            "edge": chosen_edge, "bet_side": bet_side, "odd_change_pct": odd_change_pct,
            "start_date_formatted": event.get("StartDateFormatted"),
            "result_checked": False, "target_win_a": "", "was_correct": "",
        })

    append_csv(DATA_DIR / "codere_odds_log.csv", ODDS_FIELDNAMES, odds_rows)
    append_csv(DATA_DIR / "signals_log.csv", SIGNAL_FIELDNAMES, signal_rows)

    print(f"\nPartidos vistos: {len(events)} | filas de cuotas guardadas: {len(odds_rows)} | "
          f"señales nuevas: {len(signal_rows)} | pendientes de más historial: {n_pending_movement}")
    if SIGNAL_GENERATION_PAUSED:
        print("⏸ Generación de señales PAUSADA (SIGNAL_GENERATION_PAUSED=True) - "
              "solo se están registrando cuotas de mercado.")


if __name__ == "__main__":
    main()
