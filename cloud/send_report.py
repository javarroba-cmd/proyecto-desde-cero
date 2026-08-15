"""
Reconcilia data/signals_log.csv con los resultados reales, genera el mismo
informe + gráfico que la versión local (reconcile_signals.py) y lo envía a
Telegram.

Se ejecuta dentro del workflow "refresh-data" (cada 6 horas), justo después
de haber actualizado matches_real_history.csv - por eso, a diferencia de la
versión local, este script NO vuelve a descargar el histórico él mismo (ya
está fresco por el paso anterior del mismo workflow).
"""
import os
import re
import csv
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TIME_TOLERANCE_MINUTES = 90
_CLV_PLAYER_RE = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


def load_closing_odds() -> dict:
    """Lee data/codere_odds_log.csv (tolerando su mezcla de formatos de 15
    y 20 columnas) y devuelve, para cada (node_id, jugador), la ÚLTIMA
    cuota vista - aproximación de la cuota de CIERRE."""
    path = DATA_DIR / "codere_odds_log.csv"
    if not path.exists():
        return {}

    cols_20 = ["snapshot_time", "node_id", "league_name", "is_live", "team_home", "player_home",
               "team_away", "player_away", "participant_home", "participant_away", "period_name",
               "result_home", "result_away", "start_date_formatted", "game_type_id", "game_type_name",
               "market_line", "outcome_name", "odd", "is_locked"]
    cols_15 = ["snapshot_time", "node_id", "league_name", "participant_home", "participant_away",
               "period_name", "result_home", "result_away", "start_date_formatted", "game_type_id",
               "game_type_name", "market_line", "outcome_name", "odd", "is_locked"]

    latest = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 20:
                cols = cols_20
            elif len(row) == 15:
                cols = cols_15
            else:
                continue
            d = dict(zip(cols, row))
            if d.get("game_type_id") != "97":
                continue
            key = (d["node_id"], d["outcome_name"])
            t = d["snapshot_time"]
            if key not in latest or t > latest[key][0]:
                latest[key] = (t, d["odd"])

    result = {}
    for (node_id, outcome_name), (_, odd) in latest.items():
        m = _CLV_PLAYER_RE.match(outcome_name)
        player = m.group(2).strip() if m else None
        if player is None:
            continue
        try:
            result[(node_id, player)] = float(odd)
        except ValueError:
            continue
    return result


def build_profit_chart(checked_df: pd.DataFrame, out_path: Path) -> Path:
    df = checked_df.copy().sort_values("signal_time")

    missing_mask = df["signal_time"].isna()
    if missing_mask.any():
        fallback_dates = pd.to_datetime(
            df.loc[missing_mask, "start_date_formatted"],
            format="%d/%m/%Y %H:%M:%S", errors="coerce", utc=True,
        )
        df.loc[missing_mask, "signal_time"] = fallback_dates

    df = df.dropna(subset=["signal_time"])
    df["date"] = df["signal_time"].dt.tz_localize(None).dt.floor("D")
    df["cum_profit"] = df["profit"].cumsum()

    daily = []
    running = 0.0
    for day, group in df.groupby("date"):
        day_open = running
        cum_within_day = day_open + group["profit"].cumsum()
        day_close = cum_within_day.iloc[-1]
        day_high = max(day_open, cum_within_day.max())
        day_low = min(day_open, cum_within_day.min())
        daily.append({"date": day, "open": day_open, "close": day_close,
                       "high": day_high, "low": day_low, "n_bets": len(group)})
        running = day_close
    daily_df = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)

    daily_df["ema5"] = daily_df["close"].ewm(span=5, adjust=False).mean()
    daily_df["ema13"] = daily_df["close"].ewm(span=13, adjust=False).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    for _, row in daily_df.iterrows():
        color = "#2ecc71" if row["close"] >= row["open"] else "#e74c3c"
        wick_x = mdates.date2num(row["date"]) + 0.4
        ax1.plot([wick_x, wick_x], [row["low"], row["high"]], color=color, linewidth=1)
        body_bottom = min(row["open"], row["close"])
        body_height = max(abs(row["close"] - row["open"]), 0.05)
        ax1.add_patch(Rectangle((mdates.date2num(row["date"]) + 0.1, body_bottom),
                                 0.6, body_height, color=color))

    ax1.plot(daily_df["date"], daily_df["ema5"], color="#3498db", label="EMA 5", linewidth=1.5)
    ax1.plot(daily_df["date"], daily_df["ema13"], color="#f39c12", label="EMA 13", linewidth=1.5)
    ax1.set_title("DESDE CERO (cloud) · Beneficio acumulado (unidades)")
    ax1.set_ylabel("Unidades")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    min_date, max_date = daily_df["date"].min(), daily_df["date"].max()
    span_days = max((max_date - min_date).days, 1)
    padding = pd.Timedelta(days=max(span_days * 0.15, 0.7))
    ax1.set_xlim(min_date - padding, max_date + padding)

    bar_colors = ["#2ecc71" if r["close"] >= r["open"] else "#e74c3c" for _, r in daily_df.iterrows()]
    bar_x = [mdates.date2num(d) + 0.4 for d in daily_df["date"]]
    ax2.bar(bar_x, daily_df["n_bets"], color=bar_colors, width=0.6)
    ax2.set_ylabel("Señales/día")
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def load_signals_robust() -> pd.DataFrame:
    """Lee data/signals_log.csv tolerando que algunas filas tengan un campo
    de más (odd_change_pct) que la cabecera del archivo no incluye - ver la
    misma explicación detallada en modeling/reconcile_signals.py."""
    path = DATA_DIR / "signals_log.csv"
    cols_old = ["signal_time", "node_id", "player_a", "player_b", "odds_a", "odds_b",
                "model_prob_a", "market_prob_a", "edge", "bet_side", "start_date_formatted",
                "result_checked", "target_win_a", "was_correct"]
    cols_new = ["signal_time", "node_id", "player_a", "player_b", "odds_a", "odds_b",
                "model_prob_a", "market_prob_a", "edge", "bet_side", "odd_change_pct",
                "start_date_formatted", "result_checked", "target_win_a", "was_correct"]

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == len(cols_new):
                rows.append(dict(zip(cols_new, row)))
            elif len(row) == len(cols_old):
                d = dict(zip(cols_old, row))
                d["odd_change_pct"] = ""
                rows.append(d)

    df = pd.DataFrame(rows, columns=cols_new)
    for col in ["odds_a", "odds_b", "model_prob_a", "market_prob_a", "edge",
                "target_win_a", "odd_change_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["result_checked"] = df["result_checked"].astype(str) == "True"
    df["was_correct"] = df["was_correct"].map({"True": True, "False": False, "": None})
    return df


def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    data = resp.json()
    if not data.get("ok"):
        print(f"⚠ Telegram rechazó el mensaje: {data.get('description')}")


def send_telegram_photo(photo_path: Path, caption: str = ""):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                              files={"photo": f}, timeout=30)
    data = resp.json()
    if not data.get("ok"):
        print(f"⚠ Telegram rechazó la foto: {data.get('description')}")


def main():
    signals = load_signals_robust()
    matches = pd.read_csv(DATA_DIR / "matches_real_history.csv")

    signals["signal_time"] = pd.to_datetime(signals["signal_time"], utc=True, errors="coerce", format="ISO8601")
    matches["startDate"] = pd.to_datetime(matches["startDate"], utc=True, errors="coerce", format="ISO8601")

    for col in ["target_win_a", "was_correct"]:
        signals[col] = signals[col].astype(object)

    pending = signals[signals["result_checked"] == False].copy()
    print(f"Señales pendientes de comprobar: {len(pending)}")

    updated = 0
    for idx, sig in pending.iterrows():
        candidates = matches[
            ((matches["participantAName"] == sig["player_a"]) & (matches["participantBName"] == sig["player_b"])) |
            ((matches["participantAName"] == sig["player_b"]) & (matches["participantBName"] == sig["player_a"]))
        ].copy()
        if candidates.empty:
            continue
        candidates["time_diff"] = (candidates["startDate"] - sig["signal_time"]).abs()
        best = candidates.sort_values("time_diff").iloc[0]
        if best["time_diff"] > pd.Timedelta(minutes=TIME_TOLERANCE_MINUTES):
            continue

        if best["participantAName"] == sig["player_a"]:
            target_win_a = int(best["teamAScore"] > best["teamBScore"])
        else:
            target_win_a = int(best["teamBScore"] > best["teamAScore"])
        was_correct = (
            (sig["bet_side"] == "home" and target_win_a == 1) or
            (sig["bet_side"] == "away" and target_win_a == 0)
        )
        signals.loc[idx, "result_checked"] = True
        signals.loc[idx, "target_win_a"] = target_win_a
        signals.loc[idx, "was_correct"] = was_correct
        updated += 1

    signals.to_csv(DATA_DIR / "signals_log.csv", index=False)
    print(f"Reconciliadas {updated} señales nuevas.")

    checked = signals[signals["result_checked"] == True]
    if len(checked) == 0:
        print("Todavía no hay señales reconciliadas con resultado.")
        return

    checked = checked.copy()

    def get_odds(row):
        return row["odds_a"] if row["bet_side"] == "home" else row["odds_b"]

    def profit(row):
        return (get_odds(row) - 1) if row["was_correct"] else -1

    checked["odds_used"] = checked.apply(get_odds, axis=1)
    checked["profit"] = checked.apply(profit, axis=1)
    checked["stake"] = 1.0

    def get_picked_player(row):
        return row["player_a"] if row["bet_side"] == "home" else row["player_b"]
    checked["picked_player"] = checked.apply(get_picked_player, axis=1)

    closing_odds_map = load_closing_odds()
    checked["closing_odds"] = checked.apply(
        lambda r: closing_odds_map.get((str(r["node_id"]), r["picked_player"])), axis=1
    )
    clv_available = checked.dropna(subset=["closing_odds"]).copy()
    if len(clv_available) > 0:
        clv_available["clv_pct"] = (
            (clv_available["odds_used"] - clv_available["closing_odds"]) / clv_available["closing_odds"] * 100
        )
        clv_mean = clv_available["clv_pct"].mean()
        clv_positive_pct = (clv_available["clv_pct"] > 0).mean() * 100
        n_clv = len(clv_available)
    else:
        clv_mean, clv_positive_pct, n_clv = None, None, 0

    n_bets = len(checked)
    n_wins = int(checked["was_correct"].sum())
    n_losses = n_bets - n_wins
    win_rate = n_wins / n_bets * 100
    total_staked = checked["stake"].sum()
    units_won = checked["profit"].sum()
    avg_profit = checked["profit"].mean()
    avg_odds = checked["odds_used"].mean()
    min_odds = checked["odds_used"].min()
    max_odds = checked["odds_used"].max()
    roi_pct = units_won / total_staked * 100

    fecha_inicio = checked["signal_time"].min()
    fecha_fin = checked["signal_time"].max()
    dias = max((fecha_fin - fecha_inicio).total_seconds() / 86400, 1) if pd.notna(fecha_inicio) else 1

    señales_por_dia = len(signals) / max(dias, 1)

    report_text = (
        f"🆕 <b>DESDE CERO (cloud)</b> · Informe de señales\n"
        f"────────────────────\n"
        f"Total # apuestas: {n_bets}\n"
        f"Rango cuotas: {min_odds:.2f}-{max_odds:.2f}\n"
        f"Cuota media: {avg_odds:.2f}\n"
        f"Aciertos: {n_wins}\n"
        f"Fallos: {n_losses}\n"
        f"Porcentaje de Aciertos (%): {win_rate:.2f}\n"
        f"Unidades ganadas: {units_won:.2f}\n"
        f"Beneficio medio: {avg_profit:.4f}\n"
        f"ROI (%): {roi_pct:.2f}\n"
        + (f"CLV medio (%): {clv_mean:+.2f} (n={n_clv})\n% señales con CLV positivo: {clv_positive_pct:.1f}%\n" if n_clv > 0 else "")
        + f"Sugerencias x día (aprox.): {señales_por_dia:.1f}"
    )
    if n_bets < 100:
        report_text += "\n\n⚠ Muestra pequeña (&lt;100 señales), no saques conclusiones todavía."

    print(report_text.replace("<b>", "").replace("</b>", ""))

    chart_path = build_profit_chart(checked, DATA_DIR / "profit_chart_cloud.png")

    send_telegram_message(report_text)
    send_telegram_photo(chart_path, caption="🆕 DESDE CERO (cloud) · Rentabilidad acumulada")


if __name__ == "__main__":
    main()
