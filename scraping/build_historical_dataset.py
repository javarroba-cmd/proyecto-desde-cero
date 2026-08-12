"""
Construye el dataset histórico completo de partidos de H2H GG League (eNBA2K),
iterando día a día sobre el endpoint /schedule/nba desde la fecha más antigua
disponible hasta hoy.

Se puede ejecutar directamente (python build_historical_dataset.py) o
importar su función main() desde otro script (ver reconcile_signals.py, que
lo llama automáticamente antes de generar el informe).
"""
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from hudstats_client import get_schedule_for_date

# Carpeta data/ hermana de la carpeta donde está este script (scraping/)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Ajusta esta fecha si al ejecutar ves que hay más histórico disponible del
# que detectamos en la inspección manual (~2026-06-11). Si sale vacío antes,
# el script simplemente guardará menos partidos de los esperados ese día.
START_DATE = date(2026, 6, 11)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    end_date = date.today()

    all_matches = []
    current = START_DATE

    print(f"Descargando partidos desde {START_DATE} hasta {end_date}...")

    while current <= end_date:
        date_str = current.isoformat()
        try:
            matches = get_schedule_for_date(date_str)
            n_ended = sum(1 for m in matches if m.get("matchStatus") == "MATCH_ENDED")
            print(f"  {date_str}: {len(matches)} partidos ({n_ended} finalizados)")
            all_matches.extend(matches)
        except Exception as e:
            print(f"  {date_str}: ERROR -> {e}")
        current += timedelta(days=1)

    df = pd.DataFrame(all_matches)

    # Nos quedamos solo con partidos realmente finalizados y no cancelados,
    # que son los únicos útiles para entrenar el modelo.
    df_clean = df[
        (df["matchStatus"] == "MATCH_ENDED") & (~df["isCancelled"])
    ].drop_duplicates(subset="externalId").reset_index(drop=True) if len(df) > 0 else df

    if len(df_clean) > 0:
        df_clean["startDate"] = pd.to_datetime(df_clean["startDate"])

    out_path = DATA_DIR / "matches_real_history.csv"

    # FUSIÓN, no sobrescritura: si ya existía un CSV de antes (de una
    # ejecución previa), lo combinamos con lo nuevo en vez de reemplazarlo.
    # Así, si esta vez algunas fechas fallan (ej. por un bloqueo 429 de la
    # API), NUNCA se pierde cobertura que ya tenías guardada de antes.
    if out_path.exists():
        old_df = pd.read_csv(out_path, parse_dates=["startDate"])
        old_df["externalId"] = old_df["externalId"].astype(str)
        if len(df_clean) > 0:
            df_clean["externalId"] = df_clean["externalId"].astype(str)
            combined = pd.concat([old_df, df_clean], ignore_index=True)
        else:
            combined = old_df
        combined = combined.drop_duplicates(subset="externalId", keep="last")
        print(f"\nFusionado con histórico previo: {len(old_df)} partidos antiguos + "
              f"{len(df_clean)} nuevos -> {len(combined)} únicos tras fusión.")
    else:
        combined = df_clean

    combined = combined.sort_values("startDate").reset_index(drop=True)
    combined.to_csv(out_path, index=False)

    print(f"\nGuardado: {len(combined)} partidos únicos finalizados en {out_path}")
    print(combined[["startDate", "participantAName", "participantBName",
                     "teamAScore", "teamBScore"]].head())
    return combined


if __name__ == "__main__":
    main()
