"""
Entrena el modelo con features_real.csv y lo guarda en data/model.joblib.

Se ejecuta dentro del workflow "refresh-data" (cada 6 horas), NO en cada
comprobación de señales - así el check de señales es rápido (solo carga
el modelo ya entrenado, no reentrena cada vez).
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NON_FEATURE_COLS = {
    "externalId", "startDate", "participantAName", "participantBName",
    "market_prob_a", "odds_a", "odds_b", "target_win_a", "target_total",
}


def main():
    df = pd.read_csv(DATA_DIR / "features_real.csv", parse_dates=["startDate"])
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    base_model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(df[feature_cols], df["target_win_a"])

    joblib.dump({"model": model, "feature_cols": feature_cols}, DATA_DIR / "model.joblib")
    print(f"Modelo entrenado con {len(df)} partidos y guardado en data/model.joblib")
    print(f"Features usadas ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    main()
