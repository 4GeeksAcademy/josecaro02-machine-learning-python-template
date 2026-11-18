"""Inferencia de sentimiento sobre las reseñas de WeLoveReviews.

Lee data/raw/reviews.csv, predice 1-5 estrellas con un modelo preentrenado de
Hugging Face, lo mapea a bandas de sentimiento y escribe
data/processed/reviews_with_sentiment.csv.

Uso:
    python src/app.py
"""
from pathlib import Path

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
# Commit exacto del modelo en Hugging Face (versión fijada / pinned).
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"

BATCH_SIZE = 32


def load_model():
    """Carga el modelo UNA sola vez (pesos se descargan a ~/.cache/huggingface)."""
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        revision=MODEL_REVISION,
    )


def star_to_band(stars: int) -> str:
    """1-2 estrellas -> Negativo, 3 -> Neutral, 4-5 -> Positivo."""
    if stars <= 2:
        return "Negativo"
    if stars == 3:
        return "Neutral"
    return "Positivo"


def predict(df: pd.DataFrame, classifier) -> pd.DataFrame:
    """Añade pred_stars, pred_score y sentiment_band a cada reseña."""
    results = classifier(
        df["review_text"].tolist(),
        batch_size=BATCH_SIZE,
        truncation=True,
    )
    out = df.copy()
    # La etiqueta viene como "4 stars" -> nos quedamos con el número.
    out["pred_stars"] = [int(r["label"].split()[0]) for r in results]
    out["pred_score"] = [round(r["score"], 4) for r in results]
    out["sentiment_band"] = out["pred_stars"].map(star_to_band)
    return out


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    classifier = load_model()
    enriched = predict(df, classifier)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"{len(enriched)} reseñas procesadas -> {OUTPUT_PATH}")
    print(enriched["sentiment_band"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
