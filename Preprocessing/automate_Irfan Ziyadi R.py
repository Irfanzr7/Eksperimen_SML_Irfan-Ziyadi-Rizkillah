
import os
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(input_path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan: {input_path}")
    return pd.read_csv(input_path)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Melakukan preprocessing pada dataset.

    Fitur utama perubahan:
    - Memeriksa keberadaan kolom yang dibutuhkan.
    - Mengonversi kolom fitur ke numerik bila memungkinkan, dan mengisi NA.
    - Mengabaikan kolom non-numerik dengan peringatan (agar StandardScaler tidak gagal).
    """
    # 1) Drop duplicates (sesuai notebook)
    df = df.drop_duplicates()

    # Pastikan kolom target & tanggal ada
    required_cols = {"weather", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset tidak memiliki kolom yang diperlukan: {sorted(missing)}")

    # 2) Pemilihan fitur dan target (sesuai notebook)
    X = df.drop(columns=["weather", "date"]) if len(df.columns) > 2 else pd.DataFrame()
    y = df["weather"]

    # 3) Encoding Label target (sesuai notebook)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    label_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )

    # 4) Normalisasi/standardisasi fitur numerik (sesuai notebook)
    if X.empty:
        logging.warning("Tidak ada fitur numerik ditemukan setelah menghapus 'weather' dan 'date'. Akan menghasilkan dataset hanya berisi target.")
        df_processed = pd.DataFrame({"weather": y_encoded})
        return df_processed, label_mapping

    # Coba konversi semua kolom fitur ke numerik (coerce non-numeric to NaN)
    X_numeric = X.apply(pd.to_numeric, errors="coerce")

    # Jika ada kolom yang seluruhnya NaN setelah konversi, drop dan peringatkan
    all_na_cols = [c for c in X_numeric.columns if X_numeric[c].isna().all()]
    if all_na_cols:
        logging.warning(f"Menghapus kolom non-numerik yang tidak dapat dikonversi: {all_na_cols}")
        X_numeric = X_numeric.drop(columns=all_na_cols)

    if X_numeric.shape[1] == 0:
        raise ValueError("Tidak ada kolom numerik yang tersisa untuk distandarisasi.")

    # Isi nilai NaN dengan rata-rata kolom
    X_numeric = X_numeric.fillna(X_numeric.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    X_scaled = pd.DataFrame(X_scaled, columns=X_numeric.columns)

    # 5) Dataset final (sesuai notebook)
    df_processed = X_scaled.copy()
    df_processed["weather"] = y_encoded

    return df_processed, label_mapping


def save_processed(df_processed: pd.DataFrame, output_dir: str) -> str:
    """Menyimpan dataset hasil preprocessing ke CSV."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "seattle_weather_processed.csv")
    df_processed.to_csv(output_path, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated preprocessing for Seattle Weather dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path dataset raw (CSV). Jika tidak disediakan, script akan mencari file di ../Weather_datasets_raw/seattle-weather.csv relatif terhadap file ini.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weather_preprocessing",
        help="Folder output dataset processed. Default: weather_preprocessing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Tentukan path input default relatif terhadap lokasi file ini
    if not args.input:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        default_input = os.path.join(base_dir, "Weather_datasets_raw", "seattle-weather.csv")
        args.input = default_input

    try:
        df = load_data(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        df_processed, label_mapping = preprocess_data(df)
    except Exception as e:
        print(f"Preprocessing gagal: {e}")
        return

    try:
        saved_path = save_processed(df_processed, args.output_dir)
    except Exception as e:
        print(f"Gagal menyimpan file: {e}")
        return

    # Print info untuk memudahkan pengecekan reviewer
    print("Preprocessing selesai")
    print(f"Input      : {args.input}")
    print(f"Output file: {saved_path}")
    print("Label mapping (weather -> encoded):")
    for k, v in label_mapping.items():
        print(f"  - {k}: {v}")
    print("\n Preview output:")
    print(df_processed.head())


if __name__ == "__main__":
    main()
