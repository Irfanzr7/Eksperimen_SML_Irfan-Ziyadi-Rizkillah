# Eksperimen SML — Preprocessing Seattle Weather

Ringkasan singkat cara menjalankan preprocessing dan notebook pada repositori ini.

- Script preprocessing: `Preprocessing/automate_Irfan_Ziyadi_R.py`
- Dataset sumber: `Weather_datasets_raw/seattle-weather.csv`
- Hasil preprocessing: `weather_preprocessing/seattle_weather_processed.csv`

## Persiapan

1. Buat virtual environment (opsional) dan aktifkan.

2. Install dependency:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan preprocessing (CLI)

Jalankan dari root repo:

```bash
python Preprocessing/automate_Irfan_Ziyadi_R.py --input Weather_datasets_raw/seattle-weather.csv --output_dir weather_preprocessing
```

Setelah selesai, file `weather_preprocessing/seattle_weather_processed.csv` akan dibuat.

## Notebook

Notebook contoh: `Preprocessing/Eksperimen_Irfan Ziyadi Rizkillah.ipynb`.
Anda dapat menjalankannya menggunakan Jupyter Lab/Notebook atau VS Code dengan ekstensi notebook.
