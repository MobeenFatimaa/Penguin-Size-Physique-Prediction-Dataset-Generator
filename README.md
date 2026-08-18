# Penguin Size & Physique Prediction Dataset Generator
<p align="center">
  <img src="penguin_dataset_preview.jpg" alt="Penguin Size & Physique Prediction Dataset" width="900">
</p>

A high-performance synthetic data generation pipeline built with **Python**, **Polars**, and **NumPy**. This tool generates a realistic, correlated **100,000-record dataset** (22 features) modeling physical morphometrics, health scores, and environmental traits across three Antarctic penguin species (*Adelie*, *Chinstrap*, *Gentoo*).

---

## Key Features

* **High Performance:** Generates 100,000 complex records in under **0.5 seconds** using vectorized NumPy and Polars operations.
* **Realistic Biological Modeling:** Incorporates species-specific size scaling, sexual dimorphism, age-based growth stages, and environmental dependencies.
* **ML-Ready Engineering:** Includes engineered features like `body_mass_index`, `flipper_body_ratio`, and `bill_area_index`.
* **Real-World Imperfections:** Intentionally injects ~2% missing values across physical traits for imputation practice, along with realistic morphological outliers (~0.8%).
* **Multi-Task ML Support:** Built for multiclass classification (`size_category`), continuous regression (`body_mass_g`), and clustering tasks.

---

## Generated Dataset Schema (22 Features)

| Column Name                 | Data Type   | Description                                        |
| :-------------------------- | :---------- | :------------------------------------------------- |
| `penguin_id`                | Integer     | Unique identifier for each record                  |
| `species`                   | Categorical | *Adelie*, *Chinstrap*, or *Gentoo*                 |
| `island`                    | Categorical | *Biscoe*, *Dream*, or *Torgersen*                  |
| `sex`                       | Categorical | *Male* or *Female*                                 |
| `age_years`                 | Float       | Age in years (0.5 to 15.0)                         |
| `body_length_cm`            | Float       | Total body length (cm)                             |
| `bill_length_mm`            | Float       | Culmen length (mm) (~2% missing)                   |
| `bill_depth_mm`             | Float       | Culmen depth (mm) (~2% missing)                    |
| `flipper_length_mm`         | Float       | Flipper length (mm) (~2% missing)                  |
| `body_mass_g`               | Float       | Body mass in grams (~2% missing)                   |
| `body_mass_index`           | Float       | Synthetic BMI indicator (mass / length²)           |
| `flipper_body_ratio`        | Float       | Ratio of flipper length to body length             |
| `bill_area_index`           | Float       | Calculated bill area (length × depth)              |
| `health_score`              | Float       | Health condition indicator (0–100) (~1.9% missing) |
| `growth_stage`              | Categorical | *Juvenile*, *Subadult*, or *Adult*                 |
| `size_score`                | Float       | Composite physical size metric (0–100)             |
| `size_category`             | Categorical | **Primary Target**: *Small*, *Medium*, *Large*     |
| `weight_category`           | Categorical | *Light*, *Normal*, or *Heavy*                      |
| `environment_temperature_c` | Float       | Ambient temperature (°C)                           |
| `food_availability`         | Float       | Regional prey availability score (0–100)           |
| `activity_level`            | Float       | Daily activity index (0–100)                       |
| `survival_condition_score`  | Float       | Composite condition score (0–100)                  |

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MobeenFatimaa/Penguin-Size-Physique-Prediction-Dataset-Generator.git
cd Penguin-Size-Physique-Prediction-Dataset-Generator
pip install polars numpy pandas scikit-learn lightgbm matplotlib seaborn
```

### 2. Generate Dataset

Run the generator script to produce `penguin_size_dataset.csv`:

```bash
python generate_dataset.py
```

### 3. Verify Dataset Integrity

Run the verification script to output dataset statistics, missing value ratios, and target class distributions:

```bash
python verify_dataset.py
```

---

## Kaggle Dataset

The generated dataset is available on Kaggle:

**Kaggle:** https://www.kaggle.com/datasets/mobeenfatimah/penguin-size-and-physique-prediction-dataset

---

## License

Distributed under the Apache-2.0 License.
