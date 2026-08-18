#  Penguin Physical Characteristics & Size Prediction Dataset — Data Dictionary

This data dictionary provides a comprehensive reference for all **22 features** included in the `penguin_size_dataset.csv` generated file. The dataset models realistic morphological attributes, health metrics, and environmental conditions across three Antarctic penguin species (*Adelie*, *Chinstrap*, and *Gentoo*).

---

##  Overview Table

| # | Column Name | Data Type | Nullable | Domain / Range | Target / Feature Type | Description |
|---|---|---|---|---|---|---|
| 1 | `penguin_id` | Integer | No | `1` – `100,000` | Identifier | Unique sequential identifier for each observation. |
| 2 | `species` | Categorical | No | `Adelie`, `Chinstrap`, `Gentoo` | Categorical Feature | Species of the penguin with distinct physical distribution tendencies. |
| 3 | `island` | Categorical | No | `Biscoe`, `Dream`, `Torgersen` | Categorical Feature | Island habitat where the penguin observation was recorded. |
| 4 | `sex` | Categorical | No | `Male`, `Female` | Categorical Feature | Biological sex, incorporating moderate sexual dimorphism effects. |
| 5 | `age_years` | Numerical (Float) | No | `0.5` – `15.0` | Continuous Feature | Estimated age in years, following a gamma-shaped distribution. |
| 6 | `body_length_cm` | Numerical (Float) | No | `35.0` – `85.0` | Physical Feature | Total body length measured from beak tip to tail in centimeters. |
| 7 | `bill_length_mm` | Numerical (Float) | **Yes (~2%)** | `30.0` – `65.0` | Physical Feature | Culmen length measured in millimeters. |
| 8 | `bill_depth_mm` | Numerical (Float) | **Yes (~2%)** | `12.0` – `25.0` | Physical Feature | Culmen depth measured in millimeters. |
| 9 | `flipper_length_mm` | Numerical (Float) | **Yes (~2%)** | `150.0` – `240.0` | Physical Feature | Total length of the flipper in millimeters. |
| 10 | `body_mass_g` | Numerical (Float) | **Yes (~2%)** | `2,500.0` – `8,000.0` | Primary Regression Target | Total body mass measured in grams. |
| 11 | `body_mass_index` | Numerical (Float) | **Yes (~2%)** | Derived | Engineered Feature | Ratio of mass to body length squared: `body_mass_g / (body_length_cm)^2`. |
| 12 | `flipper_body_ratio` | Numerical (Float) | No | Derived | Engineered Feature | Ratio of flipper length relative to total body length: `flipper_length_mm / body_length_cm`. |
| 13 | `bill_area_index` | Numerical (Float) | **Yes (~2%)** | Derived | Engineered Feature | Synthetic surface area index: `bill_length_mm * bill_depth_mm`. |
| 14 | `health_score` | Numerical (Float) | **Yes (~1.9%)** | `0.0` – `100.0` | Health Indicator | Composite health score based on food availability and climate stress. |
| 15 | `growth_stage` | Categorical | No | `Juvenile`, `Subadult`, `Adult` | Categorical Feature | Growth phase derived from age (`<2.5`, `2.5–4.5`, `>4.5` years). |
| 16 | `size_score` | Numerical (Float) | No | `0.0` – `100.0` | Continuous Metric | Weighted composite physical size score combining mass, length, flipper, and bill traits. |
| 17 | `size_category` | Categorical | No | `Small`, `Medium`, `Large` | **Primary Classification Target** | Binned size classification derived from normalized `size_score`. |
| 18 | `weight_category` | Categorical | **Yes (~2%)** | `Light`, `Normal`, `Heavy` | Secondary Classification Target | Categorical weight classification relative to species standards. |
| 19 | `environment_temperature_c` | Numerical (Float) | No | `-15.0` – `10.0` | Environmental Feature | Ambient temperature of the habitat environment in degrees Celsius. |
| 20 | `food_availability` | Numerical (Float) | No | `0.0` – `100.0` | Environmental Feature | Prey and food availability score in the penguin's immediate habitat. |
| 21 | `activity_level` | Numerical (Float) | **Yes (~1.9%)** | `0.0` – `100.0` | Behavioral Feature | Simulated daily physical activity score based on health and age. |
| 22 | `survival_condition_score` | Numerical (Float) | **Yes (~2%)** | `0.0` – `100.0` | Composite Indicator | Overall simulated condition index combining health, food, activity, and climate. |
