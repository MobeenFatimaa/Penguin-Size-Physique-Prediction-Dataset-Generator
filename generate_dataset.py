import time
import numpy as np
import polars as pl

def generate_penguin_dataset(n_samples: int = 100_000, seed: int = 42) -> pl.DataFrame:
    """
    Generates a 100,000-record synthetic penguin physical characteristics and size dataset.
    Uses vectorized NumPy operations and Polars for fast generation (<2s execution).
    """
    start_time = time.time()
    np.random.seed(seed)

    # 1. Base Identifiers & Categories
    penguin_id = np.arange(1, n_samples + 1, dtype=np.int32)
    
    # Island distribution
    island_choices = np.array(["Biscoe", "Dream", "Torgersen"])
    island_probs = [0.48, 0.36, 0.16]
    island = np.random.choice(island_choices, size=n_samples, p=island_probs)

    # Species conditioned on island probabilities
    species_options = np.array(["Adelie", "Chinstrap", "Gentoo"])
    species = np.empty(n_samples, dtype=object)
    
    biscoe_mask = (island == "Biscoe")
    dream_mask = (island == "Dream")
    torgersen_mask = (island == "Torgersen")
    
    species[biscoe_mask] = np.random.choice(species_options, size=biscoe_mask.sum(), p=[0.25, 0.05, 0.70])
    species[dream_mask] = np.random.choice(species_options, size=dream_mask.sum(), p=[0.35, 0.60, 0.05])
    species[torgersen_mask] = np.random.choice(species_options, size=torgersen_mask.sum(), p=[0.90, 0.05, 0.05])

    # Sex (50/50 split)
    sex = np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    
    # 2. Age & Growth Stage
    age_years = np.round(np.random.gamma(shape=2.5, scale=2.0, size=n_samples) + 0.5, 1)
    age_years = np.clip(age_years, 0.5, 15.0)

    growth_stage = np.empty(n_samples, dtype=object)
    growth_stage[age_years < 2.5] = "Juvenile"
    growth_stage[(age_years >= 2.5) & (age_years <= 4.5)] = "Subadult"
    growth_stage[age_years > 4.5] = "Adult"

    # Growth modifier for physical measurements based on age
    growth_factor = np.where(age_years < 2.5, 0.85 + (age_years / 2.5) * 0.12,
                    np.where(age_years <= 4.5, 0.97 + ((age_years - 2.5) / 2.0) * 0.03, 1.0))

    # 3. Environmental & Health Factors
    env_temp_base = np.where(island == "Biscoe", -8.0, np.where(island == "Dream", -4.0, -2.0))
    environment_temperature_c = np.round(env_temp_base + np.random.normal(0, 3.5, size=n_samples), 1)
    environment_temperature_c = np.clip(environment_temperature_c, -15.0, 10.0)

    food_availability = np.round(np.clip(np.random.normal(65, 18, size=n_samples), 0, 100), 1)
    
    health_base = 50 + (food_availability * 0.3) - (np.maximum(0, environment_temperature_c - 5) * 1.5)
    health_score = np.round(np.clip(health_base + np.random.normal(0, 10, size=n_samples), 0, 100), 1)

    # Activity level based on age and health
    activity_base = (health_score * 0.5) + (100 - (age_years * 4)) * 0.3
    activity_level = np.round(np.clip(activity_base + np.random.normal(0, 10, size=n_samples), 0, 100), 1)

    # 4. Core Physical Measurements (Correlated)
    # Base multipliers: Gentoo > Chinstrap > Adelie; Male > Female
    species_mult = np.where(species == "Gentoo", 1.22, np.where(species == "Chinstrap", 1.05, 0.92))
    sex_mult = np.where(sex == "Male", 1.06, 0.94)
    health_mult = 0.92 + (health_score / 100.0) * 0.12
    
    base_size = species_mult * sex_mult * growth_factor * health_mult

    body_length_cm = np.round(np.clip(np.random.normal(55, 6, size=n_samples) * base_size, 35.0, 85.0), 1)
    bill_length_mm = np.round(np.clip(np.random.normal(44, 4, size=n_samples) * base_size, 30.0, 65.0), 1)
    bill_depth_mm = np.round(np.clip(np.random.normal(17, 2, size=n_samples) * (base_size**0.5), 12.0, 25.0), 1)
    flipper_length_mm = np.round(np.clip(np.random.normal(195, 12, size=n_samples) * base_size, 150.0, 240.0), 0)
    
    # Body Mass derived logarithmically from physical dimensions with random noise
    mass_latent = (body_length_cm**1.4) * (flipper_length_mm**0.8) * 0.15 * health_mult
    body_mass_g = np.round(np.clip(mass_latent + np.random.normal(0, 250, size=n_samples), 2500, 8000), 0)

    # 5. Feature Engineering
    body_mass_index = np.round(body_mass_g / (body_length_cm ** 2), 4)
    flipper_body_ratio = np.round(flipper_length_mm / body_length_cm, 3)
    bill_area_index = np.round(bill_length_mm * bill_depth_mm, 2)

    # Shifted min/max bounds to shift the median score higher (~50)
    mass_norm = np.clip((body_mass_g - 2500) / (5500 - 2500), 0, 1)
    len_norm = np.clip((body_length_cm - 36) / (70 - 36), 0, 1)
    flip_norm = np.clip((flipper_length_mm - 150) / (220 - 150), 0, 1)
    bill_norm = np.clip((bill_area_index - 380) / (1100 - 380), 0, 1)

    raw_score = (mass_norm * 40) + (len_norm * 25) + (flip_norm * 20) + (bill_norm * 15)

    # Gaussian noise for soft boundaries
    fuzzy_noise = np.random.normal(0, 2.5, size=n_samples)
    size_score = np.round(np.clip(raw_score + fuzzy_noise, 0, 100), 1)

    # Calibrated class thresholds for ~25% Small, ~48% Medium, ~27% Large
    size_category = np.empty(n_samples, dtype=object)
    size_category[size_score < 36.0] = "Small"
    size_category[(size_score >= 36.0) & (size_score < 64.0)] = "Medium"
    size_category[size_score >= 64.0] = "Large"

    # Weight Category based on mass relative to species expectations
    weight_category = np.empty(n_samples, dtype=object)
    weight_category[body_mass_g < 3600] = "Light"
    weight_category[(body_mass_g >= 3600) & (body_mass_g <= 4800)] = "Normal"
    weight_category[body_mass_g > 4800] = "Heavy"

    # Survival condition score
    survival_score = (health_score * 0.35) + (food_availability * 0.30) + (activity_level * 0.20) + ((10 - np.abs(environment_temperature_c)) * 1.5)
    survival_condition_score = np.round(np.clip(survival_score, 0, 100), 1)

    # Build DataFrame
    df = pl.DataFrame({
        "penguin_id": penguin_id,
        "species": species,
        "island": island,
        "sex": sex,
        "age_years": age_years,
        "body_length_cm": body_length_cm,
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "body_mass_index": body_mass_index,
        "flipper_body_ratio": flipper_body_ratio,
        "bill_area_index": bill_area_index,
        "health_score": health_score,
        "growth_stage": growth_stage,
        "size_score": size_score,
        "size_category": size_category,
        "weight_category": weight_category,
        "environment_temperature_c": environment_temperature_c,
        "food_availability": food_availability,
        "activity_level": activity_level,
        "survival_condition_score": survival_condition_score
    })

    # 6. Inject Outliers (~0.8%) & Missing Values (~2%)
    outlier_idx = np.random.choice(n_samples, size=int(n_samples * 0.008), replace=False)
    
    # Mutate selected features in polar dataframe for outliers
    df = df.with_columns([
        pl.when(pl.col("penguin_id").is_in(outlier_idx[::2]))
        .then(pl.col("body_mass_g") * 1.4)
        .otherwise(pl.col("body_mass_g")).alias("body_mass_g"),
        
        pl.when(pl.col("penguin_id").is_in(outlier_idx[1::2]))
        .then(pl.col("bill_length_mm") * 0.6)
        .otherwise(pl.col("bill_length_mm")).alias("bill_length_mm")
    ])

    # Missing values injection on specific continuous traits (never targets or IDs)
    missing_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "health_score"]
    
    for col in missing_cols:
        null_mask = np.random.rand(n_samples) < 0.02
        df = df.with_columns(
            pl.when(pl.Series(null_mask))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

    print(f"Dataset generated successfully in {time.time() - start_time:.2f} seconds.")
    return df

if __name__ == "__main__":
    df = generate_penguin_dataset(n_samples=100_000, seed=42)
    df.write_csv("penguin_size_dataset.csv")
    print(f"File saved: penguin_size_dataset.csv (Shape: {df.shape})")