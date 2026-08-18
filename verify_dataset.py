import polars as pl

def verify_dataset(filepath: str = "penguin_size_dataset.csv"):
    df = pl.read_csv(filepath)
    
    print("=" * 60)
    print("PENGUIN SIZE DATASET - INTEGRITY REPORT")
    print("=" * 60)
    print(f"Total Rows: {df.height:,}")
    print(f"Total Columns: {df.width}")
    
    print("\n1. Target Distribution (size_category):")
    print(df["size_category"].value_counts())
    
    print("\n2. Null Value Summary:")
    null_counts = df.null_count()
    for col in df.columns:
        cnt = null_counts[col][0]
        if cnt > 0:
            print(f" - {col}: {cnt:,} missing ({cnt/df.height*100:.2f}%)")

    print("\n3. Class Distribution by Species:")
    print(df.group_by(["species", "size_category"]).len().sort(["species", "size_category"]))

    print("\n4. Feature Correlation Snippet (body_mass_g vs size_score):")
    valid_df = df.drop_nulls(subset=["body_mass_g", "size_score"])
    corr = valid_df.select(pl.corr("body_mass_g", "size_score"))[0, 0]
    print(f" Pearson Correlation: {corr:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    verify_dataset()