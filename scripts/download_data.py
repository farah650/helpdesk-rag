from pathlib import Path
from datasets import load_dataset

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Tobi-Bueck/customer-support-tickets")
print(ds)  # affiche les splits et le nombre de lignes

df = ds["train"].to_pandas()
df.to_csv(RAW / "tickets_raw.csv", index=False)

print("\nColonnes :", list(df.columns))
print("Lignes :", len(df))
print(df["language"].value_counts())
print(df.head(3).T)