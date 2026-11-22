import config
from src.data import load_raw, clean

df, _ = clean(load_raw())
sample = (df.groupby("phishing", group_keys=False)
            .sample(frac=config.SAMPLE_SIZE / len(df), random_state=config.SEED))
sample.to_csv(config.SAMPLE_CSV, index=False)
print("sample rows:", len(sample), "| phishing share:", round(sample["phishing"].mean(), 3))