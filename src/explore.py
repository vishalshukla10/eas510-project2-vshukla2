import config
from src.data import load_raw, clean

df = load_raw()
print("shape:", df.shape)
print("columns:", list(df.columns))
print("non-numeric columns:", list(df.select_dtypes(exclude="number").columns))
print(df[config.LABEL_COL].value_counts())

clean_df, report = clean(df)
print(report)
g = clean_df[config.GROUP_COL]
print("unique domains:", g.nunique(),
      "| rows sharing a domain with another row:", int(g.duplicated(keep=False).sum()))