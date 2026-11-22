SEED = 42

# data
DATA_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
FULL_CSV = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
SAMPLE_CSV = "data/sample/phiusiil_sample.csv"
SAMPLE_SIZE = 5000

# columns
LABEL_COL = "label"        # raw file: 1 = legitimate, 0 = phishing
GROUP_COL = "Domain"       # used later for the domain-grouped split
TEXT_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]