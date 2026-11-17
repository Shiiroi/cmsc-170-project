import csv
import os
import re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "table7_fies_2023.csv")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "table7_cleaned.csv")

# 10 core expenditure features
FEATURES = [
    ("food_home_pct", "food consumed at home"),
    ("food_outside_pct", "food regularly consumed outside the home"),
    ("housing_utilities_pct", "housing, water, electricity, gas and other fuels"),
    ("transport_pct", "transport"),
    ("infocomm_pct", "information and communication"),
    ("education_pct", "education"),
    ("health_pct", "health"),
    ("clothing_pct", "clothing and footwear"),
    ("recreation_pct", "recreation, sport and culture"),
    ("occasions_pct", "special family occasion"),
]


# -------------------------------------------------------------------
# DATA COLLECTION: read the raw PSA table
# -------------------------------------------------------------------

def data_collection(path=RAW_PATH):
    raw_rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = None
        deciles = {}
        area_idx = 0
        cat_idx = 1

        for row in reader:
            if not row or not any(row):
                continue

            # find header row with 10 deciles
            if header is None:
                found = {}
                for i, c in enumerate(row):
                    m = re.search(r"^(\d+)(?:st|nd|rd|th)?\s+decile$", c.strip(), re.IGNORECASE)
                    if m:
                        found[int(m.group(1))] = i
                if len(found) >= 10:
                    header = row
                    deciles = found
                    for i, c in enumerate(row):
                        cl = c.strip().lower()
                        if "region" in cl or "province" in cl or "huc" in cl:
                            area_idx = i
                        elif "expenditure" in cl and "all income" not in cl:
                            cat_idx = i

                continue

            # extract values per decile
            values = []
            for d in range(1, 11):
                values.append(row[deciles[d]])
            raw_rows.append((row[area_idx].strip(), row[cat_idx].strip(), values))
    
    return raw_rows


# -------------------------------------------------------------------
# DATA CLEANING: remove aggregate rows, clean names
# -------------------------------------------------------------------

# check if national or regional aggregate
def is_regional_or_national(area):
    clean = re.sub(r"\s+", " ", area).strip().upper()
    clean = re.sub(r"\d+/$", "", clean).strip()
    if clean == "PHILIPPINES":
        return True
    if clean.startswith("REGION "):
        return True
    if clean.startswith("NIR"):
        return True
    if "ADMINISTRATIVE REGION" in clean:
        return True
    if "NATIONAL CAPITAL REGION" in clean:
        return True
    if "MIMAROPA" in clean:
        return True
    if clean == "CARAGA":
        return True
    if "BANGSAMORO" in clean:
        return True
    if clean == "MAGUINDANAO":
        return True
    return False


# clean province or city name
def clean_area(name):
    name = name.lstrip(".").strip()
    name = re.sub(r"\d+/$", "", name).strip()
    name = name.replace("\ufffd", "ñ")
    name = re.sub(r"\s+", " ", name)
    return name


# clean category name
def clean_cat(name):
    return name.lstrip(".").strip().lower()


def data_cleaning(raw_rows):
    cleaned_rows = []
    for raw_area, raw_cat, values in raw_rows:
        # filter national and regional aggregates
        if is_regional_or_national(raw_area):
            continue
        cleaned_rows.append((clean_area(raw_area), clean_cat(raw_cat), values))
    
    return cleaned_rows


# -------------------------------------------------------------------
# DATA PREPROCESSING: select features, reshape, make labels, save
# -------------------------------------------------------------------

# map decile to income tier (0: low, 1: mid, 2: high)
def get_tier(decile):
    if decile <= 3:
        return 0
    elif decile <= 7:
        return 1
    else:
        return 2


def data_preprocessing(cleaned_rows, save_path=PROCESSED_PATH):
    cat_map = {cat: col for col, cat in FEATURES}
    feature_names = [col for col, _ in FEATURES]

    # select the feature categories: data[area][decile][feature] = value
    areas = []
    data = {}
    for area, cat, values in cleaned_rows:
        # match 10 core features
        if cat not in cat_map:
            continue

        if area not in data:
            areas.append(area)
            data[area] = {d: {} for d in range(1, 11)}
        
        # extract values per decile
        for d in range(1, 11):
            data[area][d][cat_map[cat]] = float(values[d-1].strip())
    
    # unpivot deciles into rows
    out_header = ["area_name", "decile"] + feature_names + ["income_tier"]
    out_rows = []

    for area in areas:
        for d in range(1, 11):
            tier = get_tier(d)
            r = [area, d]
            for col in feature_names:
                r.append(data[area][d][col])
            r.append(tier)
            out_rows.append(r)

    # save the processed dataset
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for r in out_rows:
            formatted_row = [r[0], r[1]]
            for val in r[2:12]:
                formatted_row.append(f"{val:.2f}")
            formatted_row.append(r[12])
            writer.writerow(formatted_row)
    
    # dataset for the next pipeline step
    X = [r[2:12] for r in out_rows]
    y = [r[12] for r in out_rows]
    row_areas = [r[0] for r in out_rows]
    row_deciles = [r[1] for r in out_rows]

    return {"X": np.array(X), "y": np.array(y), "areas": row_areas,
            "deciles": row_deciles, "features": feature_names}


# -------------------------------------------------------------------
# RUN ON ITS OWN: py src/prepare_dataset.py
# -------------------------------------------------------------------
if __name__ == "__main__":
    dataset = data_preprocessing(data_cleaning(data_collection()))

    # validation checks and summary
    labels = list(dataset["y"])
    c0 = labels.count(0)
    c1 = labels.count(1)
    c2 = labels.count(2)

    print(f"Total rows: {len(labels)}")
    print(f"Total unique areas: {len(set(dataset['areas']))}")
    print(f"Class 0 (Low-Income): {c0} ({c0/len(labels)*100:.1f}%)")
    print(f"Class 1 (Middle-Income): {c1} ({c1/len(labels)*100:.1f}%)")
    print(f"Class 2 (High-Income): {c2} ({c2/len(labels)*100:.1f}%)")
    print(f"Output saved to: {PROCESSED_PATH}")

