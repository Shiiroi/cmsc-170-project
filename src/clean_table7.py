import csv
import os
import re

# 10 core expenditure features
features = [
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

cat_map = {clean_name: col for col, clean_name in features}


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


# map decile to income tier (0: low, 1: mid, 2: high)
def get_tier(decile):
    if decile <= 3:
        return 0
    elif decile <= 7:
        return 1
    else:
        return 2


# find raw csv file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_file = os.path.join(ROOT, "data", "raw", "table7_fies_2023.csv")

# parse raw table
areas = []
data = {}

with open(raw_file, "r", encoding="utf-8", errors="replace") as f:
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
            else:
                continue

        raw_area = row[area_idx].strip()
        raw_cat = row[cat_idx].strip()

        # filter national and regional aggregates
        if is_regional_or_national(raw_area):
            continue

        # match 10 core features
        norm_cat = clean_cat(raw_cat)
        if norm_cat not in cat_map:
            continue

        feat_col = cat_map[norm_cat]
        c_area = clean_area(raw_area)

        if c_area not in data:
            areas.append(c_area)
            data[c_area] = {d: {} for d in range(1, 11)}

        # extract values per decile
        for d in range(1, 11):
            val = float(row[deciles[d]].strip())
            data[c_area][d][feat_col] = val

# unpivot deciles into rows
out_header = ["area_name", "decile"] + [col for col, _ in features] + ["income_tier"]
out_rows = []

for area in areas:
    for d in range(1, 11):
        tier = get_tier(d)
        r = [area, d]
        for col, _ in features:
            r.append(data[area][d][col])
        r.append(tier)
        out_rows.append(r)

# validation checks and summary
labels = [r[-1] for r in out_rows]
c0 = labels.count(0)
c1 = labels.count(1)
c2 = labels.count(2)

# none value check
has_nan = False
for r in out_rows:
    for val in r[2:12]:
        if val is None:
            has_nan = True
            break
    if has_nan:
        break

print(f"Total rows: {len(out_rows)}")
print(f"Total unique areas: {len(areas)}")
print(f"Class 0 (Low-Income): {c0} ({c0/len(labels)*100:.1f}%)")
print(f"Class 1 (Middle-Income): {c1} ({c1/len(labels)*100:.1f}%)")
print(f"Class 2 (High-Income): {c2} ({c2/len(labels)*100:.1f}%)")
print(f"Missing / NaN values: {'Yes' if has_nan else '0 (Clean)'}")

# save cleaned csv
out_file = os.path.join(ROOT, "data", "processed", "table7_cleaned.csv")

with open(out_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(out_header)
    for r in out_rows:
        formatted_row = [r[0], r[1]]
        for val in r[2:12]:
            formatted_row.append(f"{val:.2f}")
        formatted_row.append(r[12])
        writer.writerow(formatted_row)

print(f"Output saved to: {out_file}")

