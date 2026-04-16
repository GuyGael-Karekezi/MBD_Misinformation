from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local African dataset CSV.")
    parser.add_argument("--csv", type=Path, default=Path("docs/african_validation/template.csv"))
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    missing = {field: sum(1 for row in rows if not row[field].strip()) for field in ["image_path", "text", "label", "country_focus", "language"]}
    label_counts = Counter(row["label"] for row in rows if row["label"].strip())
    country_counts = Counter(row["country_focus"] for row in rows if row["country_focus"].strip())
    path_counts = Counter(row["image_path"] for row in rows if row["image_path"].strip())
    bad_labels = [row["id"] for row in rows if row["label"].strip() and row["label"] not in {"misinformation", "likely_consistent"}]
    missing_files = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_path"]].append(row)
        image_path = row["image_path"].strip()
        if image_path and not Path(image_path).exists():
            missing_files.append((row["id"], image_path))

    print(f"CSV: {args.csv}")
    print(f"Total rows: {len(rows)}")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Missing fields: {missing}")
    print(f"Country counts: {dict(country_counts)}")
    print(f"Unique image paths: {len(path_counts)}")
    print(f"Invalid labels: {len(bad_labels)}")
    print(f"Rows with missing local files: {len(missing_files)}")
    pair_issues = 0
    for image_path, group in grouped.items():
        if not image_path:
            continue
        labels = sorted(row["label"] for row in group if row["label"].strip())
        if labels and labels != ["likely_consistent", "misinformation"]:
            pair_issues += 1
    print(f"Pair integrity issues: {pair_issues}")

    if missing_files:
        print("Example missing files:")
        for row_id, image_path in missing_files[:10]:
            print(f"  row {row_id}: {image_path}")

if __name__ == "__main__":
    main()
