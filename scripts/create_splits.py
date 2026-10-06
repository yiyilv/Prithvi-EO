from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed 70/15/15 LUCAS point splits")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV containing Point_ID and STR18 columns")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"Point_ID", "STR18"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            parser.error(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
        ids: list[str] = []
        point_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                class_id = int(row["STR18"])
            except (TypeError, ValueError) as error:
                parser.error(f"Invalid STR18 class on manifest line {line_number}: {error}")
            point_id = (row["Point_ID"] or "").strip()
            if not 1 <= class_id <= 10:
                parser.error(f"STR18 class must be in the range 1-10 on line {line_number}")
            if not point_id:
                parser.error(f"Manifest contains an empty Point_ID on line {line_number}")
            if "/" in point_id or "\\" in point_id:
                parser.error(f"Point_ID must not contain a path separator on line {line_number}")
            if point_id in point_ids:
                parser.error(f"Manifest contains a duplicate Point_ID: {point_id}")
            point_ids.add(point_id)
            ids.append(f"{class_id}/{point_id}")

    random.Random(args.seed).shuffle(ids)
    train_end = int(0.70 * len(ids))
    val_end = train_end + int(0.15 * len(ids))
    splits = {
        "lucas_train.txt": ids[:train_end],
        "lucas_val.txt": ids[train_end:val_end],
        "lucas_test.txt": ids[val_end:],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in splits.items():
        (args.output_dir / filename).write_text("\n".join(values) + "\n", encoding="utf-8")
        print(f"{filename}: {len(values)} rows")


if __name__ == "__main__":
    main()
