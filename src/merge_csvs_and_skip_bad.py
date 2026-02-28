import os
import pandas as pd
import shutil
import csv
import io
import re

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

source_folder = os.path.join(PROJECT_ROOT, "data", "raw", "compressed-attachments")
skipped_folder = os.path.join(PROJECT_ROOT, "results", "metrics", "skipped-files")
fixed_folder = os.path.join(PROJECT_ROOT, "data", "raw", "fixedfile")
output_file = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset.csv")
skipped_report = os.path.join(PROJECT_ROOT, "results", "metrics", "skipped_report.csv")

REQUIRED_COLUMNS = [
    "Image URL", "Description", "Country", "Weather",
    "Time of Day", "Season", "Activity", "Mood/Emotion"
]

# Prepare folders
os.makedirs(skipped_folder, exist_ok=True)
os.makedirs(fixed_folder, exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
os.makedirs(os.path.dirname(skipped_report), exist_ok=True)

valid_dataframes = []
skipped_records = []

ENCODINGS_TO_TRY = ["utf-8", "cp1252", "latin1"]
EXTRA_ENCODINGS = ["utf-16", "utf-16le", "utf-16be", "iso-8859-1"]

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def normalize_column_name(name):
    return "".join(ch.lower() for ch in name if ch.isalnum())


NORMALIZED_REQUIRED = {normalize_column_name(c): c for c in REQUIRED_COLUMNS}

ALIAS_MAP_RAW = {
    "imageurl": "Image URL",
    "image_url": "Image URL",
    "imgurl": "Image URL",
    "image": "Image URL",
    "description": "Description",
    "desc": "Description",
    "countryname": "Country",
    "country": "Country",
    "weathercondition": "Weather",
    "weather": "Weather",
    "timeofday": "Time of Day",
    "time_of_day": "Time of Day",
    "season": "Season",
    "activitytype": "Activity",
    "activity": "Activity",
    "mood": "Mood/Emotion",
    "emotion": "Mood/Emotion",
    "moodemotion": "Mood/Emotion",
    "mood_emotion": "Mood/Emotion",
}
ALIAS_MAP = {normalize_column_name(k): v for k, v in ALIAS_MAP_RAW.items()}


def read_text_with_encoding(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    for enc in ENCODINGS_TO_TRY + EXTRA_ENCODINGS:
        try:
            return data.decode(enc), enc
        except Exception:
            continue
    return None, None


def repair_unbalanced_quotes(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    repaired_lines = []
    buffer = ""
    for line in lines:
        if buffer:
            buffer += "\n" + line
        else:
            buffer = line
        if buffer.count('"') % 2 == 0:
            repaired_lines.append(buffer)
            buffer = ""
    if buffer:
        repaired_lines.append(buffer)
    return "\n".join(repaired_lines)


def looks_like_url(value):
    return bool(URL_PATTERN.match(value.strip()))


def read_csv_repaired(file_path):
    text, enc = read_text_with_encoding(file_path)
    if text is None:
        return None, None, f"Unable to decode with {ENCODINGS_TO_TRY + EXTRA_ENCODINGS}", False

    repaired_text = repair_unbalanced_quotes(text)
    repaired = repaired_text != text

    reader = csv.reader(io.StringIO(repaired_text))
    rows = list(reader)
    if not rows:
        return pd.DataFrame(), enc, None, repaired

    header = [h.lstrip("\ufeff") for h in rows[0]]
    normalized_header = [normalize_column_name(c) for c in header]
    header_has_known = any(h in NORMALIZED_REQUIRED or h in ALIAS_MAP for h in normalized_header)
    first_cell = header[0] if header else ""
    header_looks_like_data = looks_like_url(first_cell)

    columns = header
    data_rows = rows[1:]

    if not header_has_known and header_looks_like_data:
        data_rows = rows
        if len(header) >= len(REQUIRED_COLUMNS):
            extra_count = len(header) - len(REQUIRED_COLUMNS)
            columns = REQUIRED_COLUMNS + [f"extra_{i}" for i in range(extra_count)]
        else:
            columns = REQUIRED_COLUMNS[:len(header)]
    elif not header_has_known and len(header) == len(REQUIRED_COLUMNS):
        data_rows = rows
        columns = REQUIRED_COLUMNS

    fixed_rows = []
    for row in data_rows:
        if len(row) < len(columns):
            row = row + [""] * (len(columns) - len(row))
        elif len(row) > len(columns):
            row = row[:len(columns)]
        fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=columns)
    return df, enc, None, repaired


# Loop over all files
if not os.path.isdir(source_folder):
    raise FileNotFoundError(
        f"Source folder not found: {source_folder}\n"
        "Create it and place raw CSV submissions there."
    )

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(root, file)

        df, used_encoding, error, repaired = read_csv_repaired(file_path)
        if df is None:
            skipped_records.append(
                {"file": file, "reason": error or "Unable to decode", "missing_columns": "", "encoding": ""}
            )
            shutil.copy(file_path, os.path.join(skipped_folder, file))
            print(f"[SKIPPED] {file} - {error}")
            continue

        if df.shape[0] == 0:
            reason = "No data rows"
            skipped_records.append(
                {"file": file, "reason": reason, "missing_columns": "", "encoding": used_encoding or ""}
            )
            shutil.copy(file_path, os.path.join(skipped_folder, file))
            print(f"[SKIPPED] {file} - {reason}")
            continue

        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        normalized_columns = {normalize_column_name(c): c for c in df.columns}

        if not any(col in NORMALIZED_REQUIRED for col in normalized_columns) and df.shape[1] == len(REQUIRED_COLUMNS):
            df.columns = REQUIRED_COLUMNS
            normalized_columns = {normalize_column_name(c): c for c in df.columns}

        rename_map = {}
        for col in df.columns:
            normalized_col = normalize_column_name(col)
            if normalized_col in NORMALIZED_REQUIRED:
                rename_map[col] = NORMALIZED_REQUIRED[normalized_col]
            elif normalized_col in ALIAS_MAP:
                rename_map[col] = ALIAS_MAP[normalized_col]

        if rename_map:
            df = df.rename(columns=rename_map)

        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        for col in missing_cols:
            df[col] = ""

        original_row_count = df.shape[0]

        df = df[REQUIRED_COLUMNS]
        df = df.fillna("")
        for col in REQUIRED_COLUMNS:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r"[\r\n]+", " ", regex=True)

        df = df[(df["Image URL"] != "") & (df["Description"] != "")]
        df = df[df["Image URL"].str.match(URL_PATTERN, na=False)]
        df["source_file"] = file
        valid_dataframes.append(df)

        row_count_changed = df.shape[0] != original_row_count

        if used_encoding != "utf-8" or rename_map or missing_cols or repaired or row_count_changed:
            fixed_path = os.path.join(fixed_folder, file)
            df.to_csv(fixed_path, index=False, encoding="utf-8")
            print(f"[FIXED+MERGED] {file}")
        else:
            print(f"[MERGED] {file}")

# Save merged output
if valid_dataframes:
    merged_df = pd.concat(valid_dataframes, ignore_index=True)
    merged_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nDONE: Final merged file saved as: {output_file}")
else:
    print("\nNo valid CSV files were found.")

if skipped_records:
    pd.DataFrame(skipped_records).to_csv(skipped_report, index=False, encoding="utf-8")
    print(f"DONE: Skipped report saved as: {skipped_report}")
