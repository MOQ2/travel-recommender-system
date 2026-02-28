import os
import pandas as pd
import shutil

# Paths
base_path = "compressed-data"
cleaned_path = "cleaned-data"
skipped_path = "skipped-files"

# Make folders if not exist
os.makedirs(cleaned_path, exist_ok=True)
os.makedirs(skipped_path, exist_ok=True)

# Keywords that must be present (not exact column names, just meaning)
REQUIRED_KEYWORDS = ["image", "time", "season", "activity", "feeling"]

def has_required_columns(columns):
    normalized = [col.strip().lower() for col in columns]
    return all(any(keyword in col for col in normalized) for keyword in REQUIRED_KEYWORDS)

# Loop through each file
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".csv") or file.endswith(".xlsx"):
            file_path = os.path.join(root, file)

            try:
                if file.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)

                if has_required_columns(df.columns):
                    shutil.copy(file_path, os.path.join(cleaned_path, file))
                    print(f"[✅ CLEANED] {file}")
                else:
                    shutil.copy(file_path, os.path.join(skipped_path, file))
                    print(f"[⛔ SKIPPED] {file} - Missing required columns")

            except Exception as e:
                shutil.copy(file_path, os.path.join(skipped_path, file))
                print(f"[💥 ERROR] {file} - {str(e)}")
