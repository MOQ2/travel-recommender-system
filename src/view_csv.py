import math
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "master_dataset.csv"))

if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(f"Dataset file not found: {MASTER_CSV}")

df = pd.read_csv(MASTER_CSV)

# Show all rows in paginated figures (close a window to see the next page).
rows_per_page = 30
total_pages = max(1, math.ceil(len(df) / rows_per_page))

for page in range(total_pages):
    start = page * rows_per_page
    end = start + rows_per_page
    page_df = df.iloc[start:end]

    fig_height = max(8, len(page_df) * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=page_df.values,
        colLabels=page_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(f"Rows {start + 1}–{min(end, len(df))} of {len(df)}")

    plt.tight_layout()
    plt.show()
