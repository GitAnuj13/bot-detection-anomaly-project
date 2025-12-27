from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("../outputs")
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("../processed/final_session_classification.csv")

def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {PLOT_DIR / filename}")

fig = plt.figure(figsize=(6,4))
df[df["actual_label"]=="Human"]["requests_per_sec"].hist(alpha=0.6, bins=30, label="Human")
df[df["actual_label"]=="Bot"]["requests_per_sec"].hist(alpha=0.6, bins=30, label="Bot")
plt.legend()
plt.title("Requests per Second: Human vs Bot")
plt.xlabel("Requests/sec")
plt.ylabel("Count")

save_plot(fig, "requests_per_sec_distribution.png")

fig = plt.figure(figsize=(6,4))
df[df["actual_label"]=="Human"]["avg_scroll_depth"].hist(alpha=0.6, bins=30, label="Human")
df[df["actual_label"]=="Bot"]["avg_scroll_depth"].hist(alpha=0.6, bins=30, label="Bot")
plt.legend()
plt.title("Scroll Depth: Human vs Bot")

save_plot(fig, "scroll_depth_distribution.png")

fig = plt.figure(figsize=(6,4))
df[df["actual_label"]=="Human"]["avg_mouse_movements"].hist(alpha=0.6, bins=30, label="Human")
df[df["actual_label"]=="Bot"]["avg_mouse_movements"].hist(alpha=0.6, bins=30, label="Bot")
plt.legend()
plt.title("Mouse Movement: Human vs Bot")

save_plot(fig, "mouse_movement_distribution.png")

features = [
    "requests_per_sec",
    "avg_scroll_depth",
    "avg_mouse_movements"
]

for col in features:
    fig = plt.figure(figsize=(5,4))
    df.boxplot(column=col, by="actual_label")
    plt.title(col)
    plt.suptitle("")

    save_plot(fig, f"boxplot_{col}.png")

corr = df[
    [
        "requests_per_sec",
        "avg_scroll_depth",
        "avg_mouse_movements",
        "pages_per_min",
        "scroll_per_sec"
    ]
].corr()

fig = plt.figure(figsize=(6,5))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Feature Correlation Matrix")

save_plot(fig, "feature_correlation_matrix.png")

df["actual_label"].value_counts().to_csv(
    TABLE_DIR / "class_distribution.csv"
)

ml_extra = df[
    (df["actual_label"]=="Bot") &
    (df["classification"]=="Human") &
    (df["if_bot"]==1)
]

ml_extra.to_csv(
    TABLE_DIR / "ml_caught_rules_missed.csv",
    index=False
)

false_positives = df[
    (df["actual_label"]=="Human") &
    (df["if_bot"]==1)
]

false_positives.to_csv(
    TABLE_DIR / "ml_false_positives.csv",
    index=False
)
