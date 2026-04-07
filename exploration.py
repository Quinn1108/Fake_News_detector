import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load CSV and build DataFrame
df = pd.read_csv("data.csv")

# Robust point_id from idea_id
if "idea_id" in df.columns:
    df["point_id"] = df["idea_id"].astype(str)
elif "id" in df.columns:
    df["point_id"] = df["id"].astype(str)
elif "uuid" in df.columns:
    df["point_id"] = df["uuid"].astype(str)
else:
    # If no ID column, use index as point_id
    df["point_id"] = df.index.astype(str)

# Extract required fields directly from columns
df["title"]   = df["title"].fillna("").astype(str).str.strip()
df["text"]    = df["text"].fillna("").astype(str).str.strip()
df["subject"] = df["subject"].fillna("Unknown").astype(str).str.strip()

df["label"] = pd.to_numeric(df["fake/true(0/1)"], errors="coerce").fillna(0).astype(int)
# label: 0 = Fake, 1 = True/Real

df["combined"] = (df["title"] + " " + df["text"]).str.lower()

print(f"Loaded {len(df):,} articles  |  Fake: {(df['label']==0).sum():,}  |  True: {(df['label']==1).sum():,}")
print(f"Columns: {df.columns.tolist()}\n")

#2. Theme definitions (tight, non-overbroad regexes)
theme_keywords = {
    "Electoral Integrity & Voting": ["ballot", "rigged", "voter fraud", "stolen", "counting", "voting machine", "election day", "recount"],
    "Administration & Governance": ["white house", "congress", "legislation", "executive order", "shadow government", "deep state", "partisan", "cabinet"],
    "Diplomatic Relations & Foreign Policy": ["russia", "china", "nato", "treaty", "foreign aid", "sanctions", "embassy", "kremlin", "intelligence"],
    "Judicial & Legal Scandals": ["investigation", "fbi", "indictment", "scandal", "subpoena", "leak", "classified", "court", "justice department"],
}

themes = {}
for theme, keywords in theme_keywords.items():
    regex = r"\b(" + "|".join(keywords) + r")\b"
    themes[theme] = df[df["combined"].str.contains(regex, na=False)]

#3. Baseline over entire dataset
base_total      = len(df)
base_fake_count = (df["label"] == 0).sum()
base_fake_ratio = 100 * base_fake_count / base_total if base_total else 0

# 4. Stats & unique traceable example per theme, top subject tables
summary_rows       = []
examples           = []
top_subject_tables = []
used_ids           = set()

for theme, subdf in themes.items():
    subdf  = subdf.copy()
    total  = len(subdf)
    n_fake = int((subdf["label"] == 0).sum())
    n_true = int((subdf["label"] == 1).sum())
    fake_ratio = 100 * n_fake / total if total else 0
    uplift     = (fake_ratio / base_fake_ratio) if base_fake_ratio else np.nan

    summary_rows.append({
        "Theme":              theme,
        "N":                  total,
        "Fake":               n_fake,
        "True":               n_true,
        "Pct_Fake":           fake_ratio,
        "Uplift_vs_Baseline": uplift if not np.isnan(uplift) else None,
    })

    # Unique example: prefer Fake articles not yet used, with some content, not URLs, and containing theme keywords
    example_row = None
    theme_kw_regex = r"\b(" + "|".join(theme_keywords[theme]) + r")\b"
    candidates  = subdf[
        (subdf["label"] == 0) & (~subdf["point_id"].isin(used_ids)) & 
        (subdf["text"].str.len() > 20) & (~subdf["text"].str.contains("http", na=False)) &
        (subdf["text"].str.contains(theme_kw_regex, na=False))
    ].copy()

    if not candidates.empty:
        candidates["text_len"] = candidates["text"].str.len()
        example_row = candidates.nsmallest(1, "text_len").iloc[0]
        used_ids.add(example_row["point_id"])

    if example_row is not None:
        txt = example_row["text"]
        if len(txt) > 120:
            txt = txt[:117] + "..."
        examples.append({
            "Theme":   theme,
            "Text":    txt,
            "Title":   example_row["title"],
            "subject": example_row["subject"],
        })

    # Top 3 subject sub-categories for the theme
    subject_stats = (
        subdf.groupby("subject")
        .agg(
            count=("point_id", "count"),
            avg_label=("label", "mean"),
        )
        .reset_index()
    )
    if not subject_stats.empty:
        fake_counts = (
            subdf[subdf["label"] == 0]
            .groupby("subject")["point_id"]
            .count()
        )
        subject_stats["pct_fake"] = (
            subject_stats["subject"].map(
                (fake_counts / subject_stats.set_index("subject")["count"]).fillna(0) * 100
            )
        )
        subject_stats = (
            subject_stats.fillna(0)
            .sort_values("count", ascending=False)
            .head(3)
        )
        top_subject_tables.append({"Theme": theme, "Table": subject_stats})

# Add baseline row
summary_rows.append({
    "Theme":              "Baseline (All articles)",
    "N":                  base_total,
    "Fake":               int(base_fake_count),
    "True":               int(base_total - base_fake_count),
    "Pct_Fake":           base_fake_ratio,
    "Uplift_vs_Baseline": 1.0,
})

summary = pd.DataFrame(summary_rows)
summary.set_index("Theme", inplace=True)

#5. Stacked Horizontal Bar Chart (Fake=red / True=amber)
summary_to_plot = summary[["Fake", "True"]].div(summary["N"], axis=0)
summary_to_plot.plot(
    kind="barh", stacked=True, color=["#f44336", "#ffc107"],
    figsize=(10, 6), edgecolor="black"
)

for idx, (name, row) in enumerate(summary.iterrows()):
    plt.text(1.01, idx, f'N={row["N"]}', va="center")

plt.title(
    "Fake vs True distribution by theme\n"
    "(N = sample size; Overlap allowed; Citation unique per theme)"
)
plt.xlabel("Proportion of Articles")
plt.legend(title="Article Label", bbox_to_anchor=(1.04, 1))
plt.xlim(0, 1.1)
plt.tight_layout()
plt.savefig("fake_news_themes.png", dpi=150, bbox_inches="tight")
plt.show()

# 6. Formatted summary table 
print("Theme | % Fake | Uplift vs Baseline")
print("-" * 55)
for theme, row in summary.iterrows():
    up = f"x{row['Uplift_vs_Baseline']:.1f}" if row["Uplift_vs_Baseline"] else "-"
    print(f"{theme:<28s} | {row['Pct_Fake']:>7.1f}% | {up}")

#7. One unique representative example per theme
print("\nRepresentative FAKE articles (one unique example per theme):\n")
for ex in examples:
    print(f"- {ex['Theme']}: \"{ex['Text']}\" (subject: {ex['subject']})")

# 8. Top 3 subjects per theme
print("\nTop 3 subjects by theme (count, avg_label, % fake):\n")
for t in top_subject_tables:
    print(f"{t['Theme']} :")
    tab = t["Table"]
    if tab.empty:
        print("  (No data)\n")
    else:
        display_tab = tab[["subject", "count", "avg_label", "pct_fake"]].rename(
            columns={
                "subject":   "Subject",
                "count":     "N",
                "avg_label": "Avg Label",
                "pct_fake":  "% Fake",
            }
        )
        display_tab["Avg Label"] = display_tab["Avg Label"].map("{:.2f}".format)
        display_tab["% Fake"]    = display_tab["% Fake"].map("{:.1f}%".format)
        print(display_tab.to_string(index=False))
        print()

# ── 4. Bar chart of proportions ──────────────────────────────────────────────
theme_names = list(themes.keys())
fake_props = []
true_props = []

for theme in theme_names:
    theme_df = themes[theme]
    total = len(theme_df)
    if total > 0:
        fake_count = (theme_df['label'] == 0).sum()
        true_count = (theme_df['label'] == 1).sum()
        fake_props.append(fake_count / total)
        true_props.append(true_count / total)
    else:
        fake_props.append(0)
        true_props.append(0)

plt.figure(figsize=(10, 6))
positions = range(len(theme_names))
plt.barh([p - 0.2 for p in positions], fake_props, height=0.4, label='Fake', color='red', alpha=0.7)
plt.barh([p + 0.2 for p in positions], true_props, height=0.4, label='True', color='blue', alpha=0.7)
plt.yticks(positions, theme_names)
plt.xlabel('Proportion')
plt.title('Proportions of Fake vs True News by Theme')
plt.legend()
plt.tight_layout()
plt.savefig('theme_proportions.png', dpi=150)
print("Bar chart saved as 'theme_proportions.png'")