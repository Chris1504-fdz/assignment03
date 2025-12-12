import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

# ============================================================
# PART 1 — LOAD DATASET AND BASIC STRUCTURE
# ============================================================

print("\n====================")
print("1. DATASET STRUCTURE")
print("====================\n")

ds = load_dataset("Anthropic/hh-rlhf", split="train")

print(f"Dataset size: {len(ds)}")
print(f"Columns: {ds.column_names}")

# Show a few example rows
print("\nExample data points:")
for i in range(1):
    print(f"\n--- Example {i} ---")
    print("Chosen:", ds[i]["chosen"][:300], "...")
    print("\nRejected:", ds[i]["rejected"][:300], "...")


# ============================================================
# PART 2 — LENGTH ANALYSIS (MEAN, MEDIAN, DISTRIBUTIONS)
# ============================================================

print("\n==========================")
print("2. LENGTH DISTRIBUTION")
print("==========================\n")

# Compute lengths
chosen_lengths = [len(x.split()) for x in ds["chosen"]]
rejected_lengths = [len(x.split()) for x in ds["rejected"]]

print(f"Mean chosen length: {np.mean(chosen_lengths)}")
print(f"Mean rejected length: {np.mean(rejected_lengths)}")
print(f"Median chosen length: {np.median(chosen_lengths)}")
print(f"Median rejected length: {np.median(rejected_lengths)}")

# Compare chosen vs rejected
length_diff = np.array(chosen_lengths) - np.array(rejected_lengths)

print("\n--- Length Comparison ---")
print("Fraction where chosen is longer:",
      np.mean(length_diff > 0))
print("Fraction where chosen is shorter:",
      np.mean(length_diff < 0))
print("Fraction where equal length:",
      np.mean(length_diff == 0))

print("Mean length difference (chosen - rejected):",
      np.mean(length_diff))
print("Median length difference:",
      np.median(length_diff))


# ============================================================
# PART 3 — HISTOGRAM OF LENGTH DIFFERENCES
# ============================================================

print("\n=============================")
print("3. HISTOGRAM OF DIFFERENCES")
print("=============================\n")

bins = [-500, -100, -50, -10, -1, 0, 1, 10, 50, 100, 500, 10000]
hist, edges = np.histogram(length_diff, bins=bins)

print("Histogram summary (chosen - rejected):")
for count, (a, b) in zip(hist, zip(edges[:-1], edges[1:])):
    print(f"{a:>6} to {b:>6}: {count}")

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(length_diff, bins=bins, edgecolor="black")
plt.title("Histogram of Chosen - Rejected Length Differences")
plt.xlabel("Length difference (words)")
plt.ylabel("Count")
plt.xticks(bins, rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PART 4 — SIMPLE BIAS / PATTERN ANALYSIS
# ============================================================

print("\n===============================")
print("4. KEYWORD BIAS ANALYSIS")
print("===============================\n")

# Keywords grouped by theme
keywords = {
    "safety": ["violence", "harm", "kill", "illegal"],
    "sexual": ["sex", "porn", "fuck", "cum"],
    "identity": ["race", "gender", "religion"],
}

# Count occurrences in chosen + rejected text
counts = {k: 0 for k in keywords}

for item in ds:
    text = (item["chosen"] + " " + item["rejected"]).lower()
    for category, keys in keywords.items():
        if any(word in text for word in keys):
            counts[category] += 1

print("Keyword category counts:")
for cat, c in counts.items():
    print(f"  {cat}: {c}")

print("\nRelative frequencies:")
for cat, c in counts.items():
    print(f"  {cat}: {c / len(ds):.3f}")


# ============================================================
# PART 5 — CHECK PAIR CONSISTENCY (OPTIONAL BUT USEFUL)
# ============================================================

print("\n======================================")
print("5. CHECK FOR EMPTY / INVALID PAIRS")
print("======================================\n")

missing_or_empty = [
    i for i, x in enumerate(ds)
    if x["chosen"] is None or x["rejected"] is None
    or len(x["chosen"].strip()) == 0
    or len(x["rejected"].strip()) == 0
]

print("Number of empty or missing pairs:", len(missing_or_empty))
if missing_or_empty:
    print("Example problematic indices:", missing_or_empty[:10])

print("\n=== TASK A ANALYSIS COMPLETE ===\n")

# ============================================================
# PART 3B — DISTRIBUTION OF CHOSEN vs REJECTED LENGTHS
# ============================================================

print("\n==========================================")
print("3B. CHOSEN vs REJECTED LENGTH DISTRIBUTION")
print("==========================================\n")

# Choose reasonable bin edges (clip extreme long responses)
max_len = np.percentile(chosen_lengths + rejected_lengths, 99)  # avoid super-long outliers
bins = np.linspace(0, max_len, 50)

plt.figure(figsize=(10, 6))
plt.hist(chosen_lengths, bins=bins, alpha=0.5, density=True,
         label="Chosen responses")
plt.hist(rejected_lengths, bins=bins, alpha=0.5, density=True,
         label="Rejected responses")

plt.title("Word Length Distribution: Chosen vs Rejected Responses")
plt.xlabel("Response length (words)")
plt.ylabel("Density")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/chosen_vs_rejected_length_distribution.png")
plt.show()
