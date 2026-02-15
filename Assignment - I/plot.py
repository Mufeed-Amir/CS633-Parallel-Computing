import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Prepare Data
data_list = []
raw_data = {
    'P=8': {
        'M=262,144': [0.060277, 0.068422, 0.062839, 0.063276, 0.060804],
        'M=1,048,576': [0.268953, 0.281943, 0.275317, 0.275278, 0.270129]
    },
    'P=16': {
        'M=262,144': [0.090931, 0.092373, 0.091415, 0.084849, 0.094549],
        'M=1,048,576': [0.359709, 0.368446, 0.371990, 0.342490, 0.374188]
    },
    'P=32': {
        'M=262,144': [0.258224, 0.297482, 0.266257, 0.245825, 0.309165],
        'M=1,048,576': [0.501498, 0.585146, 0.496710, 0.521844, 0.539692]
    }
}

for p_label, m_dict in raw_data.items():
    for m_label, times in m_dict.items():
        for t in times:
            data_list.append({'Process Count': p_label, 'Message Size (M)': m_label, 'Time (s)': t})

df = pd.DataFrame(data_list)

# 2. Style Setup (Cleaner, Academic Look)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(10, 6), dpi=150) # Higher DPI for crispness

# Define colors manually for consistent control
# Using a professional "Paired" inspired look: Soft Blue vs Soft Red
my_palette = {"M=262,144": "#7fb3d5", "M=1,048,576": "#e6b0aa"}

# 3. Create Boxplot
# width=0.6 prevents them from looking too skinny
# fliersize=0 hides the default diamond outliers so we can use our own dots
ax = sns.boxplot(data=df, x='Process Count', y='Time (s)', hue='Message Size (M)',
                 palette=my_palette, width=0.5, linewidth=1.2, fliersize=0)

# 4. Add Stripplot (The Dots)
# dodge=True is CRITICAL: it aligns the dots with the split boxes
sns.stripplot(data=df, x='Process Count', y='Time (s)', hue='Message Size (M)',
              dodge=True, jitter=True, size=5, color="#333333", alpha=0.5, legend=False, ax=ax)

# 5. Formatting
ax.set_title("MPI Pipeline Execution Time vs Process Count", fontsize=14, weight='bold', pad=15)
ax.set_ylabel("Execution Time (seconds)", weight='bold', labelpad=10)
ax.set_xlabel("Number of Processes (P)", weight='bold', labelpad=10)

# Refined Legend
# We manually set the title and frame to make it look like a proper figure key
ax.legend(title="Message Size", loc='upper left', frameon=True, framealpha=0.9, fancybox=True)

# Grid and Spines
ax.yaxis.grid(True, linestyle='--', alpha=0.6) # Lighter grid
ax.xaxis.grid(False) # No vertical grid lines
sns.despine(left=True, bottom=False) # Open look

plt.tight_layout()
plt.savefig('final_polished_plot.png', dpi=300)
plt.show()