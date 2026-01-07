import pandas as pd
import matplotlib.pyplot as plt


# sep="\t" because the files are TSV (tab-separated).
mcg = pd.read_csv("annotated_mcgill.tsv", sep="\t")
con = pd.read_csv("annotated_concordia.tsv", sep="\t")


mcg = mcg[mcg['Coding'].notna() & (mcg['Coding'].str.strip() != "")]
con = con[con['Coding'].notna() & (con['Coding'].str.strip() != "")]



mcg['Coding'] = mcg['Coding'].str.lower().str.strip()
con['Coding'] = con['Coding'].str.lower().str.strip()

categories = ["complaints", "courses", "student life", "textbooks", "graduate", "administration", "advice", "grades"]


# value_counts() counts occurrences of labels.
# reindex(..., fill_value=0) ensures that missing categories show as zero.
mc_counts = mcg['Coding'].value_counts().reindex(categories, fill_value=0)
co_counts = con['Coding'].value_counts().reindex(categories, fill_value=0)


plt.figure(figsize=(12,6))

# Bar positions for McGill (starting at 0,1,2,...)
plt.bar(range(len(categories)), mc_counts, width=0.4, label="McGill")

# Bar positions for Concordia shifted slightly to the right (+0.4)
plt.bar([i + 0.4 for i in range(len(categories))], co_counts, width=0.4, label="Concordia")

# Set category labels under the bars (centered between the two bars)
plt.xticks([i + 0.2 for i in range(len(categories))], categories, rotation=45)

plt.xlabel("Category")
plt.ylabel("Number of Posts")
plt.title("Topic Distribution: McGill vs. Concordia (After Filtering Missing Labels)")

plt.legend()

# Fix spacing so labels don't get cut off
plt.tight_layout()

# Show the final plot
plt.show()
