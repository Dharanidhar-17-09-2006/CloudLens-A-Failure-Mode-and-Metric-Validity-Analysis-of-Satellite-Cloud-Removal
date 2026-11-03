import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("cloud_removal_project/results/manual_label_summary.csv", index_col=0)
(df.div(df.sum(axis=1), axis=0) * 100).plot(kind="barh", stacked=True, figsize=(8, 4))
plt.xlabel("% of labeled harmful flips"); plt.ylabel("Fill method"); plt.title("Failure types by fill method (n=15 each)")
plt.tight_layout(); plt.savefig("cloud_removal_project/results/figures/taxonomy_by_method.png", dpi=300)