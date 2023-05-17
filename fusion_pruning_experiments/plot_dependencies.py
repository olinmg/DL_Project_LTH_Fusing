import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch

sns.set_theme(style="white")

# Load the brain networks dataset, select subset, and collapse the multi-index
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns
                  .get_level_values("network")
                  .astype(int)
                  .isin(used_networks))
df = df.loc[:, used_columns]

df.columns = df.columns.map("-".join)

# Compute a correlation matrix and convert to long-form
corr_mat = df.corr().stack().reset_index(name="correlation")

# Draw each cell as a scatter point with varying size and color


data = torch.eye(12, 12)
for i in range(12):
    for j in range(12):
        data[i, j] = (i == j or i == j+1)

"""for i in range(12):
    for j in range(12):
        data["layer"].append(i)
        data["dependency"].append(j)
        data["true"].append(int(i==j or j == (i+1)))
print(corr_mat)"""
data = pd.DataFrame(data)
cmap = sns.diverging_palette(230, 20, as_cmap=True)

f, ax = plt.subplots(figsize=(11, 9))
glue = sns.load_dataset("glue").pivot("Model", "Task", "Score")
print(glue)
g = sns.heatmap(data, vmin=-0.5, vmax=1.5,cbar=False, cmap="Blues")

#g.show()
f.savefig("./plots/whatever.svg", dpi=1200)