import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


"""
plot creates a plot with arbitrary many lines

lines: list of y-values of your lines, e.g. [[0.1, 0.3, 0.2], [0.5, 1.3, 1.9]]
x: a list containing the values on the x-axis, e.g. [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] (They will be shown as percentages in the final plot)
labels: A list containing the labels of your lines, e.g. ["PaF", "FaP"]
xlabel: The label of the x-axis
ylabel: The label of the y-axis
"""
def plot(lines, x, labels, xlabel, ylabel):
    sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    labelsize=20
    plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=labelsize)    # legend fontsize
    plt.rc('font', size=labelsize)          # controls default text sizes
    plt.rc("lines", markersize=10)
    plt.rc("lines", linewidth=4)

    
    plt.figure(figsize=(10,6), tight_layout=True)

    dict = {}
    for i in range(len(lines)):
        dict[labels[i]] = lines[i]
    dict[xlabel] = x


    data = pd.DataFrame(dict)

    ax = sns.lineplot(data=pd.melt(data, [xlabel]), x=xlabel, y="value", hue="variable", marker = "o")
    ax.set_yticklabels(['{:,.1%}'.format(y) for y in ax.get_yticks()])
    ax.set_xticklabels(['{:,.0%}'.format(spa) for spa in ax.get_xticks()])
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(title_fontsize = 13)
    #plt.show()
    plt.savefig("./plots/example.svg", dpi=1200)


if __name__ == '__main__':
    with open('./results_big_SSF.json', 'r') as f:
                results = json.load(f)
    
    results = results["results"]
 
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    accuracy_pruned = np.zeros(len(x))
    accuracy_SSF = np.zeros(len(x))

    for idx, result in enumerate(results):
        accuracy_pruned[idx] = result["vgg11"]["model_0"]["accuracy_pruned"]["76"]
        accuracy_SSF[idx] = result["vgg11"]["model_0"]["accuracy_pruned_and_fused"]["76"]
    
    with open('./results_big_SSF_more_training.json', 'r') as f:
                results = json.load(f)
    
    results = results["results"]
    for idx, result in enumerate(results):
        idx+=6
        if idx > 8:
            break

        accuracy_pruned[idx] = result["vgg11"]["model_0"]["accuracy_pruned"]["151"]

    plot([accuracy_pruned, accuracy_SSF], x, ["Pruned", "SSF"], "Sparsity", "Accuracy")

