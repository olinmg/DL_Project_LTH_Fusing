import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import pandas as pd



def plot(ax, results, num_epochs, label):
    x_data = [i for i in range(num_epochs+1)]
    y_data = [None]*(num_epochs+1)
    for key, value in results.items():
        y_data[int(key)] = value
    
    ax.plot(x_data, y_data, label=label)

def get_acc(results, x):
    accuracy_pruned_0 = [results[str(r)]["pruned"]["0"] for r in x]
    accuracy_pruned_1 = [results[str(r)]["pruned"]["1"] for r in x]
    accuracy_pruned_total = (np.array(accuracy_pruned_0) + np.array(accuracy_pruned_1))/2

    accuracy_pruned_fused_0 = [results[str(r)]["pruned_and_fused"]["0"] for r in x]
    accuracy_pruned_fused_1 = [results[str(r)]["pruned_and_fused"]["1"] for r in x]
    accuracy_pruned_fused_total = (np.array(accuracy_pruned_fused_0) + np.array(accuracy_pruned_fused_1))/2

    accuracy_paf = [results[str(r)]["paf"] for r in x]
    accuracy_fap = [results[str(r)]["fap"] for r in x]

    return accuracy_pruned_total, accuracy_pruned_fused_total

def plot(lines, x, labels, xlabel, ylabel):
    sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    labelsize=13
    plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=labelsize)    # legend fontsize
    plt.rc('font', size=labelsize)          # controls default text sizes
    plt.rc("lines", markersize=7)
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
    plt.savefig("./plots/fusion_alternative_original_pruned_fused.svg", dpi=1200)

def get_all(files, x):
    accuracy_pruned = np.zeros(len(x))
    accuracy_pruned_fused = np.zeros(len(x))
    for file in files:
        print("ITerating")
        with open(file, 'r') as f:
            results = json.load(f)
        
        accuracy_pruned_0, accuracy_pruned_fused_0 = get_acc(results, x)

        accuracy_pruned += accuracy_pruned_0
        accuracy_pruned_fused += accuracy_pruned_fused_0 
    return accuracy_pruned/len(files), accuracy_pruned_fused/len(files)

def get_all_alternative(files, x):
    accuracy_alternative = np.zeros(len(x))
    accuracy_orig = np.zeros(len(x))
    for file in files:
        print("ITerating")
        with open(file, 'r') as f:
            results = json.load(f)

            accuracy_alternative += np.array([results[str(sparsity)]["accuracy_fused"]for sparsity in x])[::-1]
            accuracy_orig += np.array([results[str(sparsity)]["accuracy_original"] for sparsity in x])[::-1]
    
    return accuracy_alternative/len(files), accuracy_orig/len(files)


def plot_new():
    sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    labelsize=10
    plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=labelsize)    # legend fontsize
    plt.rc('font', size=labelsize)          # controls default text sizes
    with open('./new_results_multiple_0.json', 'r') as f:
            results = json.load(f)
    
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    accuracy_pruned_total, accuracy_pruned_fused_total = get_all(['./new_results_multiple_0.json', './new_results_multiple_1.json', './new_results_multiple_2.json'], x)

    accuracy_alternative, accuracy_orig = get_all_alternative(["./results_fusion_alternative_0.json", "./results_fusion_alternative_1.json", "./results_fusion_alternative_2.json", "./results_fusion_alternative_3.json"], x)

    plot([accuracy_pruned_total, accuracy_orig, accuracy_alternative, accuracy_pruned_fused_total], x, ["Pruned","Original", "Fused", "SSF"], "Sparsity", "Accuracy")

    exit()
    """
    
    accuracy_pruned_total, accuracy_pruned_fused_total = get_acc(results, x)

    with open('./new_results_multiple_1.json', 'r') as f:
            results = json.load(f)

    accuracy_pruned_total_0, accuracy_pruned_fused_total_0 = get_acc(results, x)

    with open('./new_results_multiple_1.json', 'r') as f:
            results = json.load(f)
    accuracy_pruned_total_1, accuracy_pruned_fused_total_1 = get_acc(results, x)

    accuracy_pruned_total = (accuracy_pruned_total + accuracy_pruned_total_0 +accuracy_pruned_total_1)/3
    accuracy_pruned_fused_total = (accuracy_pruned_fused_total + accuracy_pruned_fused_total_0 + accuracy_pruned_fused_total_1)/3"""

    difference = accuracy_pruned_fused_total - accuracy_pruned_total
    print(np.max(difference))



    print(results)
    
    plt.figure(figsize=(10,6), tight_layout=True)

    data = pd.DataFrame({
        "Sparsity": x,
    'Pruned': accuracy_pruned_total,
    'SSF': accuracy_pruned_fused_total})

    ax = sns.lineplot(data=pd.melt(data, ["Sparsity"]), x="Sparsity", y="value", hue="variable", marker = "o")
    ax.set_yticklabels(['{:,.1%}'.format(y) for y in ax.get_yticks()])
    ax.set_xticklabels(['{:,.0%}'.format(spa) for spa in ax.get_xticks()])
    ax.set(xlabel="Sparsity", ylabel="Accuracy")
    ax.legend(title_fontsize = 13)
    plt.show()
    exit()

    del results["sparstiy"]

    """
    accuracy_pruned_0 = [results[str(r)]["pruned"]["0"] for r in x]
    accuracy_pruned_1 = [results[str(r)]["pruned"]["1"] for r in x]
    accuracy_pruned_total = (np.array(accuracy_pruned_0) + np.array(accuracy_pruned_1))/2

    accuracy_pruned_fused_0 = [results[str(r)]["pruned_and_fused"]["0"] for r in x]
    accuracy_pruned_fused_1 = [results[str(r)]["pruned_and_fused"]["1"] for r in x]
    accuracy_pruned_fused_total = (np.array(accuracy_pruned_fused_0) + np.array(accuracy_pruned_fused_1))/2

    accuracy_paf = [results[str(r)]["paf"] for r in x]
    accuracy_fap = [results[str(r)]["fap"] for r in x]"""

    #ax.plot(x, accuracy_pruned_0 , label="pruned 0")
    #ax.plot(x, accuracy_pruned_1, label="pruned 1")
    #ax.plot(x, accuracy_pruned_fused_0, label="prune&fusing post-proc 0")
    #ax.plot(x, accuracy_pruned_fused_1, label="prune&fusing post-proc 1")
    #ax.plot(x, accuracy_paf, label="PaF")
    #ax.plot(x, accuracy_fap, label="FaP")
    ax.plot(x, accuracy_pruned_total, label="pruned")
    ax.plot(x, accuracy_pruned_fused_total, label="pruned & fused")

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

    ax.legend(loc = 'lower left')

    plt.show()


if __name__ == '__main__':
    """
    with open('./results.json', 'r') as f:
            params = json.load(f)"""
    
    plot_new()
    