import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import pandas as pd



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
    plt.savefig("./plots/paf_fap.svg", dpi=1200)


def get_acc(results, x):
    accuracy_pruned_0 = [results[str(r)]["pruned"]["0"] for r in x]
    accuracy_pruned_1 = [results[str(r)]["pruned"]["1"] for r in x]
    accuracy_pruned_total = np.zeros(len(accuracy_pruned_0))
    for i in range(len(accuracy_pruned_0)):
        accuracy_pruned_total[i] = accuracy_pruned_0[i] if accuracy_pruned_0[i] > accuracy_pruned_1[i] else accuracy_pruned_1[i]

    #accuracy_pruned_total = (np.array(accuracy_pruned_0) + np.array(accuracy_pruned_1))/2

    accuracy_pruned_fused_0 = [results[str(r)]["pruned_and_fused"]["0"] for r in x]
    accuracy_pruned_fused_1 = [results[str(r)]["pruned_and_fused"]["1"] for r in x]

    accuracy_pruned_fused_total = np.zeros(len(accuracy_pruned_0))
    for i in range(len(accuracy_pruned_0)):
        accuracy_pruned_fused_total[i] = accuracy_pruned_fused_0[i] if accuracy_pruned_fused_0[i] > accuracy_pruned_fused_1[i] else accuracy_pruned_fused_1[i]
    #accuracy_pruned_fused_total = (np.array(accuracy_pruned_fused_0) + np.array(accuracy_pruned_fused_1))/2

    accuracy_paf = np.array([results[str(r)]["paf"] for r in x])
    accuracy_fap = np.array([results[str(r)]["fap"] for r in x])

    return accuracy_pruned_total, accuracy_pruned_fused_total, accuracy_paf, accuracy_fap

if __name__ == '__main__':
    with open('./new_results_multiple_0.json', 'r') as f:
            results = json.load(f)
    
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    accuracy_pruned_total, accuracy_pruned_fused_total, accuracy_paf_total,_ = get_acc(results, x)

    with open('./new_results_multiple_1.json', 'r') as f:
            results = json.load(f)

    accuracy_pruned_total_0, accuracy_pruned_fused_total_0, accuracy_paf_total_0,_ = get_acc(results, x)

    with open('./new_results_multiple_2.json', 'r') as f:
            results = json.load(f)
    accuracy_pruned_total_1, accuracy_pruned_fused_total_1, accuracy_paf_total_1, accuracy_fap_total_1 = get_acc(results, x)

    with open('./new_results_multiple_3.json', 'r') as f:
            results = json.load(f)
    accuracy_pruned_total_2, accuracy_pruned_fused_total_2, accuracy_paf_total_2, accuracy_fap_total_2 = get_acc(results, x)

    accuracy_pruned_total = (accuracy_pruned_total + accuracy_pruned_total_0 +accuracy_pruned_total_1+accuracy_pruned_total_2)/4
    #accuracy_pruned_fused_total = (accuracy_pruned_fused_total + accuracy_pruned_fused_total_0 + accuracy_pruned_fused_total_1+ accuracy_pruned_fused_total_2)/4
    accuracy_paf_total = (accuracy_paf_total + accuracy_paf_total_0 +accuracy_paf_total_1+accuracy_paf_total_2)/4
    accuracy_fap_total = (accuracy_fap_total_1 + accuracy_fap_total_2)/2

    plot(lines=[accuracy_paf_total-accuracy_pruned_total, accuracy_fap_total-accuracy_pruned_total], x=x, labels=["PaF", "FaP"], xlabel="Sparsity", ylabel="Accuracy Gain over Pruned")


