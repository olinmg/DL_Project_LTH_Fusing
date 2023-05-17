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

"""def plot(dict):
    sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    labelsize=20
    plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=labelsize)    # legend fontsize
    plt.rc('font', size=labelsize)          # controls default text sizes
    plt.rc("lines", markersize=7)
    plt.rc("lines", linewidth=3)

    
    plt.figure(figsize=(10,6), tight_layout=True)


    data = pd.DataFrame(dict)

    ax = sns.lineplot(data=data, x="Neuron Sparsity", y="Accuracy Gain", hue="layer_idx", marker = "o")
    ax.set_xticks(ax.get_xticks()[1::2]); 
    print([spa for spa in ax.get_xticks()])
    print("---------")
    print(ax.get_xticks())
    print(['{:,.0%}'.format(float(spa)) for spa in dict["Neuron Sparsity"][1::2]][:8])
    ax.set_xticklabels(['{:,.0%}'.format(float(spa)) for spa in dict["Neuron Sparsity"][1::2]][:8])
    ax.set_yticklabels(['+{:,.1%}'.format(y) for y in ax.get_yticks()])
    #ax.set_xticklabels(['{:,.0%}'.format(spa) for spa in ax.get_xticks()])

    #ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(title_fontsize = 13)
    #plt.show()
    plt.savefig("./plots/example.svg", dpi=1200)


if __name__ == '__main__':
    with open('./results_IF_per_layer_vgg11.json', 'r') as f:
                results = json.load(f)
        
    dict = {"layer_idx" : [], "Neuron Sparsity": [], "Accuracy Gain": []}


    vgg_ignore_layer = ["5", "6", "7", "8", "9"]
    resnet_ignore_layer = ["1", "2", "5", "8", "11"]
    for layer_idx in results.keys():
        if layer_idx in vgg_ignore_layer:
              continue
        res_layer_idx = results[layer_idx]
        for sparsity in res_layer_idx.keys():
            if sparsity in ["0.9"]:
                 continue
            res_spars = res_layer_idx[sparsity]

            dict["layer_idx"].append(f"Group {layer_idx}")
            dict["Neuron Sparsity"].append(sparsity)
            print(sparsity)
            dict["Accuracy Gain"].append(res_spars["intra-fusion"]["l1"] - res_spars["default"]["l1"])"""

    
def plot(dict):
    sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    labelsize=20
    plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=labelsize)    # legend fontsize
    plt.rc('font', size=labelsize)          # controls default text sizes
    plt.rc("lines", markersize=7)
    plt.rc("lines", linewidth=3)

    
    plt.figure(figsize=(10,6), tight_layout=True)


    data = pd.DataFrame(dict)

    ax = sns.lineplot(data=data, x="Neuron Sparsity", y="Accuracy", hue="Layer", style="Type", style_order=["Intra-Fusion", "Default"], marker = "o")
    ax.set_xticks(ax.get_xticks()[1::2]); 
    print([spa for spa in ax.get_xticks()])
    print("---------")
    print(ax.get_xticks())
    sparsities = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]
    print(['{:,.0%}'.format(float(spa)) for spa in dict["Neuron Sparsity"][1::2]][:8])
    ax.set_xticklabels(['{:,.0%}'.format(float(spa)) for spa in sparsities])
    ax.set_yticklabels(['{:,.1%}'.format(y) for y in ax.get_yticks()])
    #ax.set_xticklabels(['{:,.0%}'.format(spa) for spa in ax.get_xticks()])

    #ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(title_fontsize = 13)
    #plt.show()
    plt.savefig("./plots/example___.svg", dpi=1200)


if __name__ == '__main__':
    with open('./results_IF_per_layer.json', 'r') as f:
                results = json.load(f)
        
    dict = {"Layer" : [], "Neuron Sparsity": [], "Accuracy": [], "Type": []}


    vgg_ignore_layer = ["5", "6", "7", "8", "9"]
    resnet_ignore_layer = ["1", "2", "5", "8", "11"]
    for layer_idx in results.keys():
        if layer_idx not in ["4", "10"]:
              continue
        res_layer_idx = results[layer_idx]
        for sparsity in res_layer_idx.keys():
            if sparsity in ["0.9", "0.85", "0.8", "0.75"]:
                 continue
            res_spars = res_layer_idx[sparsity]
            for type in res_spars.keys():
                if type == "default":
                     type_ = "Default"
                else:
                     type_ = "Intra-Fusion"
                dict["Layer"].append(f"Group {layer_idx}")
                dict["Neuron Sparsity"].append(sparsity)
                dict["Accuracy"].append(res_spars[type]["l2"])
                dict["Type"].append(type_)
    
    plot(dict)

"""def plot(dict):
    sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    labelsize=20
    plt.rc('axes', labelsize=labelsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=labelsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=labelsize)    # legend fontsize
    plt.rc('font', size=labelsize)          # controls default text sizes
    plt.rc("lines", markersize=7)
    plt.rc("lines", linewidth=3)

    
    plt.figure(figsize=(10,6), tight_layout=True)


    data = pd.DataFrame(dict)

    ax = sns.lineplot(data=data, x="Sparsity", y="Accuracy", hue="Type", hue_order=["Intra-Fusion", "Default"],marker = "o")
    print([spa for spa in ax.get_xticks()])
    print("---------")
    print(ax.get_xticks())
    sparsities = ["0.6", "0.7", "0.8"]

    ax.set_xticklabels(['{:,.0%}'.format(float(spa)) for spa in sparsities])
    ax.set_yticklabels(['{:,.1%}'.format(y) for y in ax.get_yticks()])
    #ax.set_xticklabels(['{:,.0%}'.format(spa) for spa in ax.get_xticks()])

    #ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(title_fontsize = 13)
    #plt.show()
    plt.savefig("./plots/example.svg", dpi=1200)

if __name__ == '__main__':
    dict = {"Type": [], "Sparsity": [], "Accuracy": []}

    sparsities = ["0.6", "0.7", "0.8"]

    default_06_spars = [0.9128757911392406, 0.9111946202531646, 0.912381329113924, 0.9112935126582279]
    IF_06_spars = [0.9151503164556962, 0.9155458860759493, 0.9178204113924051, 0.9169303797468354]

    default_07_spars = [0.9033821202531646, 0.903184335443038, 0.9056566455696202, 0.9023931962025317]
    IF_07_spars = [0.9139636075949367, 0.9142602848101266, 0.9128757911392406, 0.9114912974683544]

    default_08_spars = [0.8891416139240507, 0.8888449367088608, 0.8864715189873418]
    IF_08_spars = [0.8970530063291139, 0.8963607594936709, 0.8946795886075949]
    
    default = [default_06_spars, default_07_spars, default_08_spars]
    IF_ = [IF_06_spars, IF_07_spars, IF_08_spars]

    for i, stats in enumerate(default):
         for stat in stats:
            dict["Type"].append("Default")
            dict["Sparsity"].append(sparsities[i])
            dict["Accuracy"].append(stat)
    for i, stats in enumerate(IF_):
         for stat in stats:
            dict["Type"].append("Intra-Fusion")
            dict["Sparsity"].append(sparsities[i])
            dict["Accuracy"].append(stat)

    plot(dict)"""
