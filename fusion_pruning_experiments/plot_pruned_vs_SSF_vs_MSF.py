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
    plt.savefig("./plots/fusion_alternative_original_pruned_fused.svg", dpi=1200)

def plot_new():
    with open('./new_results_multiple_0.json', 'r') as f:
            results = json.load(f)
    
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    accuracy_pruned_total, accuracy_pruned_fused_total = get_acc(results, x)

    with open('./new_results_multiple_1.json', 'r') as f:
            results = json.load(f)

    accuracy_pruned_total_0, accuracy_pruned_fused_total_0 = get_acc(results, x)

    with open('./new_results_multiple_1.json', 'r') as f:
            results = json.load(f)
    accuracy_pruned_total_1, accuracy_pruned_fused_total_1 = get_acc(results, x)

    accuracy_pruned_total = (accuracy_pruned_total + accuracy_pruned_total_0 +accuracy_pruned_total_1)/3
    accuracy_pruned_fused_total = (accuracy_pruned_fused_total + accuracy_pruned_fused_total_0 + accuracy_pruned_fused_total_1)/3

    difference = accuracy_pruned_fused_total - accuracy_pruned_total
    print(np.max(difference))


    plot(lines=[accuracy_pruned_total, accuracy_pruned_fused_total], x=x, labels=["Pruned", "SSF"], xlabel="Sparsity", ylabel="Accuracy")

    plt.show()

def get_all(files, x):
    accuracy_pruned_and_fused = np.zeros(len(x))
    accuracy_pruned = np.zeros(len(x))
    accuracy_pruned_and_fused_multi = np.zeros(len(x))
    for file in files:
        print("ITerating")
        with open(file, 'r') as f:
            params = json.load(f)
        
        results = params["results"]
        experiment = params["experiment_parameters"]

        accuracy_pruned_and_fused += np.array([r["vgg11"]["model_1"]["accuracy_pruned_and_fused"]["0"] for r in results])
        accuracy_pruned += np.array([r["vgg11"]["model_1"]["accuracy_pruned"]["0"] for r in results])
        accuracy_pruned_and_fused_multi += np.array([r["vgg11"]["model_1"]["accuracy_pruned_and_fused_multiple_sparsities"]["0"] for r in results])

        accuracy_pruned_and_fused += np.array([r["vgg11"]["model_0"]["accuracy_pruned_and_fused"]["0"] for r in results])
        accuracy_pruned += np.array([r["vgg11"]["model_0"]["accuracy_pruned"]["0"] for r in results])
        accuracy_pruned_and_fused_multi += np.array([r["vgg11"]["model_0"]["accuracy_pruned_and_fused_multiple_sparsities"]["0"] for r in results])
    
    return accuracy_pruned/len(2*files), accuracy_pruned_and_fused/len(2*files), accuracy_pruned_and_fused_multi/len(2*files)




if __name__ == '__main__':
    with open('./results_multi_sparsity.json', 'r') as f:
            params = json.load(f)
    
    results = params["results"]
    experiment = params["experiment_parameters"]

    num_epochs = experiment["num_epochs"]

    #fig, ax = plt.subplots()

    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    accuracy_pruned, accuracy_pruned_and_fused, accuracy_pruned_and_fused_multi = get_all(['./results_multi_sparsity.json', './results_multi_sparsity_1.json', './results_multi_sparsity_2.json'], x)
    print((accuracy_pruned, accuracy_pruned_and_fused, accuracy_pruned_and_fused_multi))
    plot(lines=[accuracy_pruned, accuracy_pruned_and_fused, accuracy_pruned_and_fused_multi], x=x, labels=["Pruned","SSF", "MSF"], xlabel="Sparsity", ylabel="Accuracy")

    """
    accuracy_multi_over_pruned = []
    for i in range(len(accuracy_pruned)):
        accuracy_multi_over_pruned.append(accuracy_pruned_and_fused_multi[i] - accuracy_pruned[i])
    
    ax.plot(x, accuracy_multi_over_pruned, label="Performance gain over pruning")

    accuracy_multi_over_pruned_fused = []
    for i in range(len(accuracy_pruned)):
        accuracy_multi_over_pruned_fused.append(accuracy_pruned_and_fused_multi[i] - accuracy_pruned_and_fused[i])
    ax.plot(x, accuracy_multi_over_pruned_fused, label="Performance gain over pruning and fusing")"""

    #vals = ax.get_yticks()
    #ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

    """
    accuracy_pruned_and_fused = [r["vgg11"]["accuracy_PaF_all"]["0"] for r in results]
    ax.plot(x, accuracy_pruned_and_fused, label="accuracy_Paf_all")"""



    """

    result = results[int(len(results)/2)+3]
    sparsity = result["sparsity"]
    print(sparsity)
    prune_type = result["prune_type"]
    print(prune_type)
    model_name = experiment["models"][0]["name"]

    result = result[model_name]


    plot(ax, result["accuracy_PaF"], num_epochs, label="PaF")
    plot(ax, result["accuracy_FaP"], num_epochs, label="FaP")
    plot(ax, result["model_0"]["accuracy_pruned"], num_epochs, label="Pruned")
    plot(ax, result["model_0"]["accuracy_pruned_and_fused"], num_epochs, label="Pruned&Fused")"""



    



    