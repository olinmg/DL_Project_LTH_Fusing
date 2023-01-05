import matplotlib.pyplot as plt
import numpy as np
import json



def plot(ax, results, num_epochs, label):
    x_data = [i for i in range(num_epochs+1)]
    y_data = [None]*(num_epochs+1)
    for key, value in results.items():
        y_data[int(key)] = value
    
    ax.plot(x_data, y_data, label=label)



if __name__ == '__main__':
    with open('./results_multi_sparsity.json', 'r') as f:
            params = json.load(f)
    
    results = params["results"]
    experiment = params["experiment_parameters"]

    num_epochs = experiment["num_epochs"]

    x = [i for i in range(num_epochs+1)]

    fig, ax = plt.subplots()

    x = params["experiment_parameters"]["sparsity"]

    accuracy_pruned_and_fused = [r["vgg11"]["model_1"]["accuracy_pruned_and_fused"]["0"] for r in results]
    ax.plot(x, accuracy_pruned_and_fused, label="accuracy_pruned_and_fused")

    accuracy_pruned = [r["vgg11"]["model_1"]["accuracy_pruned"]["0"] for r in results]
    ax.plot(x, accuracy_pruned, label="accuracy_pruned")

    accuracy_pruned_and_fused_multi = [r["vgg11"]["model_1"]["accuracy_pruned_and_fused_multiple_sparsities"]["0"] for r in results]
    ax.plot(x, accuracy_pruned_and_fused_multi, label="accuracy_multi_sparsity")

    """
    accuracy_multi_over_pruned = []
    for i in range(len(accuracy_pruned)):
        accuracy_multi_over_pruned.append(accuracy_pruned_and_fused_multi[i] - accuracy_pruned[i])
    
    ax.plot(x, accuracy_multi_over_pruned, label="Performance gain over pruning")

    accuracy_multi_over_pruned_fused = []
    for i in range(len(accuracy_pruned)):
        accuracy_multi_over_pruned_fused.append(accuracy_pruned_and_fused_multi[i] - accuracy_pruned_and_fused[i])
    ax.plot(x, accuracy_multi_over_pruned_fused, label="Performance gain over pruning and fusing")"""

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])

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
    ax.legend(loc = 'upper right')

    plt.show()



    



    