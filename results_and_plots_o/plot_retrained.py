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

def this_plot(lines, x, labels, xlabel, ylabel, file_name):
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
    plt.savefig(file_name, dpi=1200)



if __name__ == '__main__':

    # SETTING UP WHAT SHOULD BE PLOTTED
    folder_retrained_results = "./fullDict_results_retrained_vgg"
    target_folder_plots = "./plots_retrained"
    which_one = "diffData"  # "diffData" or "diffInit" -> different subsets of the data or different weight initializations
    only_do_1090 = False  # only plots data for sparsities up to 90%, if false does plot data up to 97.5% sparsity

    LOAD_RESULTS_FILE = folder_retrained_results+'/'+which_one+'_PaF_FaP_10_90_retrain.json'
    with open(LOAD_RESULTS_FILE, 'r') as f:
        results_1090 = json.load(f)

    LOAD_RESULTS_FILE = folder_retrained_results+'/'+which_one+'_PaF_FaP_90_97_retrain.json'
    with open(LOAD_RESULTS_FILE, 'r') as f:
        results_9097 = json.load(f)
    
    results_1090 = results_1090["results"]
    results_9097 = results_9097["results"]
 
    x_1090 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    accuracy_pruned_1090 = np.zeros(len(x_1090))
    accuracy_PaF_1090 = np.zeros(len(x_1090))
    accuracy_FaP_1090 = np.zeros(len(x_1090))
    for idx, result in enumerate(results_1090):
        accuracy_pruned_1090[idx] = result["vgg11"]["model_0"]["accuracy_pruned"]["151"] if result["vgg11"]["model_0"]["accuracy_pruned"]["151"] > result["vgg11"]["model_1"]["accuracy_pruned"]["151"] else result["vgg11"]["model_1"]["accuracy_pruned"]["151"]
        accuracy_PaF_1090[idx] = result["vgg11"]["accuracy_PaF"]["76"]
        accuracy_FaP_1090[idx] = result["vgg11"]["accuracy_FaP"]["76"]
    
    if which_one == "diffInit":
        x_9097 = [0.9, 0.925, 0.95, 0.975]
    else:
        x_9097 = [0.925, 0.95, 0.975]

    accuracy_pruned_9097 = np.zeros(len(x_9097))
    accuracy_PaF_9097 = np.zeros(len(x_9097))
    accuracy_FaP_9097 = np.zeros(len(x_9097))
    for idx, result in enumerate(results_9097):
        accuracy_pruned_9097[idx] = result["vgg11"]["model_0"]["accuracy_pruned"]["151"] if result["vgg11"]["model_0"]["accuracy_pruned"]["151"] > result["vgg11"]["model_1"]["accuracy_pruned"]["151"] else result["vgg11"]["model_1"]["accuracy_pruned"]["151"]
        accuracy_PaF_9097[idx] = result["vgg11"]["accuracy_PaF"]["76"]
        accuracy_FaP_9097[idx] = result["vgg11"]["accuracy_FaP"]["1"]

    if only_do_1090:
        total_accuracy_pruned = accuracy_pruned_1090
        total_accuracy_PaF = accuracy_PaF_1090
        total_accuracy_FaP = accuracy_FaP_1090
        total_x = x_1090
    else:
        total_accuracy_pruned = np.append(accuracy_pruned_1090, accuracy_pruned_9097)
        total_accuracy_PaF = np.append(accuracy_PaF_1090, accuracy_PaF_9097)
        total_accuracy_FaP = np.append(accuracy_FaP_1090, accuracy_FaP_9097) #accuracy_FaP_1090
        x_1090.extend(x_9097)
        total_x = x_1090
        

    ### ALSO TRAINED MODEL ON ALL OF THE DATA
    if which_one == "diffData":
        with open(folder_retrained_results+"/diffData_allData_10_97_retrain.json") as f:
            results_allData_10_97 = json.load(f)
        results_allData_10_97 = results_allData_10_97["results"]
        accuracy_allData_pruned_1097 = np.zeros(len(x_1090))
        accuracy_allData_pruned_1097_beforeRetrain = np.zeros(len(x_1090))
        for idx, result in enumerate(results_allData_10_97):
            if idx >= len(x_1090):
                break
            accuracy_allData_pruned_1097[idx] = result["vgg11"]["model_0"]["accuracy_pruned"]["151"]
        accuracy_allData_excess_original = accuracy_allData_pruned_1097 - total_accuracy_pruned
        accuraccy_PaF_excess_allData = total_accuracy_PaF - accuracy_allData_pruned_1097

    # IN CASE WANT EXCESS PERFORMANCE OVER INDIVIDUAL PRUNED MODELS
    accuracy_FaP_excess_original = total_accuracy_FaP - total_accuracy_pruned
    accuracy_PaF_excess_original = total_accuracy_PaF - total_accuracy_pruned
    
    if only_do_1090:
        explored_range = "10to90"
    else:
        explored_range = "10to97"


    if which_one == "diffInit":
        this_plot([total_accuracy_FaP, total_accuracy_PaF, total_accuracy_pruned], total_x, ["FaP", "PaF", "Best Original"], "Sparsity", "Accuracy", file_name = target_folder_plots+"/VGG_withRetraining_diffInit_FaPvsPaFvsOriginal_"+explored_range+"_TEST.svg")
        this_plot([accuracy_FaP_excess_original, accuracy_PaF_excess_original], total_x, ["FaP", "PaF"], "Sparsity", "Accuracy", file_name = target_folder_plots+"/VGG_withRetraining_diffInit_FaPvsPaF_excessOriginal_"+explored_range+"_TEST.svg")
    else:   # which_one=="diffData"
        this_plot([total_accuracy_FaP, total_accuracy_PaF, total_accuracy_pruned, accuracy_allData_pruned_1097], total_x, ["FaP", "PaF", "Best Original", "\"Whole Data\" Model"], "Sparsity", "Accuracy", file_name = target_folder_plots+"/VGG_withRetraining_diffData_FaPvsPaFvsOriginalvsAllData_"+explored_range+"_TEST.svg")
        this_plot([accuracy_PaF_excess_original, accuracy_allData_excess_original], total_x, ["PaF", "\"Whole Data\" Model"], "Sparsity", "Accuracy Gain over Pruned", file_name = target_folder_plots+"/VGG_withRetraining_diffData_PaFvsAllData_excessOriginal_"+explored_range+"_TEST.svg")
        this_plot([accuraccy_PaF_excess_allData], total_x, ["PaF"], "Sparsity", "Accuracy Gain over \"Whole Data\"", file_name = target_folder_plots+"/VGG_withRetraining_diffData_PaF_excessAllData_"+explored_range+"_TEST.svg")
    


