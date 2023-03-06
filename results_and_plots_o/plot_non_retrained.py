import matplotlib.pyplot as plt
import json
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


"""
plot creates a plot with arbitrary many lines

lines: list of y-values of your lines, e.g. [[0.1, 0.3, 0.2], [0.5, 1.3, 1.9]]
x: a list containing the values on the x-axis, e.g. [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] (They will be shown as percentages in the final plot)
labels: A list containing the labels of your lines, e.g. ["PaF", "FaP"]
xlabel: The label of the x-axis
ylabel: The label of the y-axis
"""
def plot(lines, x, labels, xlabel, ylabel, file_name):
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
    plt.show()


def get_data_per_weight(lookups, dict_entry_name, dict):

    both_models = False

    if dict_entry_name == "original":
        dict_entry_name = "original_model_accuracies"
        both_models = True
    elif dict_entry_name == "original_retrained":
        dict_entry_name = "original_model_retrained_accuracies_list"
        both_models = True
    elif dict_entry_name == "original_pruned":
        dict_entry_name = "original_pruned_models_accuracies"
        both_models = True
    elif dict_entry_name == "PaT":
        dict_entry_name = "PaT_models_epoch_accuracies_list"
        both_models = True
    elif dict_entry_name == "original_fused":
        dict_entry_name = "original_fused_model_accuracy"
    elif dict_entry_name == "FaT":
        dict_entry_name = "FaT_epoch_accuracies"
    elif dict_entry_name == "PaF":
        dict_entry_name = "PaF_model_accuracy"
    elif dict_entry_name == "PaFaP":
        dict_entry_name = "PaFaP_model_accuracy"
    elif dict_entry_name == "PaFaT":
        dict_entry_name = "PaFaT_model_epoch_accuracy"
    elif dict_entry_name == "PaTaF":
        dict_entry_name = "PaTaF_model_accuracy"
    elif dict_entry_name == "PaTaFaT":
        dict_entry_name = "PaTaFaT_model_accuracy"
    elif dict_entry_name == "FaP":
        dict_entry_name = "FaP_model_accuracy"
    elif dict_entry_name == "FaPaT":
        dict_entry_name = "FaPaT_model_epoch_accuracy"
    elif dict_entry_name == "FaTaP":
        dict_entry_name = "FaTaP_model_accuracy"
    elif dict_entry_name == "FaTaPaT":
        dict_entry_name = "FaTaPaT_model_accuracy"

    results_list = []
    if both_models:
        results_list_2 = []
        for weight in lookups:
            weight_key = str(weight)
            dict_data_at_weight = dict[weight_key][dict_entry_name]
            if isinstance(dict_data_at_weight[0], list):
                dict_data_at_weight_0 = max(dict_data_at_weight[0])
                dict_data_at_weight_1 = max(dict_data_at_weight[1])
            else:
                dict_data_at_weight_0 = dict_data_at_weight[0]
                dict_data_at_weight_1 = dict_data_at_weight[1]
            results_list.append(dict_data_at_weight_0)
            results_list_2.append(dict_data_at_weight_1)
        results_list = [results_list, results_list_2]
    else:
        for weight in lookups:
            weight_key = str(weight)
            dict_data_at_weight = dict[weight_key][dict_entry_name]
            if isinstance(dict_data_at_weight, list):
                dict_data_at_weight = max(dict_data_at_weight)
            results_list.append(dict_data_at_weight)

    return results_list, both_models


if __name__ == '__main__':

    #### ONLY THING TO CHANGE TO SWITCH BETWEEN (1) Data and (2) Init FROM PAPER
    which_version = "Data"   # "Data" or "Init" -> different subsets of the data or different weight initializations
    
    ###### Loading the available data into a usable format ############
    folder_retrained_results = "./fullDict_results_non_retrained_vgg"
    target_folder_plots = "./plots_non_retrained"

    fusion_weights = [[round(x, 1), round(1-x,1)] for x in np.linspace(0, 1, 11)]
    #sparsities = [[round(x, 1), round(1-x,1)] for x in np.linspace(0, 1, 11)]
    sparsity_names = [1, 3, 5, 7, 9]
    results_dict = {}
    for this_sparsity_name in sparsity_names:
        # for same data, different initializations: fresh_l1prune_varyingFusionWeights_sparsity01_vgg11_0vs1_diffInit.json
        # for different data, same initializations: fresh_l1prune_varyingFusionWeights_sparsity01_vgg11_0vs1_diffData.json
        LOAD_RESULTS_FILE = folder_retrained_results+"/VGG_withoutRetraining_varyingFusionWeights_sparsity0"+str(this_sparsity_name)+"_0vs1_diff"+which_version+".json"
        with open(LOAD_RESULTS_FILE, "r") as f:
            this_experiment_results = json.load(f)

        data_series_pruned, _ = get_data_per_weight(lookups=fusion_weights, dict_entry_name="original_pruned", dict=this_experiment_results)
        data_series_PaF, _ = get_data_per_weight(lookups=fusion_weights, dict_entry_name="PaF", dict=this_experiment_results)
        data_series_FaP, _ = get_data_per_weight(lookups=fusion_weights, dict_entry_name="FaP", dict=this_experiment_results)

        pruned_max = np.maximum(data_series_pruned[0], data_series_pruned[1])
        excess_FaP = data_series_FaP - pruned_max
        excess_PaF = data_series_PaF - pruned_max

        this_sparsity_dict = {}
        this_sparsity_dict["data_series_pruned_0"] = data_series_pruned[0]
        this_sparsity_dict["data_series_pruned_1"] = data_series_pruned[1]
        this_sparsity_dict["pruned_max"] = pruned_max
        this_sparsity_dict["data_series_PaF"] = data_series_PaF
        this_sparsity_dict["data_series_FaP"] = data_series_FaP
        this_sparsity_dict["excess_FaP"] = excess_FaP
        this_sparsity_dict["excess_PaF"] = excess_PaF

        results_dict[str(this_sparsity_name)] = this_sparsity_dict


    ###### TO GET THE PaF and FaP PLOTS ############
    plot_lines_FaP = []
    plot_lines_PaF = []
    plot_labels_PaF = []
    plot_labels_FaP = []
    for i in sparsity_names:
        plot_lines_FaP.append(results_dict[str(i)]["excess_FaP"])   # creating [FaP_s01, FaP_s03, FaP_s05, FaP_s07, FaP_s09]
        plot_lines_PaF.append(results_dict[str(i)]["excess_PaF"])   # creating [PaF_s01, PaF_s03, PaF_s05, PaF_s07, PaF_s09]
        plot_labels_FaP.append("FaP_s0"+str(i))
        plot_labels_PaF.append("PaF_s0"+str(i))
    x_step_labels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]    
    # plot the FaP results
    plot(lines=plot_lines_FaP, x=x_step_labels, labels=plot_labels_FaP, ylabel="Accuracy Gain over Pruned", xlabel="Fusion Weight Towards First Model", file_name=target_folder_plots+"/VGG_withoutRetraining_diff"+which_version+"_varyingWeights_sparsityLines_FaP_TEST.svg")
    # plot the PaF results
    plot(lines=plot_lines_PaF, x=x_step_labels, labels=plot_labels_PaF, ylabel="Accuracy Gain over Pruned", xlabel="Fusion Weight Towards First Model", file_name=target_folder_plots+"/VGG_withoutRetraining_diff"+which_version+"_varyingWeights_sparsityLines_PaF_TEST.svg")


    ###### TO GET THE PaF vs FaP PLOTS #############
    best_FaP_per_sparsity = []
    best_PaF_per_sparsity = []
    for j in sparsity_names:
        best_FaP_per_sparsity.append(max(results_dict[str(j)]["excess_FaP"]))
        best_PaF_per_sparsity.append(max(results_dict[str(j)]["excess_PaF"]))
    # plot the FaP vs PaF
    plot(lines=[best_FaP_per_sparsity, best_PaF_per_sparsity], x=[0.1, 0.3, 0.5, 0.7, 0.9], labels=["FaP", "PaF"], ylabel="Accuracy Gain over Pruned", xlabel="Sparsity", file_name=target_folder_plots+"/VGG_withoutRetraining_diff"+which_version+"_varyingSparsities_PaFvsFaP_excessOriginal_TEST.svg")

    
    ###### LOAD: "trained on all data"-model
    # FOR diffData: also need load the "Trained_on_all_data" network -> take the pruned versions of this as "baseline" for the direct PaF vs FaP comparison
    if which_version == "Data":
        with open(folder_retrained_results+"/VGG_withoutRetraining_diffData_varyingSparsities_allData.json", "r") as f:
            trained_on_all_data = json.load(f)

        all_data_model_pruned = [trained_on_all_data["0.1"], trained_on_all_data["0.3"], trained_on_all_data["0.5"], trained_on_all_data["0.7"], trained_on_all_data["0.9"]]
        pruned_max_partialData = [results_dict["1"]["pruned_max"][0], results_dict["3"]["pruned_max"][0], results_dict["5"]["pruned_max"][0], results_dict["7"]["pruned_max"][0], results_dict["9"]["pruned_max"][0]]

        all_data_model_pruned_excess = np.subtract(all_data_model_pruned, pruned_max_partialData)
        plot(lines=[best_FaP_per_sparsity, best_PaF_per_sparsity, all_data_model_pruned_excess], x=[0.1, 0.3, 0.5, 0.7, 0.9], labels=["FaP", "PaF", "\"Whole Data\" Model"], ylabel="Accuracy Gain over Pruned", xlabel="Sparsity", file_name=target_folder_plots+"/VGG_withoutRetraining_diff"+which_version+"_varyingSparsities_PaFvsFaPvsAllData_excessOrignal_TEST.svg")
        