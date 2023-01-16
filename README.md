### Folder Explanations

1. experiments:

2. results_and_plots: contains the results of the ran experiments and the code to create plots from them
    - plot_non_retrained.py: run this to generate the plots shown in the paper for FaP and PaF at different sparsities and fusion importances. The experiment results will be sourced from "fullDict_results_non_retrained" and the resulting plots will be stored to "plots_non_retrained".
    - plot_retrained.py: run this to generated the plots shown in the paper for FaP and PaF with retraining. The experiment results will be sourced from "fullDict_results_retrained" and the resulting plots will be stored to "plots_retrained".
    
### How to Train Models
#### Training two VGG11s using different weight initializations
Train.py file is located in the experiments folder. Run the following commands when you are in the experiments folder. Results will be saved in the models folder.
```
python train.py --num-models 2 --gpu-id 0 --model-name vgg11 --diff-weight-init
```
#### Training two VGG11s using same weight initializations but different data
```
python train.py --num-models 2 --gpu-id 0 --model-name vgg11
```
Have a look at the parameters.py file in order to see all available training arguments.

### Recreate Figure 1: Data-Free Processing comparison between MSF, SSF, and the original pruned model
Set experiment_parameters.json to:
<br />
{
    "FaP": false,
    "PaF": false,
    "PaF_all": false,
    "SSF": true,
    "MSF": true,
    "models": [
        {
            "name": "vgg11"
    }],
    "prune_type": ["l1"],
    "sparsity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "num_models": 1,
    "num_epochs": 0,
    "diff_weight_init": true,
    "gpu_id": 0,
    "dataset": "cifar"
}<br />
Models were trained using learning rate = 0.05, Optimizer = SGD, momentum=0.9. Make sure your model is located in the models folder and has name: vgg11_diff_weight_init_True_0.pth<br />
Run:
```
python performance_tester.py
```
Results will be saved in results.json<br />
### Recreate Figure 2: Data-Aware Processing comparison between SSF, and the original pruned model
Set experiment_parameters.json to:<br />

{
    "FaP": false,
    "PaF": false,
    "PaF_all": false,
    "SSF": true,
    "MSF": false,
    "models": [
        {
            "name": "vgg11"
    }],
    "prune_type": ["l1"],
    "sparsity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "num_models": 1,
    "num_epochs": 75,
    "diff_weight_init": true,
    "gpu_id": 0,
    "dataset": "cifar"
}<br />
Models were trained using the training hyperparameters specified in the appendix. Make sure your model is located in the models folder and has name: vgg11_diff_weight_init_True_0.pth<br />
Run:
```
python performance_tester.py
```
Results will be saved in results.json

### Recreate Figure 4 and 8 to 12: PaF vs FaP vs pruned models when finetuning is disabled
Set experiment_parameters.json to:<br />
{
    "FaP": true,
    "PaF": true,
    "PaF_all": false,
    "models": [
        {
            "name": "vgg11"
    }],
    "prune_type": "l1",
    "sparsity": 0.3,
    "num_models": 2,
    "num_epochs": 0,
    "diff_weight_init": true,
    "gpu_id": -1,
    "dataset": "cifar"
}<br />
In the script performance_tester_varyingWeightsAndSparsity.py it can be adjusted what should be explored. Multiple sparsities or multiple fusion importances (and which ones). In that script it can also be easily set if the different datasubsets ("1" in paper) or the different initialized weights ("2" in paper) should be explored. From the experiments folder run the following to execute the experiments:
```
python performance_tester_varyingWeightsAndSparsity.py
```
This will create performance measurements that are stored in the folder ../results_and_plots/fullDict_results_non_retrained_vgg/. From there the code plot_non_retrained.py (in folder results_and_plots) can be used to generate the plots seen in the paper.


### Recreate Figure 5 or 7: PaF vs FaP vs pruned models when finetuning is enabled
Set experiment_parameters.json to:<br />

{
    "FaP": true,
    "PaF": true,
    "PaF_all": false,
    "SSF": false,
    "MSF": false,
    "models": [
        {
            "name": "vgg11"
    }],
    "prune_type": ["l1"],
    "sparsity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975],
    "num_models": 2,
    "num_epochs": 75,
    "diff_weight_init": true,
    "gpu_id": 0,
    "dataset": "cifar"
}<br />
Models were trained using the training hyperparameters specified in the appendix. Make sure your models located in the models folder and have names: vgg11_diff_weight_init_True_0.pth, vgg11_diff_weight_init_True_1.pth<br />
Run:
```
python performance_tester.py
```
Results will be saved in results.json
