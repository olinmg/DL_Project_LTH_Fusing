The GitRepo for the project of the Deep Learning class at ETH Zurich.

### File Explanations

1. base_convNN.py: Call it to train two MLPs or two CNNs. Networks are saved in the models folder. See Sample Commands below.
2. main.py: Call it to fuse two models, whose networks are saved in the models folder. The accuracy will be printed in the terminal. See Sample Commands below.
3. fusion.py: Takes two models as input and outputs the fused model.

4. results_and_plots: contains the results of the ran experiments and the code to create plots from them
    - plot_non_retrained.py: run this to generate the plots shown in the paper for FaP and PaF at different sparsities and fusion importances. The experiment results will be sourced from "fullDict_results_non_retrained" and the resulting plots will be stored to "plots_non_retrained".
    - plot_retrained.py: run this to generated the plots shown in the paper for FaP and PaF with retraining. The experiment results will be sourced from "fullDict_results_non_retrained" and the resulting plots will be stored to "plots_non_retrained".
    
### Sample Commands

#### Training two MLPs
Results will be saved in the models folder.
```
python base_convNN.py --num-models 2 --gpu-id 0 --model-name mlp
```

#### Training two CNNs
Results will be saved in the models folder.
```
python base_convNN.py --num-models 2 --gpu-id 0 --model-name cnn
```

#### Fusing two MLPs
Trained networks have to be in the models folder.
```
python main.py --num-models 2 --gpu-id 0 --model-name mlp
```

#### Fusing two CNNs
Trained networks have to be in the models folder.
```
python main.py --num-models 2 --gpu-id 0 --model-name cnn
```

### Sample workflow for fusing two MLPs
1. Train two MLPs. The result will be saved in the models folder.
```
python base_convNN.py --num-models 2 --gpu-id 0 --model-name mlp
```

2. Fuse two MLPs. The accuracy of the individual models and the fused model will be printed to the terminal.
```
python main.py --num-models 2 --gpu-id 0 --model-name mlp
```
