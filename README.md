The GitRepo for the project of the Deep Learning class at ETH Zurich.

Our team decided to explore the performance of a network that is initialized by a fusion of winning lottery tickets of trained networks N1, N2.
How will this perform when compared to the fusion of non-pruned N1, N2?

### File Explanations

1. base_convNN.py: Call it to train two MLPs or two CNNs. Networks are saved in the models folder. See Sample Commands below.
2. main.py: Call it to fuse two models, whose networks are saved in the models folder. The accuracy will be printed in the terminal. See Sample Commands below.
3. fusion.py: Takes two models as input and outputs the fused model.

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
