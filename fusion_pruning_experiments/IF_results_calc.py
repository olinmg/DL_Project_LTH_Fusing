import torch
import json

def get(pruned, IF):
    print("pruned: ", pruned.mean()*100)
    print("prune var: ", pruned.var()*100)
    print("IF: ", IF.mean()*100)
    print("IF var: ", IF.var()*100)

    print((IF-pruned).var()*100)
    print("Diff: ", (IF.mean() - pruned.mean())*100)
"""
pruned_06 = torch.Tensor([0.9111946202531646, 0.912381329113924, 0.9128757911392406])
IF_06 = torch.Tensor([0.9151503164556962, 0.9155458860759493, 0.9178204113924051])

get(pruned_06, IF_06)

pruned_07 = torch.Tensor([0.9033821202531646, 0.903184335443038, 0.9056566455696202])
IF_07 = torch.Tensor([0.9142602848101266, 0.9139636075949367, 0.9128757911392406])

get(pruned_07, IF_07)

pruned_08 = torch.Tensor([0.8891416139240507, 0.8888449367088608, 0.8864715189873418])
IF_08 = torch.Tensor([0.8963607594936709, 0.8970530063291139, 0.8946795886075949])

get(pruned_08, IF_08)"""

sparsities = ["0.4", "0.6", "0.7", "0.8"]
seeds = ["1", "2"]
types = ["default", "intra-fusion"]

dict = {}
for sparsity in sparsities:
    dict[sparsity] = {
        "default": [],
        "intra-fusion": []
    }

with open('./results_intrafusion_resnet18_L2.json', 'r') as f:
    results = json.load(f)

for seed in seeds:
    for sparsity in sparsities:
        for type in types:
            print(results[seed].keys())
            dict[sparsity][type].append(results[seed]["model_0"][sparsity][type][-1])

for sparsity in sparsities:
    print("Sparsity: ", sparsity)
    get(torch.Tensor(dict[sparsity]["default"]), torch.Tensor(dict[sparsity]["intra-fusion"]))
    print("---------------------")
