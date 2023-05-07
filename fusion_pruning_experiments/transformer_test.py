import torch
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPT2Model, AutoTokenizer, GPTNeoForCausalLM
import torchmetrics
from tqdm import tqdm
import copy
import numpy as np
import ot

# replace this with any model you like
# popular models include [gpt2, gpt2-large, gpt2-medium, gpt2-xl]
# pythia, gpt-neo etc ...

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.cpu()
# There is also a dropout layer, but in eval mode should be easy to remove 

# We can prune these
# model.transformer.h[i].mlp.c_fc, model.transformer.h[i].mlp.c_proj

### Evaluate
# Typical ways to evaluate (see more https://github.com/EleutherAI/lm-evaluation-harness) 
# include word perplexity, loss, accuracy etc ..
# Here https://arxiv.org/pdf/2301.00774.pdf they also focus on perplexity


def collate_fn(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch])
    targets = torch.roll(input_ids, -1, dims=-1)
    # here we ignore masking as it is already taken care of in the dataset
    targets[:, -1] = -1  # ignore index is set to -1
    return input_ids, targets

# a dataset of wikitext and bookcorpus that is alraedy pre-processed
dataset = load_dataset("sanagnos/processed_gpt_dataset_big", split="train[0:500]")

# This is what a random sample looks like, text with 1-24 tokens
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_histogram(cardinality, indices, typ):

    if typ == "uniform":
        return np.ones(cardinality)/cardinality # uniform probability distribution
    elif typ == "prune":
        result = np.zeros(cardinality)
        for indice in indices:
            result[indice] = cardinality/indices.size()[0]

        return result/np.sum(result)
    else:
        result = np.ones(cardinality)
        for indice in indices:
            result[indice] = cardinality/indices.size()[0]

        return result/np.sum(result)

def normalize_vector(coordinates, eps=1e-9):
    norms = torch.norm(coordinates, dim=-1, keepdim=True)
    return coordinates / (norms + eps)

# compute_euclidian_distance_matrix computes a matrix where c[i, j] corresponds to the euclidean distance of x[i] and y[j]
def compute_euclidian_distance_matrix(x, y, p=1, squared=True): # For some reason TA prefers squared to be Truez
    x_col = x.unsqueeze(1).cpu()
    y_lin = y.unsqueeze(0).cpu()
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    if not squared:
        c = c ** (1/2)
    return c

def pairwise_distances(x, y=None, squared=True):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)


        if not squared:
            print("dont leave off the squaring of the ground metric")
            dist = dist ** (1/2)

        return dist

def normalize_ground_metric(t):
    mean, std, var = torch.mean(t), torch.std(t), torch.var(t)
    t  = (t-mean)/std
    return t

# get_ground_metric computes the cost matrix
# if bias is present, the bias will be appended to the weight matrix and subsequently used to calculate the cost
# cost matrix is based on the weights, not the activations
def get_ground_metric(coordinates1, coordinates2, bias1, bias2):
    print("I am called!")
    if bias1 != None: # and bias2 != None
        assert bias2 != None
        coordinates1 = torch.cat((coordinates1, bias1.view(bias1.shape[0], -1)), 1)
        coordinates2 = torch.cat((coordinates2, bias2.view(bias2.shape[0], -1)), 1)
    coordinates1 = normalize_vector(coordinates1)
    coordinates2 = normalize_vector(coordinates2)


    print("coordinates1 has: ", coordinates1.device)
    coordinates1.detach()
    coordinates2.requires_grad = False
    """print("coordinates1 has: ", coordinates1.requires_grad)
    iter = int(coordinates1.shape[1]/8)
    result = torch.zeros(coordinates1.shape[0], coordinates2.shape[0])
    count = 0
    for i in range(0, coordinates1.shape[1], iter):
        print(f"{i}:{i+iter}")
        result += compute_euclidian_distance_matrix(coordinates1[:, i:i+iter], coordinates2[:, i:i+iter])
        print("----Done -----------")
        count += 1
        if count == 10:
            break
    print("Right before return!!!")"""
    return compute_euclidian_distance_matrix(coordinates1, coordinates2)# + compute_euclidian_distance_matrix(coordinates1[:, 384:], coordinates2[:, 384:])


"""print(model)
exit()"""
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


class PruneType(str):
    RANDOM = "random"
    L1 = "l1"
    L2 = "l2"

meta_prunes = ["prune", "IF"]
with torch.no_grad():
    sparsity = 0.5 # corresponds to x% of the parameters remaining
    prune_type = PruneType.L2 # How the layers should be pruned. See PruneType
    nodes_remaining = int(3072 * (1-sparsity))

    idx_layer = [6]
    for i in range(len(model.transformer.h)):
        if not (i in idx_layer):
            continue
        weight_fc = model.transformer.h[i].mlp.c_fc.weight
        bias_fc = model.transformer.h[i].mlp.c_fc.bias
        weight_proj = model.transformer.h[i].mlp.c_proj.weight

        indices = None
        if prune_type == PruneType.RANDOM:
            indices = torch.randperm(3072)[:nodes_remaining]
        else:
            p = 1 if prune_type == PruneType.L1 else 2
            norm_layer_fc = torch.norm(weight_fc, p=p, dim=0)
            norm_layer_proj = torch.norm(weight_proj, p=p, dim=1)
            norm_layer = (norm_layer_fc + norm_layer_proj)/2
            _, indices = torch.topk(norm_layer, nodes_remaining)
        
        comp_vec = [weight_fc.clone().t(), weight_proj]
        basis_vec = [torch.index_select(weight_fc, 1, indices).t(), torch.index_select(weight_proj, 0, indices)]

        comp_hist = get_histogram(3072, indices=indices, typ = "prune")
        basis_hist = get_histogram(nodes_remaining, indices=None, typ = "uniform")

        M = [get_ground_metric(comp_vec[0].clone(), basis_vec[0].clone(), None, None), get_ground_metric(comp_vec[1].clone().contiguous(), basis_vec[1].clone().contiguous(), None, None)]

        
        M = torch.stack(M)
        M = torch.mean(M, dim=0)
        cpuM = M.data.cpu().numpy() # POT does not accept array on GPU
        T = ot.emd(comp_hist, basis_hist, cpuM)
        T_var = torch.from_numpy(T).cpu().float()

        torch.set_printoptions(threshold=10_000)
        #T_var *= torch.pow(norm_layer[:,None], 10)
        T_var = T_var / T_var.sum(dim=0)

        """print(T_var[:,0])
        exit()"""

        weight_fc_n = weight_fc.t()
        weight_fc_n = torch.matmul(T_var.t(), weight_fc_n.contiguous().view(weight_fc_n.shape[0], -1))
        bias_fc_n = torch.matmul(T_var.t(), bias_fc.contiguous().view(bias_fc.shape[0], -1))
        weight_proj_n = torch.matmul(T_var.t(), weight_proj.contiguous().view(weight_proj.shape[0], -1))


        model.transformer.h[i].mlp.c_fc.weight = torch.nn.Parameter(weight_fc_n.t().contiguous())
        model.transformer.h[i].mlp.c_fc.bias = torch.nn.Parameter(bias_fc_n.view(-1))
        model.transformer.h[i].mlp.c_fc.nf = nodes_remaining

        model.transformer.h[i].mlp.c_proj.weight = torch.nn.Parameter(weight_proj_n)


        """print("weight_fc end is: ", weight_fc[:,1])
        model.transformer.h[i].mlp.c_fc.weight = torch.nn.Parameter(torch.index_select(weight_fc, 1, indices))
        model.transformer.h[i].mlp.c_fc.bias = torch.nn.Parameter(torch.index_select(bias_fc, 0, indices))
        model.transformer.h[i].mlp.c_fc.nf = nodes_remaining

        model.transformer.h[i].mlp.c_proj.weight = torch.nn.Parameter(torch.index_select(weight_proj, 0, indices))"""

def eval_model(model, test_dataloader, max_iters=None):
    model.eval()

    total = 0
    losses = []
    accs = []
    
    device = model.device
    
    for cnt, test_inputs in tqdm(enumerate(test_dataloader)):
        input_ids, targets = test_inputs[0].to(device), test_inputs[1].to(device)
        #print("targets shape: ", targets.shape)

        with torch.no_grad():
            logits = model(input_ids).logits
            perplexity = torchmetrics.text.perplexity.Perplexity(ignore_index=-1).to(device)
            #print("logits is: ", logits.shape)
            loss = perplexity(
                logits, targets
            )

        losses.append(loss.item() * input_ids.shape[0])

        preds = torch.argmax(logits, axis=-1)
        acc = (preds == targets).float().mean().item() * input_ids.shape[0]

        accs.append(acc)
        
        total += input_ids.shape[0]

        if max_iters is not None and cnt >= max_iters:
            break

    return sum(losses) / total, sum(accs) / total


model.cuda();
accuracy = eval_model(model, dataloader)
print(f"Accuracy: {accuracy}")
# some people just report perplexity, maybe we should do the same
# some people also ignore the first tokens in the sequence as these have a smaller context window