import abc
import torch
import torch.nn as nn
import numpy as np
import ot
from sklearn.decomposition import PCA

import typing
from .pruner import function
from ._helpers import _FlattenIndexMapping
from . import ops
import math
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.ensemble import IsolationForest


class OptimalTransport:
    def __init__(
        self,
        p=1,
        target_probability="uniform",
        source_probability="uniform",
        target = "most_important",
        coefficients = "importance",
        gpu_id = 0,
    ):
        self.p = p
        self.target_probability = target_probability
        self.source_probability = source_probability
        self.target = target
        self.coefficients = coefficients
        self.gpu_id = gpu_id

    def _normalize(self, cost, normalizer):
        if normalizer is None:
            return cost
        elif isinstance(normalizer, typing.Callable):
            return normalizer(cost)
        elif normalizer == "sum":
            return cost / cost.sum()
        elif normalizer == "standardization":
            return (cost - cost.min()) / (cost.max() - cost.min() + 1e-8)
        elif normalizer == "mean":
            return cost / cost.mean()
        elif normalizer == "max":
            return cost / cost.max()
        elif normalizer == "gaussian":
            return (cost - cost.mean()) / (cost.std() + 1e-8)
        else:
            raise NotImplementedError

    def _probability(self, probability_type, cardinality, importance, keep_idxs=None):
        if probability_type == "uniform":
            return np.ones(cardinality).astype(dtype="float64") / cardinality
        elif probability_type == "importance":
            imp = importance.numpy().astype(dtype="float64")
            return imp / np.sum(imp)
            # return np.exp(imp) / sum(np.exp(imp))
        elif probability_type == "radical":
            result = np.ones(cardinality).astype(dtype="float64")
            for indice in keep_idxs:
                result[indice] = cardinality / len(keep_idxs)
            return result / np.sum(result)
        else:
            raise NotImplementedError

    def _cost(self, weights0, weights1):
        if self.gpu_id != -1:
            weights0 = weights0.cuda(self.gpu_id)
            weights1 = weights1.cuda(self.gpu_id)

        norm0 = torch.norm(weights0, dim=-1, keepdim=True)
        norm1 = torch.norm(weights1, dim=-1, keepdim=True)
        if self.gpu_id != -1:
            norm0 = norm0.cuda(self.gpu_id)
            norm1 = norm1.cuda(self.gpu_id)

        distance = torch.cdist(weights0 / norm0, weights1 / norm1, p=self.p).cpu()

        weights0 = weights0.cpu()
        weights1 = weights1.cpu()
        return self._normalize(distance, "mean").cpu()

    @torch.no_grad()
    def __call__(self, group, importance, pruning_idxs):
        keep_idxs = None
        w_all = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)

            # in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight)[idxs].flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1)[idxs].flatten(1)

            if keep_idxs == None:
                keep_idxs = list(
                    set([i for i in range(w.shape[0])])
                    - set(int(i) for i in pruning_idxs)
                )

            w_all.append(w)

        if len(w_all) == 0:
            return


        w_all = torch.cat(w_all, dim=1)

        cost = None
        if self.target == "most_important":
            cost = self._cost(w_all, w_all[keep_idxs])
        else:
            #gm = KMeans(n_clusters=len(keep_idxs)).fit(w_all)
            gm = GaussianMixture(n_components=len(keep_idxs), random_state=0, covariance_type="spherical").fit(w_all)
            cost = self._cost(w_all, torch.from_numpy(gm.means_).float())
        
        source_prob = self._probability(
            self.source_probability, cost.shape[0], importance, keep_idxs
        )
        target_prob = self._probability(
            self.target_probability, cost.shape[1], importance[keep_idxs], keep_idxs
        )


        ot_map = ot.emd(
                source_prob, target_prob, cost.detach().cpu().numpy()
            ).transpose()

        ot_map = torch.from_numpy(ot_map).float()

        ot_map /= source_prob[None, :]
        ot_map *= self._probability(
            self.coefficients, cost.shape[0], importance, keep_idxs
        )

        ot_map = ot_map / ot_map.sum(dim=0)

        return ot_map.float()
