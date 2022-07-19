import copy

from Data import Data
from DataGenerator import DataGenerator
import scipy.stats
import numpy as np


class SimpsonsParadoxGenerator:
    def __init__(self, base_model, groups, protected_attributes, labels, group_label="group", force_paradox=False): # fiddle with weights
        super().__init__()
        self._prot = protected_attributes
        self._lab = labels

        for ind, g in enumerate(groups):
            if "weight" not in g:
                g["weight"] = 1
            if "name" not in g:
                g["name"] = ind
            if "model" not in g:
                g["model"] = copy.deepcopy(base_model)
                for p in g["probabilities"]:
                    g["model"].addProbability(*p)

        self._total_weight = 0
        for g in groups:
            self._total_weight += g["weight"]

        self._generators = [
            DataGenerator(data_generator=g["model"], protected_attributes=protected_attributes, labels=labels)
            for g in groups]
        self.groups = groups
        self.group_label = group_label

    # generates n items
    def simulate(self, n=10):
        # n_i of each generator
        datas = []
        for ind, val in enumerate(self._generators):
            j = int(self.groups[ind]["weight"] / self._total_weight * n)
            datas.append(val.simulate(j))
            datas[-1].df()[self.group_label] = self.groups[ind]["name"]
        # combine objects into one big object
        import pandas as pd
        data = Data(pd.concat(d.df() for d in datas), self._prot, self._lab)

        return data
