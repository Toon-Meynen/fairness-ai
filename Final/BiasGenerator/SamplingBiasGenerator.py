import numpy as np

from DataGenerator import BiasGenerator


class SamplingBiasGenerator(BiasGenerator):
    """
    The sampling bias generator will remove rows based on four parameters. First it will only select
    rows for which parameter == pvalue, of these rows bias_strength determines how many rows are dropped.
    This dropping is weighted by the weight dictionary. Thus for each row where parameter == pvalue there
    is a probability of (weight / total weight) * bias_strength for this row to be dropped.

    Attributes
    ----------
    bias_strength : float
        indicates the probability for a candidate item to be selected
    parameter : str
        we introduce bias with respect to this parameter
    pvalue : int
        we introduce bais with respect to this value of parameter
    weight : dict
        this dictionary contains a weight for each potential value of the labels in the dataset
        if weights are not provided these are generated based on the value. This weight will indicate
        how likely it is for a row with a certain value to be removed.

    Methods
    -------
    apply(data)
        Applies the bias on a given dataset, returns the biased set.
    """
    def __init__(self, parameter, parameter_value, weight=None, bias_strength=0.3, seed=None):
        super().__init__(seed)

        self.bias_strength = bias_strength  # Probability for items of selected group to be removed
        self.parameter = parameter  # parameter to remove values for
        self.pvalue = parameter_value  # value of parameter to remove values for
        if weight is None:  # optional weight dictionary
            weight = dict()  # example: {label1: {val1: weight, val2: weight}, label2 : {..}
        self.weight = weight  # unfilled weights use value as their weight, if this value is 0, the lowest value above zero is halved and used instead

    def apply(self, data):
        # ensure we don't change the object
        data = data.copy()

        # ensure that a fully functional weight dictionary is present. Fill where empty
        max_w = np.NINF
        total_weight = dict()
        if len(self.weight) == 0:
            label_values = data.label_values()
            for key in label_values:
                total_weight[key] = 0
                if key not in self.weight:
                    self.weight[key] = dict()
                for value in label_values[key]:
                    if value not in self.weight[key]:
                        self.weight[key][value] = 1
                    total_weight[key] += self.weight[key][value]
        elif len(self.weight) == 1:
            for key in self.weight:
                if key not in total_weight:
                    total_weight[key] = 0
                    for value in self.weight[key]:
                        total_weight[key] += self.weight[key][value]
                        if self.weight[key][value] > max_w:
                            max_w = self.weight[key][value]
        else:
            raise NotImplementedError("Can only introduce sample bias with respect to one attribute at a time")

        #print(self.weight, total_weight)
        # drop rows based on weight
        for key in self.weight:
            for value in self.weight[key]:
                # fraction is multiplied by bias_strength to determine the probability for an item to be dropped
                # normalize by dividing by maximum
                #fraction = float(self.weight[key][value]) / float(total_weight[key])
                fraction = float(self.weight[key][value]) / max_w
                # potential flaw: as these happen after one another the later values use already modified data

                f = self.bias_strength * fraction
                #print(f)
                data.df(weight=True).drop(
                    data.df(weight=True).loc[(data.df(weight=True)[self.parameter] == self.pvalue) & (data.df(weight=True)[key] == value)].sample(
                        frac=f, random_state=self.seed).index, inplace=True)
        return data
SelectionBiasGenerator = SamplingBiasGenerator
