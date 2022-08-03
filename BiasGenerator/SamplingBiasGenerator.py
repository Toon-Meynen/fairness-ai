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
        if weight is None:  # optional weight dictionary # TODO: Allow non-label values
            weight = dict()  # example: {label1: {val1: weight, val2: weight}, label2 : {..}
        self.weight = weight  # unfilled weights use value as their weight, if this value is 0, the lowest value above zero is halved and used instead

    def apply(self, data):
        # ensure that a fully functional weight dictionary is present. Fill where empty
        total_weight = dict()
        label_values = data.label_values()
        for key in label_values:
            total_weight[key] = 0
            if key not in self.weight:
                self.weight[key] = dict()
            for value in label_values[key]:
                # fill in missing values
                if value not in self.weight[key]:
                    tmp_value = value
                    # avoid including zero-weight values as these wouldn't receive bias
                    if value == 0:
                        # use half of the lowest non-zero value for these instead
                        tmp_value = min(x for x in label_values[key] if x != 0) / 2.0
                    self.weight[key][value] = tmp_value
                    total_weight[key] += tmp_value
                else:
                    total_weight[key] += self.weight[key][value]
        #print(self.weight)
        # drop rows based on weight
        for key in self.weight:
            for value in self.weight[key]:
                # fraction is multiplied by bias_strength to determine the probability for an item to be dropped
                fraction = float(self.weight[key][value]) / float(total_weight[key])
                # potential flaw: as these happen after one another the later values use already modified data
                data.df().drop(
                    data.df().loc[(data.df()[self.parameter] == self.pvalue) & (data.df()[key] == value)].sample(
                        frac=self.bias_strength * fraction, random_state=self.seed).index, inplace=True)
        return data
SelectionBiasGenerator = SamplingBiasGenerator
