from DataGenerator import BiasGenerator


class OmittedVariableBiasGenerator(BiasGenerator):
    """
    The omitted variable bias generator removes a column from the dataset or, removes all rows that
    contain a specific value.

    Attributes
    ----------
    parameter_to_omit : str
        The parameter that we want to remove
    pvalue : int (optional)
        When provided, removes all occurrences of this value instead of the entire column of data

    Methods
    -------
    apply(data)
        Applies the bias on a given dataset, returns the biased set.
    """
    def __init__(self, parameter_to_omit, parameter_value=None):
        super().__init__()
        self.parameter_to_omit = parameter_to_omit
        self.pvalue = parameter_value

    def apply(self, data):
        data = data.copy() # ensure original object isn't changed

        # if no value is provided, remove entire column
        if self.pvalue is None:
            return data.remove_variable(self.parameter_to_omit)
        # if value is provided, remove rows that contain said value
        values = data.df(weight=True)[self.parameter_to_omit].unique()
        if len(values) == 0:
            return data
        # if it is the final value of this column, remove the column instead of all the rows
        if len(values) == 1 and values[0] == self.pvalue:
            return data.remove_variable(self.parameter_to_omit)
        else:
            data.df(weight=True).drop(data.df(weight=True).loc[(data.df(weight=True)[self.parameter_to_omit] == self.pvalue)].index, inplace=True)
            return data
