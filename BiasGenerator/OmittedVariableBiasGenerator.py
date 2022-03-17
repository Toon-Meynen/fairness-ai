from DataGenerator import BiasGenerator


class OmittedVariableBiasGenerator(BiasGenerator):
    def __init__(self, parameter_to_omit, parameter_value=None):
        super().__init__()
        self.parameter_to_omit = parameter_to_omit
        self.pvalue = parameter_value

    def apply(self, data):
        # if no value is provided, remove entire column
        if self.pvalue is None:
            return data.remove_variable(self.parameter_to_omit)
        # if value is provided, remove rows that contain said value
        values = data.df()[self.parameter_to_omit].unique()
        if len(values) == 0:
            return data
        # if it is the final value of this column, remove the column instead of all the rows
        if len(values) == 1 and values[0] == self.pvalue:
            return data.remove_variable(self.parameter_to_omit)
        else:
            data.df().drop(data.df().loc[(data.df()[self.parameter_to_omit] == self.pvalue)].index, inplace=True)
            return data
