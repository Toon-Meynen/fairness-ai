from DataGenerator import BiasGenerator
import pgmpy.models


class CauseEffectBiasGenerator(BiasGenerator):
    def __init__(self, new_parameter, correlated_parameter, probability=1, consecutive=False):
        super().__init__()
        self.new_p = new_parameter
        self.cor_p = correlated_parameter
        self.P = probability
        self.C = consecutive

    def apply(self, data):
        if self.C:
            # https://stackoverflow.com/questions/59862619/how-can-i-select-a-sequence-of-random-rows-from-a-pandas-dataframe
            random_position = data.df().sample(1).index
            no_consecutives = self.P * len(data.df())
            if random_position + no_consecutives > len(data.df()):
                random_position = len(data.df()) - no_consecutives

            subset = data.df().loc[random_position:random_position + no_consecutives]
        else:
            subset = data.df().sample(frac=self.P)

        model = pgmpy.models.BayesianNetwork()
        model.fit(data.df())
        print(model.simulate(10))

        return data
