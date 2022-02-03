import scipy.stats

import DataGenerator

if __name__ == "__main__":
    model = DataGenerator.sample_model()
    bias = DataGenerator.DataGenerator(data_generator=model, protected_attributes=["sex"], labels=["income"])

    data = bias.simulate(100_000)
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = DataGenerator.SamplingBiasGenerator(parameter="sex", parameter_value=0)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    bias = DataGenerator.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=4)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    bias = DataGenerator.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=3)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    bias = DataGenerator.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=2)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    bias = DataGenerator.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=1)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    bias = DataGenerator.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=0)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

