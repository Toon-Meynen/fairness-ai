import Bias

if __name__ == "__main__":
    model = Bias.sample_model()
    bias = Bias.DataGenerator(data_generator=model, protected_attributes=["gender"], labels=["income"])

    data = bias.simulate(100_000)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = Bias.SamplingBiasGenerator(parameter="gender", parameter_value=0)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=4)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=3)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=2)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=1)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.OmittedVariableBiasGenerator(parameter_to_omit="age", parameter_value=0)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

