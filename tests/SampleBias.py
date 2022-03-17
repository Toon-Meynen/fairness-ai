import Bias

if __name__ == "__main__":
    model = Bias.sample_model()
    generator = Bias.DataGenerator(data_generator=model, protected_attributes=["gender"], labels=["income"])
    data = generator.simulate(100_000)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = Bias.SamplingBiasGenerator(parameter="gender", parameter_value=0)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = Bias.SamplingBiasGenerator(parameter="gender", parameter_value=1)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = Bias.SamplingBiasGenerator(parameter="gender", parameter_value=0, weight={"income": {1: 5}}, bias_strength=0.5)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())
