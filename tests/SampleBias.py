import DataGenerator

if __name__ == "__main__":
    model = DataGenerator.sample_model()
    generator = DataGenerator.DataGenerator(data_generator=model, protected_attributes=["sex"], labels=["income"])
    data = generator.simulate(100_000)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = DataGenerator.SamplingBiasGenerator(parameter="sex", parameter_value=0)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = DataGenerator.SamplingBiasGenerator(parameter="sex", parameter_value=1)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = DataGenerator.SamplingBiasGenerator(parameter="sex", parameter_value=0, weight={"income": {1: 5}}, bias_strength=0.5)
    data = bias.apply(data)
    print(len(data.df()))
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())
