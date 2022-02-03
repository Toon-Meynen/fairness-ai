import scipy.stats

import DataGenerator

if __name__ == "__main__":
    model = DataGenerator.sample_model()
    bias = DataGenerator.DataGenerator(data_generator=model, protected_attributes=["sex"], labels=["income"])

    data = bias.simulate(100_000)
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    bias = DataGenerator.MeasurementBiasGenerator(parameter="sex", parameter_value=0, measurement="age", bias_strength=1)
    data = bias.apply(data)
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    bias = DataGenerator.MeasurementBiasGenerator(parameter="sex", parameter_value=1, measurement="age", weight={"invalid_ratio": 0.1}, bias_strength=1)
    data = bias.apply(data)
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())

    bias = DataGenerator.MeasurementBiasGenerator(parameter="sex", parameter_value=0, measurement="age", weight={"measurement_error": scipy.stats.gamma(1)}, bias_strength=1)
    data = bias.apply(data)
    DataGenerator.test_data(data)
    print(data.metrics().disparate_impact())
