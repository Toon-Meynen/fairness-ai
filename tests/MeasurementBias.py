import scipy.stats
import Bias

if __name__ == "__main__":
    model = Bias.sample_model()
    bias = Bias.DataGenerator(data_generator=model, protected_attributes=["gender"], labels=["income"])

    data = bias.simulate(100_000)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.MeasurementBiasGenerator(parameter="gender", parameter_value=0, measurement="age", bias_strength=1)
    data = bias.apply(data)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.MeasurementBiasGenerator(parameter="gender", parameter_value=1, measurement="age", weight={"invalid_ratio": 0.1}, bias_strength=1)
    data = bias.apply(data)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.MeasurementBiasGenerator(parameter="gender", parameter_value=0, measurement="age", weight={"measurement_error": scipy.stats.gamma(1)}, bias_strength=1)
    data = bias.apply(data)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    bias = Bias.MeasurementBiasGenerator(parameter="gender", parameter_value=0, measurement="age", weight={"measurement_error": [0.33, 0.67]}, bias_strength=1)
    data = bias.apply(data)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())