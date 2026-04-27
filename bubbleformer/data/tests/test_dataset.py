import pytest
from bubbleformer.data import BubbleForecast, TempPredict

@pytest.mark.parametrize("input_fields", [
    ["dfun"],
    ["temperature", "velx", "vely"],
    ["dfun", "temperature", "velx", "vely"]
])
@pytest.mark.parametrize("output_fields", [
    ["temperature"],
    ["temperature", "velx", "vely"],
    ["dfun", "temperature", "velx", "vely"]
])
@pytest.mark.parametrize("norm", ["none", "std", "minmax", "tanh"])
@pytest.mark.parametrize("downsample_factor", [1, 2, 4])
@pytest.mark.parametrize("time_window", [5, 10])
def test_bubbleforecastdataset(
    input_fields,
    output_fields,
    norm,
    downsample_factor,
    time_window
    ):
    """
    Test the BubbleForecast dataset
    The samples are 2 50x64x64 (TxHxW) trajectories
    """
    dataset = BubbleForecast(
        filenames=["samples/sample_1.hdf5", "samples/sample_2.hdf5"],
        input_fields=input_fields,
        output_fields=output_fields,
        norm=norm,
        downsample_factor=downsample_factor,
        time_window=time_window,
        start_time=5
    )
    _, _ = dataset.normalize()
    sample = dataset[0]

    expected_input_shape = (
        time_window,
        len(input_fields),
        64//downsample_factor,
        64//downsample_factor
    )
    expected_output_shape = (
        time_window,
        len(output_fields),
        64//downsample_factor,
        64//downsample_factor
    )

    assert len(dataset) == 2 * (50 - 5 - 2 * time_window + 1)
    assert sample[0].shape == expected_input_shape 
    assert sample[1].shape == expected_output_shape

@pytest.mark.parametrize("norm", ["none", "std", "minmax", "tanh"])
@pytest.mark.parametrize("downsample_factor", [1, 2, 4])
@pytest.mark.parametrize("time_window", [5, 10])
def test_temppredictdataset(norm, downsample_factor, time_window):
    """
    Test the TempPredict dataset
    The samples are 2 50x64x64 (TxHxW) trajectories
    """
    dataset = TempPredict(
        filenames=["samples/sample_1.hdf5", "samples/sample_2.hdf5"],
        norm=norm,
        downsample_factor=downsample_factor,
        time_window=time_window,
        start_time=5
    )
    _, _ = dataset.normalize()
    sample = dataset[0]

    expected_input_shape = (
        time_window,
        5,  # input fields: temperature, velx, vely, future_velx, future_vely
        64//downsample_factor,
        64//downsample_factor
    )
    expected_output_shape = (
        time_window,
        1,  # output field: temperature
        64//downsample_factor,
        64//downsample_factor
    )

    assert len(dataset) == 2 * (50 - 5 - 2 * time_window + 1)
    assert sample[0].shape == expected_input_shape
    assert sample[1].shape == expected_output_shape
