from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter

import datasets


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
):
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times)

    dataset = [
        {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
    ]
    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


# Get the HF dataset in their format
ds = datasets.load_dataset("autogluon/chronos_datasets", "m4_daily", split="train")
ds.set_format("numpy")
# Extract values
# start_times = [ds[i]['timestamp'] for i in range(len(ds))]
time_series_values = [ds[i]['target'] for i in range(len(ds))]
assert len(time_series_values) == len(ds)

convert_to_arrow("./tsmixup-data.arrow", time_series=time_series_values, start_times=None)
