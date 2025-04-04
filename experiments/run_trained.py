import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from chronos import BaseChronosPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map=device,
    torch_dtype=torch.bfloat16,
)

text = "https://raw.githubusercontent.com/AileenNielsen/"
text += "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
df = pd.read_csv(text)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
# mean is an fp32 tensor with shape [batch_size, prediction_length]
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(df["#Passengers"]),
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],
)

forecast_index = range(len(df), len(df) + 12)
low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

sns.set(style="whitegrid", font_scale=3)
sns.set_palette("Set1")
plt.figure(figsize=(16, 8), dpi=100)
plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()
