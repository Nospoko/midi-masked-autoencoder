import json
import numpy as np
from datasets import load_dataset

def calculate_mean_std():
    ds = load_dataset("JasiekKaczmarczyk/maestro-v1-sustain-masked", split="train")

    #  2 ** -5 is used to prevent log2 of 0
    dstart = np.nan_to_num(ds["dstart"]) + 1e-8
    duration = ds["duration"]

    log2_dstart = np.log2(dstart)
    log2_duration = np.log2(duration)

    mean_dstart = np.mean(log2_dstart)
    std_dstart = np.std(log2_dstart)

    mean_duration = np.mean(log2_duration)
    std_duration = np.std(log2_duration)

    features = {
        "mean_dstart": mean_dstart,
        "std_dstart": std_dstart,
        "mean_duration": mean_duration,
        "std_duration": std_duration,
    }

    with open("artifacts/time_features.json", "x") as f:
        json.dump(features, f)
        f.close()

if __name__ == "__main__":
    calculate_mean_std()