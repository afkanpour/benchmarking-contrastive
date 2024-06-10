"""Parse the results of several experiments from json files and make a single csv file"""

import sys
import os
import json
import pandas as pd


slurm_job_id = sys.argv[1]
root_dir = os.path.join("../../src", str(slurm_job_id))
datasets = ["vindr_mammo", "mimic_cxr_chexpert", "pad_ufes_20", "skin_cancer", "pcam", 
            "nct_crc_he_100k", "lc25000lung", "lc25000colon", "bach",
            "sicap", "chestmnist_plus", "pathmnist_plus", "dermamnist_plus", "octmnist_plus", "pneumoniamnist_plus", \
            "retinamnist_plus", "breastmnist_plus", "bloodmnist_plus", "tissuemnist_plus", "organamnist_plus", \
            "organcmnist_plus", "organsmnist_plus"]

results_df = {}
for dataset in datasets:
    file_name = os.path.join(root_dir, f"result_{dataset}.json")
    if os.path.exists(file_name):
        with open(file_name) as f:
            data = json.load(f)

        data = data["metrics"]

        if results_df == {}:
            results_df["dataset"] = [dataset]
            results_df.update({key: [value] for key, value in data.items()})
        else:
            for key, value in results_df.items():
                if key != "dataset" and key not in data:
                    results_df[key].append(float("nan"))

            for key, value in data.items():
                if key in results_df:
                    results_df[key].append(value)
                else:
                    results_df[key] = [float("nan")] * len(results_df["dataset"])
                    results_df[key].append(value)
            results_df["dataset"].append(dataset)
    

results_df = pd.DataFrame.from_dict(results_df, orient="columns")

output_file = os.path.join(root_dir, f"results_table.csv")
results_df.to_csv(output_file, sep=",")
print(f"Table saved to {output_file}")



