import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


class SicapDataset(torch.utils.data.Dataset):
    def __init__(
        self, root="./data/SICAPv2", image_dir="images", transform=None, split="test",
    ):

        image_dir = os.path.join(root, image_dir)

        if split == "train":
            csv_file = os.path.join(root, "partition/Test", "Train.xlsx")
            self.data = pd.read_excel(csv_file)
        elif split == "test":
            csv_file = os.path.join(root, "partition/Test", "Test.xlsx")
            self.data = pd.read_excel(csv_file)

        # drop all columns except image_name and the label columns
        label_columns = ["NC", "G3", "G4", "G5"]  # , 'G4C']
        self.data = self.data[["image_name"] + label_columns]

        # get the index of the maximum label value for each row
        self.data["labels"] = self.data[label_columns].idxmax(axis=1)

        # replace the label column values with categorical values
        self.cat_to_num_map = label_map = {
            "NC": 0,
            "G3": 1,
            "G4": 2,
            "G5": 3,
        }  # , 'G4C': 4}
        self.data["labels"] = self.data["labels"].map(label_map)

        self.image_paths = self.data["image_name"].values
        self.labels = self.data["labels"].values
        self.image_dir = image_dir
        self.transform = transform
        self.train = True if split == "train" else False
        self.classes = [
            "non-cancerous well-differentiated glands",
            "gleason grade 3 with atrophic well differentiated and dense glandular regions",
            "gleason grade 4 with cribriform, ill-formed, large-fused and papillary glandular patterns",
            "gleason grade 5 with nests of cells without lumen formation, isolated cells and pseudo-roseting patterns",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        return image, label


class PCAM(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        if os.path.exists(os.path.join(root, f"cache/{name}_{split}.pkl")):
            print("!!!Using cached dataset")
            dataset = pickle.load(
                open(os.path.join(root, f"cache/{name}_{split}.pkl"), "rb"),
            )
        else:
            os.makedirs(os.path.join(root, "cache/"), exist_ok=True)

            dataset = load_dataset(
                "1aurent/PatchCamelyon",
                cache_dir=os.path.join(root, "scratch/"),
            )[split]

            pickle.dump(
                dataset,
                open(os.path.join(root, f"cache/{name}_{split}.pkl"), "wb"),
            )

        self.transform = transform
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        image = self.transform(image)
        return image, int(self.data[idx]["label"])



class RoCo(Dataset):
    def __init__(self, name, transform) -> None:
        if os.path.exists(f"cache/{name}.pkl"):
            print("!!!Using cached dataset")
            dataset = pickle.load(
                open(f"cache/{name}.pkl", "rb"),
            )
        else:
            os.makedirs("cache/", exist_ok=True)

            dataset = load_dataset(
                "MedIR/roco",
                cache_dir="~/scratch/.cache",
            )["test"]

            pickle.dump(
                dataset,
                open(f"cache/{name}.pkl", "wb"),
            )

        self.transform = transform
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        image = self.transform(image)
        return image, [self.data[idx]["caption"]]


class MIMIC_CXR(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["train", "validate", "test"], f"split {split} is not supported in dataset {name}."
        df = pd.read_csv(os.path.join(root, f"{name}_image_{split}.csv"), sep=",")

        self.images = df["image_path"].tolist()
        self.captions = df["caption"].tolist()
        self.image_root = root
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.image_root, str(self.images[idx]))))
        return image, [str(self.captions[idx])]

class MIMIC_CXR_LP(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["train", "validate", "test"], f"split {split} is not supported in dataset {name}."

        if os.path.exists(f"cache/{name}_{split}.pkl"):
            print("!!!Using cached dataset")
            entries = pickle.load(
                open(f"cache/{name}_{split}.pkl", "rb"),
            )
        else:
            os.makedirs("cache/", exist_ok=True)

            data_path = os.path.join(root, f"{name}_{split}.json")
            if not os.path.exists(data_path):
                raise FileNotFoundError(data_path)
            with open(data_path, "rb") as file:
                entries = json.load(file)

            # remove entries with no label
            entries_df = pd.DataFrame(entries)
            entries_df = entries_df.loc[entries_df["label"].apply(len) > 0]
            entries = entries_df.to_dict("records")
            pickle.dump(
                entries,
                open(f"cache/{name}_{split}.pkl", "wb"),
            )

        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.entries[idx]["image_path"]).convert("RGB"))
        label = torch.tensor(self.entries[idx]["label"])
        return image, torch.nan_to_num(label, nan=2.0)


class VinDr_Mammo(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["training", "test"], f"split {split} is not supported in dataset {name}."

        if os.path.exists(f"cache/{name}_{split}.pkl"):
            print("!!!Using cached dataset")
            entries = pickle.load(
                open(f"cache/{name}_{split}.pkl", "rb"),
            )
        else:
            os.makedirs("cache/", exist_ok=True)

            df = pd.read_csv(os.path.join(root, "finding_annotations.csv"))
            # filter split samples
            df = df.loc[df["split"] == split]
            # keep only required columns
            df = df[["study_id", "image_id", "finding_categories"]]
            # create list of integer labels
            df["label"] = df["finding_categories"].apply(self._build_label)
            df.drop(columns=["finding_categories"], inplace=True)
            # create image paths
            df["path"] = df.apply(lambda row: os.path.join(root, "Processed_Images", row["study_id"], f"{row['image_id']}.png"), axis=1)
            df.drop(columns=["study_id", "image_id"], inplace=True)

            entries = df.to_dict("records")
            pickle.dump(
                entries,
                open(f"cache/{name}_{split}.pkl", "wb"),
            )
        # only consider multi-class samples
        entries = [entry for entry in entries if sum(entry["label"]) < 2]
        for e in entries:
            e["label"] = np.argmax(e["label"])
        self.entries = entries
        self.transform = transform

    def _build_label(self, str_label):
        classes = [
                "Mass",
                "Suspicious Calcification",
                "Asymmetry",
                "Focal Asymmetry",
                "Global Asymmetry",
                "Architectural Distortion",
                "Skin Thickening",
                "Skin Retraction",
                "Nipple Retraction",
                "Suspicious Lymph Node"
        ]
        str_label = str_label[2:-2].split("', '")
        return [1 if lbl in str_label else 0 for lbl in classes]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.entries[idx]["path"]).convert("RGB"))
        label = torch.tensor(self.entries[idx]["label"])
        return image, label


class SkinCancer(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["train", "test"], f"split {split} is not supported in dataset {name}."

        if os.path.exists(f"cache/{name}_{split}.pkl"):
            print("!!!Using cached dataset")
            entries = pickle.load(
                open(f"cache/{name}_{split}.pkl", "rb"),
            )
        else:
            os.makedirs("cache/", exist_ok=True)

            df = pd.read_csv(os.path.join(root, "HAM10000_metadata.csv"))
            # keep only required columns
            df = df[["image_id", "dx"]]
            # create integer label
            df["label"] = df["dx"].apply(self._build_label)
            df.drop(columns=["dx"], inplace=True)
            # create image paths
            df["path"] = df.apply(lambda row: os.path.join(root, "skin_cancer", f"{row['image_id']}.jpg"), axis=1)
            df.drop(columns=["image_id"], inplace=True)
            # split into train and test
            dataset = {}
            dataset["test"] = df.sample(frac = 0.2)
            dataset["train"] = df.drop(dataset["test"].index)

            # cache datasets
            for spl in ["train", "test"]:
                pickle.dump(
                    dataset[spl].to_dict("records"),
                    open(f"cache/{name}_{spl}.pkl", "wb"),
                )

            entries = dataset[split].to_dict("records")

        self.entries = entries
        self.transform = transform

    def _build_label(self, str_label):
        classes = {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6
        }
        return classes[str_label]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.entries[idx]["path"]).convert("RGB"))
        return image, int(self.entries[idx]["label"])


class PadUfes20(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["train", "test"], f"split {split} is not supported in dataset {name}."

        if os.path.exists(f"cache/{name}_{split}.pkl"):
            print("!!!Using cached dataset")
            entries = pickle.load(
                open(f"cache/{name}_{split}.pkl", "rb"),
            )
        else:
            os.makedirs("cache/", exist_ok=True)

            df = pd.read_csv(os.path.join(root, "metadata.csv"))
            # keep only required columns
            df = df[["img_id", "diagnostic"]]
            # create integer label
            df["label"] = df["diagnostic"].apply(self._build_label)
            df.drop(columns=["diagnostic"], inplace=True)
            # create image paths
            df["path"] = df["img_id"].apply(lambda imgid: os.path.join(root, "Dataset", imgid))
            df.drop(columns=["img_id"], inplace=True)
            # split into train and test
            dataset = {}
            dataset["test"] = df.sample(frac = 0.2)
            dataset["train"] = df.drop(dataset["test"].index)

            # cache datasets
            for spl in ["train", "test"]:
                pickle.dump(
                    dataset[spl].to_dict("records"),
                    open(f"cache/{name}_{spl}.pkl", "wb"),
                )

            entries = dataset[split].to_dict("records")

        self.entries = entries
        self.transform = transform

    def _build_label(self, str_label):
        classes = {
        "BCC": 0,
        "MEL": 1,
        "SCC": 2,
        "ACK": 3,
        "NEV": 4,
        "SEK": 5
        }
        return classes[str_label]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.entries[idx]["path"]).convert("RGB"))
        return image, int(self.entries[idx]["label"])


class MedMNIST(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["train", "val", "test"], f"split {split} is not supported in dataset {name}."
        if os.path.exists(f"cache/{name}_{split}.pkl"):
            print("!!!Using cached dataset")
            dataset = pickle.load(
                open(f"cache/{name}_{split}.pkl", "rb"),
            )
        else:
            os.makedirs("cache/", exist_ok=True)

            dataset = load_dataset(
                "albertvillanova/medmnist-v2",
                name,
                split=split,
                verification_mode="no_checks",
            )
            # save with pickle
            pickle.dump(
                dataset,
                open(f"cache/{name}_{split}.pkl", "wb"),
            )

        self.transform = transform
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        image = self.transform(image)
        return image, torch.tensor(self.data[idx]["label"])


class DeepEyeNet(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["train", "valid", "test"], f"split {split} is not supported in dataset {name}."

        if os.path.exists(f"cache/{name}_{split}.pkl"):
            print("!!!Using cached dataset")
            entries = pickle.load(open(f"cache/{name}_{split}.pkl", "rb"))
        else:
            os.makedirs("cache/", exist_ok=True)

            with open(os.path.join(root, f"DeepEyeNet_{split}.json")) as file:
                self.entries = json.load(file)
            temp = {list(row.keys())[0]: list(list(row.values())[0].values()) for row in self.entries}
            df = pd.DataFrame.from_dict(temp, orient="index", columns=["keywords", "clinical-description"])

            # add image_path and captions to dataframe
            df["image_path"] = df.index
            df["image_path"] = df["image_path"].apply(lambda path: os.path.join(root, path))
            df["caption"] = df.apply(lambda row: "\n".join([row["keywords"], row["clinical-description"]]), axis=1)
            df.drop(columns=["keywords", "clinical-description"], inplace=True)
            entries = df.to_dict("records")

            pickle.dump(
                entries,
                open(f"cache/{name}_{split}.pkl", "wb"),
            )

        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.entries[idx]["image_path"]).convert("RGB"))
        return image, [str(self.entries[idx]["caption"])]


class Quilt1M(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["train", "val"], f"split {split} is not supported in dataset {name}."
        if os.path.exists(f"cache/{name}_{split}.pkl"):
            print("!!!Using cached dataset")
            entries = pickle.load(
                open(f"cache/{name}_{split}.pkl", "rb"),
            )
        else:
            os.makedirs("cache/", exist_ok=True)

            df = pd.read_csv(os.path.join(root, "quilt_1M_lookup.csv"))
            # declare the desired subsets
            if name == "quilt_1m":
                subsets = ["openpath", "pubmed", "quilt", "laion"]
            elif name == "quilt_1m_no_pubmed":
                subsets = ["openpath", "quilt", "laion"]
            elif name == "quilt_1m_no_laion":
                subsets = ["openpath", "pubmed", "quilt"]
            elif name == "quilt_1m_no_pubmed_no_laion":
                subsets = ["openpath", "quilt"]
            # filter lookup table based on split and subsets
            df = df.loc[df.apply(lambda row: row["split"] == split and row["subset"] in subsets, axis=1)]
            # keep only necessary info
            df = df[["image_path", "caption", "split", "subset"]]
            entries = df.to_dict("records")

            # save with pickle
            pickle.dump(
                entries,
                open(f"cache/{name}_{split}.pkl", "wb"),
            )

        self.entries = entries
        self.root_dir = root
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        try:
            image = Image.open(os.path.join(self.root_dir, "quilt_1m", self.entries[idx]["image_path"])).convert("RGB")
            image = self.transform(image)
            return image, [self.entries[idx]["caption"]]
        except Exception:
            print(f"Error loading image {self.entries[idx]['image_path']}")
            idx = (idx + 1) % len(self)
            return self.__getitem__(idx)


class MedMNISTPlus(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        assert split in ["train", "val", "test"], f"split {split} is not supported in dataset {name}."

        # note: loading the data might take 5-6 minutes
        data = np.load(os.path.join(root, name.replace("_plus", "_224.npz")), mmap_mode="r")

        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"]
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        image = self.transform(Image.fromarray(image).convert("RGB"))
        label = self.labels[idx].astype(int)
        if len(label) == 1:
            label = int(label[0])
        return image, label


class LC25000(Dataset):
    def __init__(self, name, root, split, transform, organ="lung") -> None:
        from datasets import load_from_disk

        organ = 0 if organ == "lung" else 1
        if os.path.exists(os.path.join(root, f"cache/{name}_{split}.arrow")):
            print("!!!Using cached dataset")
            dataset = load_from_disk(os.path.join(root, f"cache/{name}_{split}.arrow"))
        else:
            os.makedirs(os.path.join(root, "cache/"), exist_ok=True)

            dataset = load_dataset(
                "1aurent/LC25000",
                cache_dir=os.path.join(root, "scratch/"),
            )["train"]
            dataset = dataset.filter(lambda row: row["organ"] == organ)

            datasets_dict = dataset.train_test_split(test_size=0.2, shuffle=True,
                                                     train_indices_cache_file_name=os.path.join(root, f"cache/{name}_train_indices.arrow"),
                                                     test_indices_cache_file_name=os.path.join(root, f"cache/{name}_test_indices.arrow"))

            datasets_dict["train"].save_to_disk(os.path.join(root, f"cache/{name}_train.arrow"))
            datasets_dict["test"].save_to_disk(os.path.join(root, f"cache/{name}_test.arrow"))

        self.transform = transform
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        image = self.transform(image)
        return image, int(self.data[idx]["label"])


class BACH(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        os.makedirs(os.path.join(root, "cache/"), exist_ok=True)

        dataset = load_dataset(
            "1aurent/BACH",
            cache_dir=os.path.join(root, "scratch/"),
            split="train",
        )
        data_dict = dataset.train_test_split(test_size=0.25, train_size=0.75, shuffle=True, seed=0)

        self.data = data_dict[split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        image = self.transform(image)
        return image, int(self.data[idx]["label"])


class NCK_CRC(Dataset):
    def __init__(self, name, root, split, transform) -> None:
        from datasets import load_from_disk

        assert split in ("train", "train_nonorm", "validation")
        self.class_maping = {
            "ADI": 0,
            "DEB": 1,
            "LYM": 2,
            "MUC": 3,
            "MUS": 4,
            "NORM": 5,
            "STR": 6,
            "TUM": 7,
        }

        if os.path.exists(os.path.join(root, f"cache/{name}_{split}.arrow")):
            print("!!!Using cached dataset")
            dataset = load_from_disk(os.path.join(root, f"cache/{name}_{split}.arrow"))
        else:
            os.makedirs(os.path.join(root, "cache/"), exist_ok=True)

            dataset = load_dataset(
                "DykeF/NCTCRCHE100K",
                cache_dir=os.path.join(root, "scratch/"),
                split=split,
            )

            # exclude label "BACK" as in Quilt-1M's paper
            dataset = dataset.filter(lambda row: row["label"] != "BACK")

            dataset.save_to_disk(os.path.join(root, f"cache/{name}_{split}.arrow"))

        self.transform = transform
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        image = self.transform(image)
        return image, self.class_maping[self.data[idx]["label"]]
