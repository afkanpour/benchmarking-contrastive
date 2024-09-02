import os
import json
from tqdm import tqdm


def fix_cache(filename):
    # load file
    with open(filename, encoding="utf-8") as f:
        entries = json.load(f)

    for idx in tqdm(range(len(entries))):
        # remove aieng/
        if "aieng/" in entries[idx]["image_path"]:
            entries[idx]["image_path"] = entries[idx]["image_path"].replace("aieng/", "")
        # remove nything before /projects
        folders = entries[idx]["image_path"].split("/")
        for fold in folders:
            if folders[0] == "projects":
                break
            else:
                folders.pop(0)
        folders.insert(0, "")
        entries[idx]["image_path"] = "/".join(folders)
    
    # overwrite file
    with open(filename, "w") as f:
        json.dump(entries, f)



if __name__ == "__main__":
    for split in ["train", "test"]:
        filename = f"/projects/multimodal/datasets/VQARAD/cache/{split}_data.json"
        fix_cache(filename)
    
    for split in ["train", "val", "test"]:
        filename = f"/projects/multimodal/datasets/PathVQA/cache/{split}_data.json"
        fix_cache(filename)