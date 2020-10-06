import os
from pathlib import Path

train_file_name = "train_data.csv"
valid_file_name = "valid_data.csv"

if os.path.isfile(train_file_name) or os.path.isfile(valid_file_name):
    try:
        os.remove(train_file_name)
    except:
        pass
    try:
        os.remove(valid_file_name)
    except:
        pass


def create_manifest(path, label, ext, mode):
    audio_path = list(Path(path).glob("**/*." + ext))
    if mode.lower() == "train":
        file_name = train_file_name
    else:
        file_name = "valid_data.csv"
    file = open(file_name, "a+", encoding="utf-8")
    for _ in audio_path:
        print(str(str(_) + "," + str(label)), file=file)


if __name__ == "__main__":
    create_manifest(path="/home/jupyter/lid_data/hindi", label=0, ext="wav", mode="Train")
    create_manifest(path="/home/jupyter/lid_data/english", label=1, ext="wav", mode="Train")