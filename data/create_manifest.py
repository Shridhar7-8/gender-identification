# import os
# from pathlib import Path

# train_file_name = "train_data.csv"
# valid_file_name = "valid_data.csv"

# if os.path.isfile(train_file_name) or os.path.isfile(valid_file_name):
#     try:
#         os.remove(train_file_name)
#     except:
#         pass
#     try:
#         os.remove(valid_file_name)
#     except:
#         pass


# def create_manifest(path, label, ext, mode):
#     audio_path = list(Path(path).glob("**/*." + ext))
#     if mode.lower() == "train":
#         file_name = train_file_name
#     else:
#         file_name = "valid_data.csv"
#     file = open(file_name, "a+", encoding="utf-8")
#     for path in audio_path:
#         print(str(str(path) + "," + str(label)), file=file)


# if __name__ == "__main__":
#     #For train
#     create_manifest(path="path_to_train_data_class_0", label=0, ext="wav", mode="train")
#     create_manifest(path="path_to_train_data_class_1", label=1, ext="wav", mode="train")

#     #For Valid
#     create_manifest(path="path_to_valid_data_class_0", label=0, ext="wav", mode="valid")
#     create_manifest(path="path_to_valid_data_class_1", label=1, ext="wav", mode="valid")


import os
import json
import random
from pathlib import Path

# File paths for the output CSV files
train_file_path = "/mnt/32mins/gender_detection/ekstep-language-identification/data/train.csv"
valid_file_path = "/mnt/32mins/gender_detection/ekstep-language-identification/data/valid.csv"

# Remove existing files if they exist
for file_path in [train_file_path, valid_file_path]:
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed existing file: {file_path}")
    except OSError as e:
        print(f"Error removing {file_path}: {e}")

def collect_audio_paths(metadata_path, base_wav_path):
    """Collect audio paths and their corresponding gender labels from metadata."""
    audio_data = []
    
    # Open the metadata JSON file and load its contents
    with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
        for line in metadata_file:
            # Each line in the metadata file is a JSON object
            entry = json.loads(line.strip())
            
            # Extract the relevant information
            gender = entry.get('gender')
            filepath = entry.get('filepath')
            
            # Set label: 0 for Female, 1 for Male
            if gender == "Female":
                label = 0
            elif gender == "Male":
                label = 1
            else:
                continue  # Skip if gender is not defined
            
            # Construct full path to the WAV file
            full_audio_path = os.path.join(base_wav_path, filepath)
            
            # Append the path and label
            audio_data.append((full_audio_path, label))
    
    return audio_data

if __name__ == "__main__":
    # Path to the metadata and wav directories
    base_data_path = "/mnt/32mins/gender_detection/datasets"
    
    # Metadata paths for each language (you can extend this for multiple languages)
    languages_metadata = {
        "Bengali": {
            "metadata_path": "/mnt/32mins/gender_detection/datasets/Bengali/metadata_train.json",
            "wav_path": "/mnt/32mins/gender_detection/datasets/Bengali/wavs"
        },
        "Hindi": {
            "metadata_path": "/mnt/32mins/gender_detection/datasets/Hindi/metadata_train.json",
            "wav_path": "/mnt/32mins/gender_detection/datasets/Hindi/wavs"
        },
        "Kannada": {
            "metadata_path": "/mnt/32mins/gender_detection/datasets/Kannada/metadata_train.json",
            "wav_path": "/mnt/32mins/gender_detection/datasets/Kannada/wavs"
        },
        "Telugu": {
            "metadata_path": "/mnt/32mins/gender_detection/datasets/Telugu/metadata_train.json",
            "wav_path": "/mnt/32mins/gender_detection/datasets/Telugu/wavs"
        }
        # You can add more languages here if necessary
    }
    
    all_audio_data = []
    
    # Collect audio data for all languages
    for language, paths in languages_metadata.items():
        metadata_path = paths['metadata_path']
        wav_path = paths['wav_path']
        
        # Collect audio data from the metadata
        audio_data = collect_audio_paths(metadata_path, wav_path)
        all_audio_data.extend(audio_data)

    # Shuffle the entire dataset to ensure randomness
    random.shuffle(all_audio_data)
    
    # Split data into 80% training and 20% validation
    train_split = 0.8
    split_idx = int(len(all_audio_data) * train_split)
    train_data = all_audio_data[:split_idx]
    valid_data = all_audio_data[split_idx:]

    # Write train data to train.csv
    with open(train_file_path, "w", encoding="utf-8") as train_file:
        for path, label in train_data:
            train_file.write(f"{path},{label}\n")

    # Write valid data to valid.csv
    with open(valid_file_path, "w", encoding="utf-8") as valid_file:
        for path, label in valid_data:
            valid_file.write(f"{path},{label}\n")

    print(f"Data successfully written to {train_file_path} and {valid_file_path}")
