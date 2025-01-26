# gender-identification
Gender Identification Model
This project is based on the EkStep Language Identification repository, originally developed as a part of [Vakyansh's](https://open-speech-ekstep.github.io/) recipes to build state of the art Speech Recogniition Model. The repository was modified and trained to classify gender (male/female) instead of language identification.

# Overview
The model takes audio utterances as input and classifies them based on gender. It uses a deep learning-based architecture and is trained on labeled speech datasets.

# Dataset
The dataset consists of audio samples labeled with gender categories (male, female). The data is preprocessed and formatted according to the structure required by the ekstep-language-identification repository.


# Model Checkpoints
Trained model checkpoints are stored in the checkpoints directory:

📌 Path: gender-identification\checkpoints

The following checkpoint files are available:

best_model – The model with the best validation accuracy during training.
current_checkpoint – The latest model checkpoint from training (useful for resuming training).
final_model – The final trained model after all epochs.

