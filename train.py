# import torch.nn as nn
# import torch.optim as optim

# from models.model import get_model
# from utils.training import *

# torch.manual_seed(0)

# # Set Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use_cuda = torch.cuda.is_available()

# train_parameters = load_yaml_file("train_config.yml")["train_parameters"]

# # inputs
# train_manifest = train_parameters["train_manifest"]
# valid_manifest = train_parameters["valid_manifest"]
# # outputs
# checkpoint_path = train_parameters["checkpoint_path"]

# # Hyperparameters
# batch_size = int(train_parameters["batch_size"])
# learning_rate = float(train_parameters["learning_rate"])
# num_epochs = int(train_parameters["num_epochs"])
# num_workers = int(train_parameters["num_workers"])
# num_classes = int(train_parameters["num_classes"])

# # Load_Data
# loaders = load_data_loaders(train_manifest, valid_manifest, batch_size, num_workers)

# # Load Model
# model = get_model(device, num_classes, pretrained=False)

# # Display model parameters
# show_model_parameters(model)

# # Set model hyperparameters
# optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()

# # start model trainning
# trained_model = train(1, num_epochs, device, np.Inf, loaders, model, optimizer, criterion, use_cuda, checkpoint_path,
#                       save_for_each_epoch=True)

import torch.nn as nn
import torch.optim as optim
import wandb  # Import wandb
from models.model import get_model
from utils.training import *


wandb.init(project="gender-detection", config={
    "learning_rate": 1e-3,
    "epochs": 100,
    "batch_size": 128,
    "model_architecture": "ResNet18"
})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

train_parameters = load_yaml_file("train_config.yml")["train_parameters"]


train_manifest = train_parameters["train_manifest"]
valid_manifest = train_parameters["valid_manifest"]

checkpoint_path = train_parameters["checkpoint_path"]


batch_size = int(train_parameters["batch_size"])
learning_rate = float(train_parameters["learning_rate"])
num_epochs = int(train_parameters["num_epochs"])
num_workers = int(train_parameters["num_workers"])
num_classes = int(train_parameters["num_classes"])


loaders = load_data_loaders(train_manifest, valid_manifest, batch_size, num_workers)


model = get_model(device, num_classes, pretrained=False)


show_model_parameters(model)


optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


trained_model = train(1, num_epochs, device, np.inf, loaders, model, optimizer, criterion, use_cuda, checkpoint_path,
                      save_for_each_epoch=True, log_wandb=True) 


wandb.finish()
