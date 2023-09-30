import torch
import json

from src.data.data_preprocessing import X_train, y_train, df, embedding_X
from src.model.neural_network import MyNeuralNetwork

with open("model_config.json", 'r') as fp:
    hyperparameters = json.load(fp)


# model hyperparameters
input_size = embedding_X.shape[1]
hidden_size = int(hyperparameters["hyperparameters"]["hidden_size"])  # 64
batch_size = int(hyperparameters["hyperparameters"]['batch_size'])  # 64
num_classes = df.category.nunique()  # 5
num_epochs = int(hyperparameters["hyperparameters"]['num_epochs'])  # 300
lr = float(hyperparameters["hyperparameters"]['lr'])  # 0.01

# initialize model
my_model = MyNeuralNetwork(input_size, hidden_size, num_classes)
my_model.fit(X_train, y_train, lr=lr, batch_size=batch_size, num_epochs=num_epochs)

# save model
torch.save(my_model, "my_model.pt")
