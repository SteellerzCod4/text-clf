import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.embeddings = nn.Linear(input_size, hidden_size)
        self.ac1 = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embs = self.embeddings(x)
        outputs = self.ac1(embs)
        y = self.hidden_layer(outputs)
        return y

    def prepare_data(self, X_train, y_train, batch_size):
        print(f"X_train.dtype: {X_train.dtype}")
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    def fit(self, x_train, y_train, lr, batch_size, num_epochs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        train_loader = self.prepare_data(x_train, y_train, batch_size)

        print("Model has {:,} parameters".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for epoch in tqdm(range(num_epochs)):
            train_loss, test_loss = 0, 0
            for docs, labels in train_loader:
                docs, labels = docs.to(device), labels.to(device)

                self.train()
                outputs = self.forward(docs)
                loss = loss_fn(outputs, labels)
                train_loss += loss.item()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

    def predict(self, x_test):
        x_test = torch.from_numpy(x_test).float()
        self.eval()
        outputs = self.forward(x_test)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

# X_train, X_test, y_train, y_test = train_test_split(embedding_X, y.values, test_size=0.2, stratify=y, random_state=42)

# input_size = embedding_X.shape[1]
# hidden_size = 64
# batch_size = 64
# num_classes = NUM_CLASSES # 5
# num_epochs = 300
# lr = 0.01

# my_model = MyNeuralNetwork(input_size, hidden_size, num_classes)
# my_model.fit(X_train, y_train, lr=lr, batch_size=batch_size, num_epochs=num_epochs)
# y_pred = my_model.predict(X_test)
