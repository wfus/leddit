import pandas as pd
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np

from typing import List
from tqdm.notebook import tqdm
tqdm.pandas()


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x


def convert_df_to_dataloaders(df: pd.DataFrame,
                               feature_cols: List[str],
                               label_col: str,
                               **kwargs) -> torch.utils.data.DataLoader:
    """Converts a dataframe into a Pytorch dataloader, where each batch in the
    dataloader contains the features as a torch.Tensor and labels_ids also
    as a torch.Tensor.

    Feature columns MUST be numeric, and label column needs to be a long.
    Additional keyword arguments passed into the DataLoader.

    Inputs:
        df: Pandas dataframe to convert into pytorch format.
        feature_cols: list of strings, corresponding to numeric features.
        label_col: string corresponding to the label column in dataframe.
    
    Returns:
        Pytorch DataLoader with features and label.
    """
    train_X = torch.tensor(
        df[feature_cols].apply(
        lambda x: np.hstack([np.array(a) for a in x]), axis=1).tolist()
    )
    train_y = torch.tensor(df[label_col].tolist())

    dataset = data_utils.TensorDataset(train_X, train_y)
    return data_utils.DataLoader(dataset, **kwargs)


def train_simple_feedforward(train_dataloader, test_dataloader,
                             num_features, num_classes, lr=1e-3, epochs=10,
                             print_every=2000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FeedForward(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0
        
        correct_output = []
        predicted_output = []
        for inputs, labels in test_dataloader:
            with torch.no_grad():
                inputs = inputs.to(device).float()
                labels = labels.tolist()
                outputs = model(inputs)
                predicted_output += outputs.cpu().numpy().tolist()
                correct_output += labels
        
        total_testing = len(correct_output)
        total_correct = (np.array(correct_output) == np.array(predicted_output).argmax(axis=-1)).sum()
        print(f"Validation: {total_correct}/{total_testing} = {total_correct/total_testing}")
        
    return model