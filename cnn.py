import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Assuming input shape is (batch_size, 1, 255)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        # self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3)


        self.fc = nn.Linear(64 * 58, 14)  # set output feature as we need.
        # ############################

    def forward(self, x):
        x = x.unsqueeze(1)  # add 1 dimension for convolutional layer

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x
