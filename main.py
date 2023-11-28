import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from cnn import CNN
from util import *

# load data - acc
data = pd.read_csv('keyboard_tap_full.csv')

# 4. Standardization
psd_x = standarize(dataStrToNum(data["psd_x"]))
psd_y = standarize(dataStrToNum(data["psd_y"]))
psd_z = standarize(dataStrToNum(data["psd_z"]))

fft_x = standarize(dataStrToNum(data["fft_x"]))
fft_z = standarize(dataStrToNum(data["fft_y"]))
fft_y = standarize(dataStrToNum(data["fft_z"]))

acf_x = standarize(dataStrToNum(data["acf_x"]))
acf_y = standarize(dataStrToNum(data["acf_y"]))
acf_z = standarize(dataStrToNum(data["acf_z"]))

audio_mfcc = standarize(dataStrToNum(data["audio_mfcc"]))
audio_fft = standarize(dataStrToNum(data["audio_fft"]))

labels = data["label"]
features = np.concatenate((psd_x, psd_y, psd_z, fft_x, fft_y, fft_z, acf_x, acf_y, acf_z, audio_mfcc, audio_fft), axis=1)
features = pd.DataFrame(features)

# 5. Splitting Data
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

train_data, validation_data, test_data = np.split(features.sample(frac=1, random_state=42),
                                                  [int(train_ratio*len(features)), int((train_ratio+validation_ratio)*len(features))])

train_labels, validation_labels, test_labels = labels[train_data.index], labels[validation_data.index], labels[test_data.index]

print(train_data.shape)  # (N * 0.8, 255)
print(validation_data.shape)  # (N * 0.1, 255)
print(test_data.shape)  # (N * 0.1, 255)

# 6. Conversion to PyTorch Tensors
X_train = torch.tensor(train_data.values, dtype=torch.float32)
y_train = torch.tensor(train_labels.values, dtype=torch.long)

X_val = torch.tensor(validation_data.values, dtype=torch.float32)
y_val = torch.tensor(validation_labels.values, dtype=torch.long)

X_test = torch.tensor(test_data.values, dtype=torch.float32)
y_test = torch.tensor(test_labels.values, dtype=torch.long)


# 7. Dataloader
batch_size = 32

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

cnn = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(cnn.parameters(), lr=1e-4)
optimizer = optim.Adam(cnn.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
lossHistory = []

cnn.train()  # 학습을 위함
for epoch in range(20):
    for index, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 기울기 초기화
        output = cnn(data)  # forward
        loss = criterion(output, target)
        loss.backward()  # back-prop
        optimizer.step()

        if index % 50 == 0:
            lossHistory.append(loss.item())
            print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))


cnn.eval()  # test case 학습 방지를 위함
test_loss = 0
correct = 0

TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives

with torch.no_grad():
    for data, target in test_loader:
        output = cnn(data)
        test_loss += criterion(output, target).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Update TP, FP, FN
        for i in range(len(target)):
            if pred[i] == target[i]:
                if pred[i] == 1:  # Assuming 1 is the positive class
                    TP += 1
            else:
                if pred[i] == 1:
                    FP += 1
                elif target[i] == 1:
                    FN += 1

# Calculate precision, recall, and F1 score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

print('Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}\n'.format(precision, recall, f1_score))

plt.plot(lossHistory)
plt.title("loss")
plt.show()