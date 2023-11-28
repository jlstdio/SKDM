import numpy as np
from util import *

class conv1d:
    def __init__(self, outCh, kernelSize, stride, padding):
        self.outCh = outCh
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(outCh, kernelSize) / 9
        self.bias = np.zeros(outCh)

    def forward(self, input):
        self.lastInput = input

        batchSize, inLen = input.shape
        padding = np.pad(input, ((0, 0), (self.padding, self.padding)), mode='constant', constant_values=0)
        outLen = (inLen + 2 * self.padding - self.kernelSize) // self.stride + 1
        output = np.zeros((batchSize, self.outCh, outLen))

        for i in range(outLen):
            input_slice = padding[:, i*self.stride:i*self.stride+self.kernelSize]
            for f in range(self.outCh):
                output[:, f, i] = np.sum(input_slice * self.weights[f], axis=1) + self.bias[f]

        return output

    def backward(self, dout):
        batchSize, inLen = self.lastInput.shape
        dPadding = np.zeros((batchSize, inLen + 2 * self.padding))
        dWeights = np.zeros_like(self.weights)
        dBias = np.zeros_like(self.bias)

        outLen = dout.shape[2]

        for i in range(outLen):
            for f in range(self.outCh):
                input_slice = self.lastInput[:, i * self.stride:i * self.stride + self.kernelSize]
                # Ensure the input_slice is correctly shaped
                if input_slice.shape[1] != self.kernelSize:
                    continue
                # Update gradients
                dWeights[f] += np.sum(input_slice * dout[:, f, i].reshape(-1, 1), axis=0)
                dBias[f] += np.sum(dout[:, f, i], axis=0)
                dPadding[:, i * self.stride:i * self.stride + self.kernelSize] += dout[:, f, i].reshape(-1, 1) * \
                                                                                        self.weights[f]

        return dPadding[:, self.padding:-self.padding]

class maxpool1d:
    def __init__(self, kernelSize, stride):
        self.kernelSize = kernelSize
        self.stride = stride

    def forward(self, input):
        self.lastInput = input

        batchSize, outCh, inLen = input.shape
        outLen = (inLen - self.kernelSize) // self.stride + 1
        output = np.zeros((batchSize, outCh, outLen))

        for i in range(outLen):
            input_slice = input[:, :, i*self.stride:i*self.stride+self.kernelSize]
            output[:, :, i] = np.max(input_slice, axis=2)

        return output

    def backward(self, dout):
        batchSize, outCh, inLen = self.lastInput.shape
        dinput = np.zeros_like(self.lastInput)

        for i in range(dout.shape[2]):
            input_slice = self.lastInput[:, :, i*self.stride:i*self.stride+self.kernelSize]
            max_indices = input_slice == np.max(input_slice, axis=2, keepdims=True)
            dinput[:, :, i*self.stride:i*self.stride+self.kernelSize] += dout[:, :, i].reshape(batchSize, outCh, 1) * max_indices

        return dinput

class fc:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros(output_size)

    def forward(self, input):
        self.lastInput_shape = input.shape
        batchSize = input.shape[0]
        input_flattened = input.reshape(batchSize, -1)
        self.lastInput = input_flattened
        output = np.dot(input_flattened, self.weights) + self.bias
        return output

    def backward(self, dout):
        batchSize = self.lastInput_shape[0]
        dinput_flat = np.dot(dout, self.weights.T)
        dinput = dinput_flat.reshape(self.lastInput_shape)
        dWeights = np.dot(self.lastInput.T, dout) / batchSize
        dBias = np.sum(dout, axis=0) / batchSize
        return dinput, dWeights, dBias

def cross_entropy_loss(predicted, actual):
    m = actual.shape[0]
    log_likelihood = -np.log(predicted[range(m), actual])
    loss = np.sum(log_likelihood) / m
    return loss

def softmax(x):
    # exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    exp_x = np.exp(x - np.max(x, keepdims=True))
    return exp_x / np.sum(exp_x, keepdims=True)

def update_parameters(parameters, gradients, lr):
    for key in parameters.keys():
        parameters[key] -= lr * gradients[key]

def train(model, X_train, y_train, epochs, lr):
    for epoch in range(epochs):
        # Forward pass
        print(X_train.shape)
        out = model['conv'].forward(X_train)
        print(out.shape)
        out = model['pool'].forward(out)
        print(out.shape)
        out = model['fc'].forward(out)
        print(out.shape)

        # Compute loss
        probs = softmax(out)
        loss = cross_entropy_loss(probs, y_train)

        # Backward pass
        dout = probs
        dout[range(X_train.shape[0]), y_train] -= 1
        dout /= X_train.shape[0]

        dinput, dWeights_fc, dBias_fc = model['fc'].backward(dout)
        dinput = model['pool'].backward(dinput)
        dinput = model['conv'].backward(dinput)

        # Update parameters
        update_parameters({'weights_fc': model['fc'].weights, 'bias_fc': model['fc'].bias},
                          {'weights_fc': dWeights_fc, 'bias_fc': dBias_fc},
                          lr)

        print("Epoch %d, Loss: %f" % (epoch, loss))


def predict(model, X):
    out = model['conv'].forward(X)
    out = model['pool'].forward(out)
    out = model['fc'].forward(out)
    probs = softmax(out)
    return np.argmax(probs, axis=1)


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

# Conversion to NumPy Arrays
X_train = train_data.values.astype(np.float32)
y_train = train_labels.values.astype(np.int64)

X_val = validation_data.values.astype(np.float32)
y_val = validation_labels.values.astype(np.int64)

X_test = test_data.values.astype(np.float32)
y_test = test_labels.values.astype(np.int64)

# Define the model
model = {
    'conv': conv1d(outCh=16, kernelSize=3, stride=1, padding=1),
    'pool': maxpool1d(kernelSize=2, stride=2),
    'fc': fc(input_size=791 * 16, output_size=14)
}

# Train the model (X_train and y_train need to be defined)
train(model, X_train, y_train, epochs=1, lr=0.001)

# Predict on new data (X_test needs to be defined)
predictions = predict(model, X_test)
print(predictions)
print(y_test)
