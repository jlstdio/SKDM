# Mobile sensor based keyboard decoding model (Acc, Audio)
Using accelerometer and microphone data to decode victim's keyboard input without any electronic hacking.

> **I Strongly prohibit any usage of this model and concept to be used in exploit situation.**

> **Therefore, I'm not going to open data sets and exact methods directly to prevent exploit usage of method.**

## How it works?
<img width="500" alt="introduction" src="https://github.com/JoonLee-K/SKDM/assets/35446381/239a0a94-ed0b-4397-ade6-333495077972">

### Training phase
Record vibration & audio of the nearby device with a keypad using a smartphone.

from 0 to 9 and 'z', 'x', 'c', 'v', total of 14 classes of keyboard data is collected.

MacBook Pro, Macbook Air, Gaming Laptop, Wireless keyboard are used. Cafe noise and silent noise are combined.

Run through feature extractor & neural net to train.

### Test phase
Split data for test & validation is used for testing.

### Real-world usage
TBA, Demonstration of the project will be announced shortly


## Finding a right picking place
<img width="500" alt="Screenshot 2023-11-28 at 17 53 11" src="https://github.com/JoonLee-K/SKDM/assets/35446381/36de7385-7db2-4a08-a047-ef68c21a290e">

by comparing acc & audio data in plot I found that upper part of the laptop picks the most sharp edges of data

## Data Preparation
### Record data
<img width="300" alt="data collection" src="https://github.com/JoonLee-K/SKDM/assets/35446381/975d1c5f-c2b6-4061-a29c-7bda239b0e75">

I've collected data using [Sensor Logger](https://play.google.com/store/apps/details?id=com.kelvin.sensorapp&hl=ko&gl=US)application. Turn audio & accelerometer data option on.

### Split data
<img width="500" alt="data preparation" src="https://github.com/JoonLee-K/SKDM/assets/35446381/dc9f0c29-46c8-44eb-bfba-c264f1f3917e">

### Feature Extraction
<img width="500" alt="feature extraction" src="https://github.com/JoonLee-K/SKDM/assets/35446381/12e5c924-0427-4eec-b5dd-9125c0e0eb87">

for Accelerometer, Auto-Correlation, PSD, and FFT were used for feature extraction.

and for Audio, MFCC, and FFT were used.

## Train & Test
### neural net
Inspired from [TapNet](https://arxiv.org/abs/2009.01469) we've made simple convolutional layers

### result
<img width="300" alt="Screenshot 2023-11-28 at 18 02 13" src="https://github.com/JoonLee-K/SKDM/assets/35446381/7b2724b2-13f2-4b98-9419-6e4b01cf03e2">

The best data is 93% accuracy.

This model also had F1 score: 0.8990. Precision: 0.8812 and Recall: 0.9175


# file description
- util.py : 연산에 필요한 각종 함수가 있습니다. CSV에 쓰기 등이 있습니다
- feature Extractor.py : 데이터에 대한 featureExtractor함수가 들어있습니다.
- data Preprocess.py : raw 데이터를 1 stroke에 맞게 가공하고, feature extraction을 수행합니다
  - slice() : 데이터를 자르는 interface입니다.
  - 이후 자동으로 데이터에 대한 feature extraction이 실행됩니다.
  - range(10, 14, 1) : range값을 조절하여 처리하고자하는 데이터를 지정합니다.
  - 데이터는 CSV가 존재할 경우 이어쓰기를 합니다. 필요시 파일을 삭제후 진행하십시오.
- backup : 원본 데이터의 사본이 있습니다.
- experimentData : 아이디어 검증 및 센서 데이터 수집 위치를 결정하기 위한 프로젝트입니다.
- feature extraction을 구현하고, feature의 가짓수와 종류를 결정하는 프로젝트입니다.
- data : raw한 데이터가 들어있습니다.
