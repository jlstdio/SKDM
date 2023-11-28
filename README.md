# Vibration based keyboard decoding model
파일 및 데이터 활용 및 각 파일에 대한 설명입니다.

## files & data download
다음 링크에서 받아주세요

Link : https://drive.google.com/drive/folders/1bn99L3aeaEJMU354ZHrfB2t-6j78ompR?usp=sharing

## train & predict
### pytorch 사용하기 (빠름)
- 학습에 필요한 데이터는 `keyboard_tap_full.csv`에 있습니다. 필요에 따라 dir을 수정해주세요.

- 사용법
`main.py`를 실행합니다.

지정된 epoch(default : 100)후 자동으로 prediction을 진행합니다.

- trouble shooting
가끔 `pandas.errors.ParserError: Error tokenizing data. C error: Expected 12 fields in line 12852, saw 23` 과 같은 에러가 발생합니다.

원인을 찾지 못하였으나 해당 csv 파일을 열어서 line에 적힌 숫자에 해당되는 행을 지우면 해결됩니다.

### homemadeCnn 사용하기 : numpy로만 만든 cnn (느림)
`homemadeCnn.py`을 실행합니다.

# file 설명
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
