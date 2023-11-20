# 제주 특산물 가격 예측 AI 경진대회

### 제주도 특산물의 가격을 예측하는 AI 모델 개발 및 인사이트 발굴
  ![Python Version](https://img.shields.io/badge/Python-3.8.10-blue)   
제주도에는 다양한 특산물이 존재하며, 

그 중 양배추, 무(월동무), 당근, 브로콜리, 감귤은 제주도의 대표적인 특산물들 중 일부입니다. 

이런 특산물들의 안정적이고 효율적인 수급을 위해서는 해당 특산물들의 가격에 대한 정확한 예측이 필요합니다.


## Project structure

```
jeju-price-prediction
├─ .gitignore
├─ README.md
├─ archive
│  └─ archive file
├─ data  # 원본 데이터 
├─ garbage
│  ├─ gate_model.py
│  ├─ temp copy.ipynb
│  └─ temp.ipynb
├─ main.py  # DNN train & inference
├─ AutoGluon.ipynb  # ML train & inference
├─ notebooks 
│  └─ EDA
├─ result
│  └─ Result file
└─ src
   ├─ config  # config file
   │  └─ config.py
   ├─ data  # feature_extraction & preprocessing
   │  ├─ dataset.py
   │  ├─ feature_extraction.py
   │  └─ preprocessing.py
   ├─ importance.py
   ├─ inference  
   │  └─ inference.py
   ├─ model  # DNN model
   │  └─ gate_unit.py
   └─ train  # train loop
      └─ train.py

```

## Data

```
Dataset Info.

1. train.csv
train 데이터 : 2019년 01월 01일부터 2023년 03월 03일까지의 유통된 품목의 가격 데이터
item: 품목 코드
corporation : 유통 법인 코드
법인 A부터 F 존재
location : 지역 코드
supply(kg) : 유통된 물량, kg 단위
price(원/kg) : 유통된 품목들의 kg 마다의 가격, 원 단위

2. test.csv
test 데이터 : 2023년 03월 04일부터 2023년 03월 31일까지의 데이터
```
## Experiment
#### Model
`Deep Neural Network(Embedding + Sigmoid Gate Unit)`  
`AutoML`
#### Feature
`time과 item을 활용한 Feature Engineering `  
각 case별 평균 및 중앙값을 이용한 이상치 처리 및 변수간 차이를 줄 수 있는 feature를 중심으로 생성  



## Development Environment
```
OS: Ubuntu 20.04 LTS
CPU: Intel i9-14900K
RAM: 128GB
GPU: NVIDIA GeFocrce RTX4090
```

## Host
주최: 제주특별자치도  
주관: 제주테크노파크, 데이콘  
기간: 2023.10.26 ~ 2023.11.20   
[Competition Link](https://dacon.io/competitions/official/236176/overview/description)
