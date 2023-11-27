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
`Deep Neural Network(Embedding + Sigmoid Gate Unit)`  
`AutoML`  
### Architecture Details:

- **Initialization and Embedding Layers**:
  - The model initializes with `input_dims` and `embedding_dim` parameters, determining the dimensions of the input and embedding layers.
  - Embedding layers are created for each dimension in `input_dims`, facilitating the handling of categorical data through embeddings.

- **Main Layers**:
  - The model includes dropout layers (`nn.Dropout(0.2)`) to prevent overfitting. There are five such layers, each associated with the subsequent processing steps.
  - Gate units (`nn.Linear`), a series of five layers, are used to control the flow of information through the network. They employ a sigmoid function to regulate the contribution of each input.
  - Linear layers are integrated to further process the input data. These layers work in tandem with the gate units, enhancing the model's ability to capture complex relationships in the data.

- **Forward Pass and Output Layers**:
  - In the forward pass, the model first computes embeddings for each input feature and concatenates them.
  - It then applies a series of dropout, gate, and linear operations to process the concatenated embeddings.
  - The model features ten output layers (`nn.Linear`), each producing a prediction. The final output is the mean of these predictions, giving a consolidated result.


## Preprocessing

The preprocessing phase in our project is meticulously designed to refine the dataset, enhancing its suitability for the AI models. This phase includes a series of critical steps, each tailored to address specific aspects of our data.

### Categorical Data Encoding
- **Label Encoding:** We employ label encoding for categorical columns in both the training and test datasets. This process converts categorical variables, such as item codes and location codes, into a numeric format. It's crucial for models that require numerical input.

### Outlier Removal
- **Custom Outlier Handling:** The dataset is scrutinized for outliers, which are handled in a bespoke manner. For example, prices are set to 0 under specific conditions (like certain weekdays or item IDs). This step is essential for maintaining data quality and model accuracy.

### Feature Extraction and Enhancement
- **Time-based Features:** From the `timestamp` data, we extract crucial time-based features such as year, month, day, weekday, and weekend flags. These features are pivotal in understanding time-related trends affecting product prices.
- **Seasonal and Holiday Features:** We incorporate seasonality and holiday effects into our dataset through custom functions like `group_season` and `holiday`. These features help capture the seasonal dynamics and special occasions that might influence market prices.
- **Item-specific Composite Features:** The dataset is enriched with composite features crafted by combining various attributes, such as item ID, location, corporation, etc. This approach aims to uncover complex patterns that simple, singular data points might not reveal.
- **Harvest Weight Calculation:** We use a custom function, `determine_harvest_weight`, to calculate a 'harvest weight' based on the item type and the month. This unique feature could provide insights into supply volume variations throughout the year.





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


