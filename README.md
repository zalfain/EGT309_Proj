# EGT309_Proj

## 1. Project Overview

This project focuses on developing a predictive maintenance machine learning system using machine sensor readings. The objective is to identify patterns associated with maintenance requirements and develop models that can predict whether maintenance is needed.

## 2. Exploratory Data Analysis (EDA)

The dataset contains 100,000 records and 13 columns.

### 2.1 Data Quality Findings

- Temperature contains 14,000 missing values (14%).
- Vibration contains 6,703 negative readings (6.7%).
- No duplicated records were identified.
- All other columns contain no missing values.

### 2.2 Target Distribution

The maintenance_required target is imbalanced:

- 80.3% of records do not require maintenance.
- 19.7% of records require maintenance.

This imbalance should be considered when selecting machine learning evaluation metrics.

### 2.3 Visualisation Findings

- Temperature readings are generally centred around 75°C.
- Records requiring maintenance have a higher median temperature than records not requiring maintenance.
- Negative vibration readings appear as unusual outliers across both maintenance categories.
- Humidity, pressure and energy consumption show relatively uniform distributions.

### 2.4 Correlation Findings

Temperature has the strongest positive linear correlation with maintenance_required (0.28), followed by vibration (0.05).

Humidity, pressure and energy consumption show almost no linear correlation with the target individually.

### 2.5 Potential Data Leakage

The following variables demonstrate strong relationships with the target:

- machine_status
- anomaly_flag
- failure_type
- downtime_risk

For example, records with machine_status = 2 and anomaly_flag = 1 all correspond to maintenance_required = 1.

These variables are provisionally excluded from the initial model features because they may reveal information about existing machine failures rather than predict future maintenance requirements.

predicted_remaining_life is also provisionally excluded pending verification of its availability at prediction time.

## 3. Data Cleaning and Feature Engineering

The preprocessing module is implemented using a Python class named DataPreprocessor.

The following operations are performed:

1. Missing temperature values are replaced using median imputation.
2. Negative vibration readings are treated as missing values and replaced using median imputation.
3. The timestamp column is converted into datetime format.
4. Three time-based features are extracted:
   - hour
   - day_of_week
   - month
5. The original timestamp column is removed after feature extraction.
6. Potential data leakage columns are excluded through configuration settings.

Median values are calculated during the fitting stage using training data to avoid using information from the testing dataset.

## 4. Configuration

Project settings are centralised in config.yaml.

This allows dataset paths, target column settings, preprocessing options, feature extraction and excluded columns to be modified without directly changing the preprocessing implementation.