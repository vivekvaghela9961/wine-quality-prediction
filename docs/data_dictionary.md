# Data Dictionary — Wine Quality Dataset

This project utilizes the Red and White Wine Quality datasets from the UCI Machine Learning Repository. Both datasets contain the same chemical features, with a sensory quality score as the target variable.

## Features

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **fixed acidity** | Numeric (float) | Most acids involved with wine or fixed or nonvolatile (do not evaporate readily). |
| **volatile acidity** | Numeric (float) | The amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste. |
| **citric acid** | Numeric (float) | Found in small quantities, citric acid can add 'freshness' and flavor to wines. |
| **residual sugar** | Numeric (float) | The amount of sugar remaining after fermentation stops. Rare to find < 1 g/L; > 45 g/L is considered sweet. |
| **chlorides** | Numeric (float) | The amount of salt in the wine. |
| **free sulfur dioxide** | Numeric (float) | The free form of SO2; prevents microbial growth and the oxidation of wine. |
| **total sulfur dioxide** | Numeric (float) | Amount of free and bound forms of SO2; detectable at high levels. |
| **density** | Numeric (float) | The density of wine, close to water depending on the percent alcohol and sugar content. |
| **pH** | Numeric (float) | Describes how acidic or basic a wine is (usually between 3.0 and 4.0). |
| **sulphates** | Numeric (float) | A wine additive which contributes to SO2 levels (acts as antimicrobial and antioxidant). |
| **alcohol** | Numeric (float) | The percent alcohol content of the wine. |

## Target

| Target Name | Type | Description |
| :--- | :--- | :--- |
| **quality** | Integer | Output variable based on sensory data. Integer score between 0 and 10 (higher is better). |
