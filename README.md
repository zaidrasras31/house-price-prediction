# House Price Prediction Using Machine Learning

## 1. Project Overview

This project predicts residential house sale prices using supervised machine learning techniques.

The project was developed using the **House Prices – Advanced Regression Techniques** competition on Kaggle.

The main objective is to explore regression algorithms, apply appropriate preprocessing techniques, evaluate model performance, and generate predictions for unseen houses.

## 2. Problem Statement

House prices depend on several factors, including living area, overall quality, neighborhood, construction year, and other property characteristics.

The goal is to build a regression model that predicts `SalePrice` using 79 input features.

## 3. Technologies Used

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Environment:** Kaggle Notebooks, Jupyter Notebook

## 4. Dataset

**Source:** [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

| Dataset | Number of Houses |
|---|---:|
| Training | 1,460 |
| Testing | 1,459 |

The dataset contains 79 explanatory features and one target variable, `SalePrice`.

## 5. Machine Learning Workflow

### Exploratory Data Analysis

- Examined the dataset structure and feature types.
- Investigated missing values.
- Visualized the distribution of house prices.
- Explored relationships between house characteristics and sale prices.

### Data Preprocessing

- Applied a logarithmic transformation to `SalePrice`.
- Handled missing numerical values using median imputation.
- Handled missing categorical values.
- Encoded categorical features using OneHotEncoder.
- Applied StandardScaler to numerical features.
- Used scikit-learn Pipelines and ColumnTransformer to organize preprocessing and reduce data leakage.

### Model Development

Trained and compared four regression models:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regression
4. XGBoost Regression

Used 5-fold cross-validation and hyperparameter tuning to evaluate model performance.

## 6. Model Performance

| Model | Mean CV RMSE |
|---|---:|
| Linear Regression | 0.16588 |
| Ridge Regression | 0.14249 |
| Tuned Random Forest | 0.14704 |
| **XGBoost Regression** | **0.12461** |

XGBoost achieved the lowest cross-validation error among the tested models.

### Final XGBoost Results

| Metric | Result |
|---|---:|
| Training RMSE | 0.06799 |
| Validation RMSE | 0.12552 |
| Kaggle Score | **0.12795** |
| Leaderboard Rank at Submission | **1,693 / 4,474** |
| Leaderboard Position | **Top 38%** |

The RMSE values are evaluated in log-transformed price space.

## 7. Project Structure

```text
house-price-prediction/
├── README.md
└── house-prices.ipynb
```

## 8. Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

## 9. Usage

1. Download the notebook from this repository.
2. Download the competition dataset from Kaggle.
3. Update the dataset file paths if running locally.
4. Run the notebook cells in order.
5. Generate `submission.csv` using the final XGBoost model.

The notebook was originally developed in Kaggle Notebooks.

## 10. Key Learnings

This project helped develop practical understanding of:

- Regression algorithms and their differences.
- Data preprocessing and missing-value handling.
- Target leakage and its prevention.
- Regularization and overfitting.
- Cross-validation and hyperparameter tuning.
- Machine learning pipelines.
- Model generalization and evaluation.

## 11. Future Improvements

Potential improvements include:

- Additional feature engineering.
- More extensive hyperparameter optimization.
- Feature importance analysis.
- Experimenting with additional ensemble methods.

## 12. Conclusion

XGBoost achieved the best results among the evaluated regression models, with a Kaggle score of **0.12795**.

The project provided practical experience in building an end-to-end supervised machine learning workflow, from exploratory data analysis to Kaggle submission.

## License

This project is available for educational and portfolio purposes.
