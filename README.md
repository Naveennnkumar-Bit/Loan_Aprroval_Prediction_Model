# Loan Approval Prediction

## Overview

This project aims to develop a machine learning model that predicts the loan approval status based on applicant data. Loan approval is a critical process for financing major purchases, and banks rely on applicant data to make informed decisions. While historically, loan officers manually evaluated applications, machine learning has automated and improved the process, enabling quicker and more consistent predictions.

## Data Collection and Features

The dataset used in this project is sourced from platforms like Kaggle and contains applicant details, along with their loan status (approved or not). Key features in the dataset include:

- **Gender**
- **Marital Status**
- **Income**
- **Education**
- **Credit History**
- **Loan Amount**

The target variable, **Loan Status**, indicates whether the loan was approved (yes/no) and is influenced by various applicant characteristics.

## Data Preparation

The machine learning lifecycle starts with data collection and progresses to data cleaning. During the cleaning process, missing values and outliers are handled. The following steps were performed:

1. **Data Cleaning**: Imputation of missing values and handling of outliers using visualization techniques.
2. **Feature Engineering**: Creation of new features, such as **Total Income**, derived from the sum of the applicant and co-applicant incomes. Categorical variables were encoded into numerical formats.
3. **Data Visualization**: Visualizations were used to understand feature distributions and correlations, helping to identify the most relevant predictors for the model.

## Model Selection and Training

Several machine learning models were used for binary classification, including:

- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

The training set was used to train the models, and the testing set was used to evaluate their predictive performance. Model effectiveness was assessed using key metrics such as accuracy, precision, recall, and F1 score.

## Model Evaluation and Comparison

Multiple models were implemented and compared. Logistic Regression achieved an accuracy of around 77-80%. To address the issue of class imbalance, **RandomOverSampler** was applied to oversample the minority class, which resulted in improved model performance and more balanced results.

### Performance Metrics:
- **Accuracy**: The percentage of correctly predicted instances.
- **Precision**: The proportion of positive predictions that were correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between both metrics.

## Conclusion

After addressing class imbalance and retraining the models, improvements in accuracy and classification reports were observed. This project demonstrates the significant potential of machine learning in automating the loan approval process, with the potential for enhanced accuracy through proper data handling and model selection.

## Future Improvements

- Explore advanced techniques like hyperparameter tuning or ensemble methods to improve model performance.
- Investigate other sampling techniques (oversampling/undersampling) to better handle class imbalance.
- Expand the dataset with additional features or more diverse data to further enhance model generalization.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib
- **Machine Learning Models**: Logistic Regression, Decision Trees, Random Forest, KNN

