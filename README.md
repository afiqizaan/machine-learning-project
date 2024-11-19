# Allocating Medical Resources Efficiently: Identifying COVID-19 Severity Based on Medical History

This project leverages machine learning to classify the severity of COVID-19 infections based on patients' medical history. 
By accurately predicting severity levels, healthcare providers can allocate medical resources more effectively, 
ensuring that critical cases receive timely intervention.

## Project Objectives

1. Develop a robust machine learning pipeline to classify COVID-19 severity.
2. Compare the performance of various algorithms, including:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Logistic Regression
3. Provide actionable insights into feature importance for interpretability and clinical relevance.

## Project Structure

```bash
├── Covid_DecisionTree.ipynb        # Decision Tree implementation and analysis
├── Covid_RandomForest.ipynb        # Random Forest implementation and analysis
├── Covid_GradientBoosting.ipynb    # Gradient Boosting implementation and analysis
├── Covid_XGBoost.ipynb             # XGBoost implementation and analysis
├── Covid_LogisticRegression.ipynb  # Logistic Regression implementation and analysis
├── requirements.txt                # List of dependencies
└── README.md                       # Project documentation
```
## Features

- **Interpretability**: Decision Tree for understanding classification logic.
- **Robustness**: Random Forest for handling large datasets with missing values.
- **Performance**: Gradient Boosting and XGBoost for tackling complex patterns in data.
- **Simplicity**: Logistic Regression for baseline comparisons.


## Dataset
The dataset (from Kaggle) used in this project include anonymized patient medical histories and corresponding COVID-19 severity levels. 
Each notebook preprocesses the dataset and applies the respective algorithm.

## Getting Started
### Prerequisites

- Python 3.7+
- Jupyter Notebook or a compatible IDE.

### Installation
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/covid19-severity-classification.git
cd covid19-severity-classification
```
2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Project
1. Open any of the provided notebooks in Jupyter:
```bash
jupyter notebook
```
2. Run the cells in the notebook to preprocess the data,
   train the model, and evaluate its performance.

## Results
- **Model Performance**: Each algorithm was evaluated using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
- **Feature Importance**: Random Forest and XGBoost provided insights into the most critical features affecting COVID-19 severity.

## Known Issues
- **Imbalanced Data**: Addressed using oversampling techniques or class weights.
- **Missing Values**: Handled via imputation methods specific to each algorithm.
