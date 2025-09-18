# MLOps Assignment 1

This project demonstrates comprehensive MLOps practices including model training, experiment tracking with MLflow, and model registry. It implements a complete machine learning pipeline for iris flower classification.

## Project Structure

```
mlops-assignment-1/
├── data/           # Dataset files and train/test splits
├── notebooks/      # Jupyter notebooks for exploration and analysis
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/           # Source code for models and utilities
│   ├── utils.py
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── svm.py
│   └── mlflow_config.py
├── models/        # Saved trained models
├── results/       # Results, plots, and outputs
└── README.md      # Project documentation
```

## Dataset

This project uses the **Iris dataset**, a classic dataset for classification tasks containing:
- **150 samples** of iris flowers
- **4 features**: sepal length, sepal width, petal length, petal width
- **3 species**: Setosa, Versicolor, Virginica
- **Balanced dataset**: 50 samples per species

## Models Implemented

1. **Logistic Regression** - Linear classification model with feature scaling
   - Accuracy: 93.33%
   - Precision: 93.33%
   - Recall: 93.33%
   - F1-Score: 93.33%

2. **Random Forest** - Ensemble method using decision trees
   - Accuracy: 90.00%
   - Precision: 90.24%
   - Recall: 90.00%
   - F1-Score: 89.97%
   - Feature importance analysis included

3. **Support Vector Machine (SVM)** - Kernel-based classification (**Best Model**)
   - Accuracy: **96.67%**
   - Precision: 96.97%
   - Recall: 96.67%
   - F1-Score: 96.66%

## MLflow Tracking & Experiment Management

### Experiment Setup
- **Experiment Name**: `iris-classification`
- **Tracking**: All model parameters, metrics, and artifacts logged
- **Artifacts**: Confusion matrices, model comparison plots, trained models

### Logged Information
- **Parameters**: Model hyperparameters, configuration settings
- **Metrics**: Accuracy, precision, recall, F1-score for each model
- **Artifacts**: 
  - Confusion matrices for each model
  - Model comparison visualizations
  - Trained model objects

### MLflow UI Access
```bash
mlflow ui
```
Navigate to `http://localhost:5000` to view experiment tracking dashboard.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-assignment-1
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data exploration**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

4. **Train models and track experiments**
   ```bash
   jupyter notebook notebooks/02_model_training.ipynb
   ```

5. **Launch MLflow UI**
   ```bash
   mlflow ui
   ```

## Model Registration & Deployment

### Best Model Selection
The **Support Vector Machine (SVM)** was identified as the best performing model based on accuracy (96.67%).

### Model Registry Process
1. **Automatic Selection**: Best model identified based on accuracy metric
2. **Registration**: Model registered in MLflow Model Registry as `iris-classifier-best`
3. **Versioning**: Model version 1 created with performance metadata
4. **Description**: "Best performing iris classifier with accuracy: 96.67%"

### Model Registry Details
- **Model Name**: `iris-classifier-best`
- **Version**: 1
- **Stage**: None (newly registered)
- **Run ID**: `ed67ca2912844ef881a41fd994f9d8c8`
- **Model Type**: Support Vector Machine with RBF kernel

### Accessing Registered Model
```python
import mlflow.sklearn

# Load the registered model
model = mlflow.sklearn.load_model("models:/iris-classifier-best/1")

# Make predictions
predictions = model.predict(X_test)
```

## Results & Visualizations

### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| SVM | **96.67%** | **96.97%** | **96.67%** | **96.66%** |
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| Random Forest | 90.00% | 90.24% | 90.00% | 89.97% |

### Generated Artifacts
- **Pairplot**: Feature relationships by species
- **Correlation Matrix**: Feature correlation heatmap
- **Confusion Matrices**: Per-model classification results
- **Model Comparison**: Performance metrics visualization

## MLflow Monitoring & Observability

### Experiment Tracking
- **3 successful runs** logged to MLflow
- **Comprehensive metrics** tracked for each model
- **Reproducible experiments** with seeded random states
- **Artifact logging** for analysis and debugging

### Model Performance Monitoring
- Real-time metric comparison across models
- Historical run tracking and comparison
- Parameter impact analysis available through MLflow UI

## Assignment Completion Status

✅ **Part 1 - GitHub Setup**: Repository created with proper structure  
✅ **Part 2 - Model Training**: 3 models trained with comprehensive evaluation  
✅ **Part 3 - MLflow Tracking**: Complete experiment tracking with artifacts  
✅ **Part 4 - Model Registration**: Best model registered with version control  

**Final Score**: SVM achieved **96.67% accuracy** and is registered as the production model.