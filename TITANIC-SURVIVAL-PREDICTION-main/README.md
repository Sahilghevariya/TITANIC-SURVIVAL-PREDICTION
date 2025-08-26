# 🚢 Titanic Survival Prediction

A machine learning project to predict passenger survival on the Titanic using demographic and travel data. This project demonstrates data preprocessing, feature engineering, and building classification models with evaluation metrics.

📂 Dataset

- Source: [Titanic Dataset - GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- Features:
  - PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Target:
  - Survived (0 = No, 1 = Yes)

## 🛠️ Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- VSCode

## 🧹 Data Preprocessing

1. Dropped irrelevant columns:
   - `Name`, `Ticket`, `Cabin`
2. Handled missing values:
   - `Age`: filled with the median
   - `Embarked`: filled with the mode
3. Converted categorical variables to numeric using:
   - **One-Hot Encoding** on `Sex` and `Embarked`
4. Applied **feature scaling** using:
   - `StandardScaler` for Logistic Regression

## 🤖 Models Implemented

### 1. Logistic Regression
- Trained on **scaled features**
- Accuracy: 80%
- Evaluated with:
  - Precision, Recall, F1-Score
  - Confusion Matrix

### 2. Decision Tree Classifier
- Trained without scaling
- Accuracy: 72%
- Same evaluation metrics as above

## 📈 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## ▶️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
````

2. **Install dependencies**

```bash
pip install pandas numpy scikit-learn jupyter
```

3. **Run the Jupyter Notebook**

```bash
jupyter notebook
```

Open `Titanic_Survival_Prediction.ipynb` to explore the full workflow.

## 📊 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 80%      |
| Decision Tree       | 72%      |

## 🚀 Future Enhancements

* Visualizing confusion matrices with heatmaps
* Hyperparameter tuning (GridSearchCV)
* Model deployment using Flask/Streamlit
* Adding SVM, Random Forest, or XGBoost models

## 👨‍💻 Author

Meet Limbachiya

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
