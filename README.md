# Credit Card Fraud Detection using Decision Tree

## Credit Card Fraud Detection using Decision Tree & Trees Ensemble (Random Forest & XGBoost)
![Credit Card](https://media.istockphoto.com/id/1307675090/photo/fraud-alert-concept-with-security-lock-on-fake-credit-cards.jpg?s=612x612&w=0&k=20&c=R2djChH2SEFXNRFOo3K7z_Jd4aZl8Yl5Tn64oFVuhh4=)

# Decision Trees in Machine Learning

**Introduction**

Decision trees (DTs) stand out as powerful non-parametric supervised learning methods. They find applications in both classification and regression tasks. The primary objective of DTs is to create a predictive model for a target variable by learning simple decision rules derived from the features of the data. Two key entities in decision trees are the root node, where the data splits, and decision nodes or leaves, where the final output is obtained.

## Decision Tree Algorithms

### ID3

Developed by Ross Quinlan in 1986, the Iterative Dichotomiser 3 (ID3) algorithm aims to identify categorical features at each node that yield the largest information gain for categorical targets. It allows the tree to grow to its maximum size and then employs a pruning step to enhance its performance on unseen data. The output of this algorithm is a multiway tree.

### C4.5

C4.5, the successor to ID3, dynamically defines discrete attributes that partition continuous attribute values into a set of intervals. This eliminates the restriction of categorical features. C4.5 transforms the ID3-trained tree into sets of 'IF-THEN' rules. To determine the sequence in which these rules should be applied, the accuracy of each rule is evaluated.

### C5.0

Similar to C4.5, C5.0 uses less memory and builds smaller rulesets. It operates by generating binary splits using features and thresholds that yield the largest information gain at each node. While it shares similarities with C4.5, it offers increased accuracy.

### CART

Classification and Regression Trees (CART) algorithm generates binary splits by utilizing features and thresholds that maximize information gain, as measured by the Gini index, at each node. Homogeneity is determined by the Gini index, with higher values indicating greater homogeneity. Unlike C4.5, CART does not compute rule sets and does not support numerical target variables (regression).

## Decision Tree Algorithm Implementation

In practice, the term "decision tree algorithm" commonly refers to a family of algorithms responsible for constructing decision trees. The specific implementation details can vary based on the chosen algorithm and its parameters. Despite these differences, a common theme across these algorithms is the idea of recursively partitioning the data using features and criteria.

This recursive partitioning process involves iteratively making decisions at each node to split the data based on specific conditions. The goal is to create a tree structure where the leaves represent the final outcomes or predictions. The variations among decision tree algorithms lie in how they select features, determine splitting criteria, handle categorical and numerical data, and address overfitting through techniques like pruning.

The flexibility in implementation allows practitioners to choose the decision tree algorithm that best suits their specific use case, considering factors such as interpretability, computational efficiency, and performance on different types of data.

# Dataset
From [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data)

## Description
This dataset contains credit card transactions made by European cardholders in the year 2023. It comprises over 550,000 records, and the data has been anonymized to protect the cardholders' identities. The primary objective of this dataset is to facilitate the development of fraud detection algorithms and models to identify potentially fraudulent transactions.

## Key Features
- **id:** Unique identifier for each transaction
- **V1-V28:** Anonymized features representing various transaction attributes (e.g., time, location, etc.)
- **Amount:** The transaction amount
- **Class:** Binary label indicating whether the transaction is fraudulent (1) or not (0)

## Potential Use Cases
1. **Credit Card Fraud Detection:** Build machine learning models to detect and prevent credit card fraud by identifying suspicious transactions based on the provided features.
2. **Merchant Category Analysis:** Examine how different merchant categories are associated with fraud.
3. **Transaction Type Analysis:** Analyze whether certain types of transactions are more prone to fraud than others.

# Comprehensive Model Evaluation Report:

#### 1. Decision Tree:

- **Accuracy:** 98.59%

- **Classification Report:**
  ```
                 precision    recall  f1-score   support
           0       0.99      0.98      0.99    132815
           1       0.98      0.99      0.99    132814
    accuracy                           0.99    265629
   macro avg       0.99      0.99      0.99    265629
weighted avg       0.99      0.99      0.99    265629
  ```

- **Confusion Matrix:**
  ```
 [[130652   2163]
 [  1572 131242]]
  ```

- **Key Observations:**
  - The decision tree model performs admirably with high precision, recall, and F1-score for both classes.
  - A minor number of false positives and false negatives, as indicated by the confusion matrix, demonstrate the model's robustness.

#### 2. Random Forest:

- **Test Accuracy:** 99.74%

- **Test Classification Report:**
  ```
               precision    recall  f1-score   support
           0       1.00      1.00      1.00    132815
           1       1.00      1.00      1.00    132814
    accuracy                           1.00    265629
   macro avg       1.00      1.00      1.00    265629
weighted avg       1.00      1.00      1.00    265629
  ```

- **Test Confusion Matrix:**
  ```
 [[132653    162]
 [   521 132293]]
  ```

- **Key Observations:**
  - The Random Forest model showcases exceptional performance, achieving near-perfect accuracy and precision for both classes.
  - The confusion matrix indicates minimal misclassifications, highlighting the model's effectiveness.

#### 3. XGBoost:

- **Test Accuracy:** 99.86%

- **Test Classification Report:**
  ```
               precision    recall  f1-score   support
           0       1.00      1.00      1.00    132815
           1       1.00      1.00      1.00    132814
    accuracy                           1.00    265629
   macro avg       1.00      1.00      1.00    265629
weighted avg       1.00      1.00      1.00    265629
  ```

- **Test Confusion Matrix:**
  ```
 [[132492    323]
 [    39 132775]]
  ```

- **Key Observations:**
  - The XGBoost model demonstrates outstanding performance, achieving the highest accuracy among the evaluated models.
  - The confusion matrix reveals minimal misclassifications, highlighting the model's robustness and precision.

### Comparative Analysis:

- All three models exhibit remarkable accuracy, precision, and recall, showcasing their effectiveness in fraud detection.

- Random Forest and XGBoost, being ensemble methods, outperform the standalone Decision Tree, achieving near-perfect accuracy and precision.

- Considering the context of fraud detection, where minimizing false positives is crucial, the high precision values across all models are promising.

- Further investigation into feature importance, model explainability, and potential tuning can provide insights into enhancing model performance and interpretability.

- Model selection should be based on specific application requirements, computational considerations, and interpretability preferences.

### Conclusion:

- In conclusion, the ensemble models, Random Forest and XGBoost, stand out for their exceptional performance in fraud detection.

- The decision tree model, while robust, is surpassed by the ensemble models in achieving higher accuracy and precision.

- Continuous monitoring, periodic retraining, and model interpretability assessments are recommended for maintaining optimal fraud detection capabilities
