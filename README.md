
# Fraud Detection using Gaussian Mixture Models for Synthetic Data Generation

## Project Objective

This project tackles the challenge of building an effective fraud detection model from a highly imbalanced dataset. The primary goal is to apply a sophisticated, model-based approach using a **Gaussian Mixture Model (GMM)** to generate synthetic data for the minority (fraudulent) class. The effectiveness of this technique is evaluated by comparing the performance of a classifier trained on the GMM-balanced data against a baseline model trained on the original, imbalanced data.

This repository contains the complete Jupyter Notebook that walks through the entire pipeline, from data analysis to final model evaluation and recommendation.

---

## Dataset

*   **Name:** Credit Card Fraud Detection
*   **Source:** [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
*   **Description:** The dataset contains transactions made by European cardholders in September 2013. It presents a classic case of severe class imbalance, with only **492 fraudulent transactions (0.172%)** out of a total of 284,807. The features are anonymized numerical variables (`V1` through `V28`) resulting from a PCA transformation, along with `Time` and `Amount`.

---

## Methodology

The project is structured into three distinct parts, following a logical progression from establishing a baseline to implementing and evaluating the advanced GMM-based solution.

### Part A: Data Exploration and Baseline Model

1.  **Exploratory Data Analysis (EDA):** The dataset was loaded and analyzed, confirming the extreme class imbalance. Visualizations were created to illustrate the disparity between legitimate and fraudulent transactions.
2.  **Data Preprocessing:** The `Amount` feature was scaled using `StandardScaler`, and the `Time` feature was dropped.
3.  **Baseline Model:** A standard **Logistic Regression** classifier was trained on the original, imbalanced training data.
4.  **Baseline Evaluation:** The model's performance was evaluated on a stratified, imbalanced test set. We focused on metrics like **Precision, Recall, and F1-score** for the fraud class, as overall accuracy is a misleading indicator in this context. The baseline model achieved a high precision but a poor recall of **61%**, meaning it failed to identify 39% of all fraudulent transactions.

### Part B: Gaussian Mixture Model (GMM) for Synthetic Sampling

1.  **Theoretical Foundation:** We established why GMM is theoretically superior to simpler methods like SMOTE, particularly for its ability to model complex, multi-modal distributions that may exist within the fraud class (e.g., different types of fraud).
2.  **GMM Implementation:** A GMM was fitted exclusively to the minority class data from the training set. The optimal number of components (`k`) for the GMM was determined using the **Bayesian Information Criterion (BIC)** to find the best balance between model fit and complexity.
3.  **Hybrid Rebalancing Strategy:** A sophisticated hybrid strategy was employed to create a balanced dataset:
    *   **Clustering-Based Undersampling (CBU):** The majority (legitimate) class was undersampled using `MiniBatchKMeans` to reduce its size to a manageable level while preserving its underlying data structure.
    *   **GMM Oversampling:** The optimized GMM was then used to generate high-quality synthetic samples for the minority (fraud) class, bringing its count up to match the newly reduced majority class size.

This resulted in a new, perfectly balanced training dataset (`X_train_rebalanced`, `y_train_rebalanced`).

### Part C: Performance Evaluation and Conclusion

1.  **Model Retraining:** A new Logistic Regression classifier was trained on the GMM-balanced dataset.
2.  **Comparative Analysis:** This new model was evaluated on the **same original, imbalanced test set** to ensure a fair and realistic comparison against the baseline.
3.  **Final Recommendation:** A conclusive recommendation was made based on the significant performance improvement.

---

## Results and Key Findings

The GMM-based rebalancing strategy yielded a dramatic improvement in the model's ability to detect fraud. The comparison below clearly shows the superiority of the GMM-enhanced model.

| Metric      | Baseline Model (Imbalanced) | GMM-Balanced Model | Change |
| :---------- | :-------------------------: | :----------------: | :----: |
| **Precision** |            0.86             |        0.77        |  -9%   |
| **Recall**    |          **0.61**           |      **0.91**      | **+30%** |
| **F1-Score**  |            0.72             |        0.83        | +11%   |

  <!-- You would replace this with an actual image of your bar chart -->

**Key Takeaways:**
*   **Massive Improvement in Recall:** The model's ability to identify actual fraudulent transactions skyrocketed from 61% to **91%**. This is the most critical metric for a fraud detection system, as it directly relates to minimizing financial losses from missed fraud.
*   **Managed Precision-Recall Trade-off:** While there was a slight decrease in precision (more false positives), the enormous gain in recall resulted in a much higher overall F1-score. This trade-off is highly desirable in a fraud detection context, where the cost of a false negative is significantly higher than a false positive.
*   **Validation of the GMM Approach:** The results empirically validate that GMM is an excellent tool for generating realistic synthetic data that captures the complex nuances of the minority class, leading to a more robust and effective classifier.

---

## How to Run the Code

1.  **Prerequisites:**
    *   Python 3.7+
    *   Jupyter Notebook or JupyterLab
    *   Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

2.  **Setup:**
    *   Clone this repository:
        ```bash
        git clone <repository-url>
        ```
    *   Install the required libraries:
        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn
        ```
    *   Download the dataset `creditcard.csv` from the [Kaggle link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory of the project.

3.  **Execution:**
    *   Open and run the `fraud_detection_gmm.ipynb` notebook in Jupyter. The notebook is self-contained and will execute all steps from data loading to final evaluation.
