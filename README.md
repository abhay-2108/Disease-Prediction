
# Human Disease Prediction: Leveraging Machine Learning for Early Diagnosis

**Project Overview:**

This project aims to develop a robust machine learning model capable of predicting human diseases based on patient-reported symptoms and key health indicators. Early and accurate disease prediction is crucial for timely intervention, improved patient outcomes, and efficient healthcare resource allocation. By leveraging machine learning, we can analyze complex patterns within patient data to identify potential diseases, even before they manifest in severe symptoms. This project focuses on building a predictive model using a Random Forest Classifier within a scikit-learn pipeline, ensuring efficient data preprocessing and model training.

**The Need:**

Traditional diagnostic methods often rely on time-consuming clinical assessments and laboratory tests. Machine learning offers a promising avenue to expedite this process, particularly in resource-constrained settings or for preliminary screening. By analyzing readily available patient data, such as age, gender, symptoms like fever and cough, and basic medical measurements like blood pressure and cholesterol levels, our model aims to provide valuable insights for healthcare professionals. This can lead to earlier detection, prompt treatment, and ultimately, better patient care.

**Dataset:**

The dataset used in this project, `disease.csv`, contains a collection of patient records with the following features:

-   **Age**: Patient's age (numerical).
-   **Gender**: Patient's gender (categorical).
-   **Fever**: Presence of fever (categorical).
-   **Cough**: Presence of cough (categorical).
-   **Fatigue**: Presence of fatigue (categorical).
-   **Difficulty Breathing**: Presence of breathing difficulties (categorical).
-   **Blood Pressure**: Patient's blood pressure level (categorical).
-   **Cholesterol Level**: Patient's cholesterol level (categorical).
-   **Disease**: The target variable, indicating the predicted disease (categorical).

**Dependencies:**

The following Python libraries are required to run this project:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `ydata-profiling`

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ydata-profiling
```

**Usage:**

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/abhay-2108/Disease-Prediction.git](https://github.com/abhay-2108/Disease-Prediction.git)
    cd Disease-Prediction
    ```

2.  **Install the dependencies:**

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn ydata-profiling
    ```

3.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook disease_prediction.ipynb
    ```

    Follow the instructions within the notebook to execute the code and reproduce the results.

**Model Evaluation Metrics:**

The model's performance is assessed using standard classification metrics, including:

-   **Precision:** The proportion of correctly predicted positive cases out of all predicted positive cases.
-   **Recall (Sensitivity):** The proportion of correctly predicted positive cases out of all actual positive cases.
-   **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
-   **Accuracy:** The overall proportion of correctly classified instances.
-   **Classification Report:** A comprehensive report from scikit-learn providing precision, recall, F1-score, and support for each class.

These metrics are crucial for evaluating the model's ability to accurately predict diseases and ensure its reliability in practical applications.

**License:**

This project is licensed under the [MIT License](LICENSE) (if you create one).
