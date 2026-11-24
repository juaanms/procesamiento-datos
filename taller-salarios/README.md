# Income Prediction with PySpark MLlib — Census Income Dataset

**Pontificia Universidad Javeriana** · *Large-Scale Data Processing (Procesamiento de Datos a Gran Escala)*

- **Author:** Juan Pablo Mórtigo Salinas
- **Professor:** John Jairo Corredor Franco
- **Notebook:** [`MortigoSalario_1055.ipynb`](MortigoSalario_1055.ipynb) (written in Spanish)

---

## Objective

Build and compare **five MLlib classification models** that predict whether a person earns **more than USD 50,000 per year**, based on U.S. census data. The data is stored in **Hadoop HDFS** and processed with **PySpark**, following the **CRISP-DM** methodology.

## Dataset

**Census Income (Adult)** from the UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/20/census+income>

- Extracted by Barry Becker from the **1994 U.S. Census** database.
- File used: `adult.data`, with **32,561 records** and **15 columns** (age, work class, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country, among others).
- Target variable: `income` (`<=50K` or `>50K`), recoded as `IngresoAlto` (1 = earns more than 50K).

## Tech Stack

| Tool | Use |
|---|---|
| Apache Spark (PySpark) | Distributed data processing |
| Spark MLlib | Feature engineering and classification models |
| Hadoop HDFS | Data storage (datalake of the course cluster) |
| Pandas, Matplotlib, Seaborn | Charts and exploratory analysis |
| scikit-learn | Confusion matrices |

## Methodology (CRISP-DM)

| CRISP-DM phase | Notebook section |
|---|---|
| 1. Business understanding | Context and objectives |
| 2. Data understanding | 2. Data loading into HDFS · 4. Descriptive statistics and charts |
| 3. Data preparation | 3. Diagnosis and cleaning · 5. Transformation for MLlib |
| 4. Modeling | 6. Five MLlib models |
| 5. Evaluation | 7. Model comparison |
| 6. Conclusions | 8. Conclusions and recommendations |

## Data Preparation

1. Uploaded `adult.data` to HDFS and loaded it with PySpark (the file has no header).
2. Renamed the columns, trimmed whitespace and cast numeric columns to integer.
3. Replaced missing values (marked as `"?"`) with the **mode** of each column: `workclass` (5.64%), `occupation` (5.66%) and `native-country` (1.79%).
4. Removed **24 duplicate records**, leaving **32,537** records.
5. Dropped `education` (redundant with `education-num`), indexed the 7 categorical variables with `StringIndexer` and built a 13-feature vector with `VectorAssembler`.
6. Split the data into **80% train / 20% test** (`seed = 2026`).

## Key Findings (EDA)

- **24.1%** of people earn more than 50K, so the classes are moderately imbalanced. A model that always predicts `<=50K` would reach about 76% accuracy.
- **Education** shows the clearest relationship: 16% of high-school graduates earn >50K, compared with 41.5% of bachelor's graduates and 74.1% of doctorate holders.
- **Age:** the share of high incomes peaks between **41 and 60 years** (about 38%).
- **Marital status, occupation and hours worked** also matter. Married people (44.7%) and executive/managerial roles (48.4%) have the highest shares of high incomes.
- There are clear **gaps by sex and race**: 30.6% of men earn >50K, compared with 11% of women. These gaps reflect the inequalities of the 1994 census.

## Models and Results

All five models follow the same procedure: train → predict → accuracy / weighted precision / weighted recall → confusion matrix. The test set has 6,613 people, 1,557 of whom earn >50K.

| Model | Accuracy | Weighted precision | Weighted recall | >50K detected | False positives |
|---|---|---|---|---|---|
| Logistic Regression | 84.29% | 83.41% | 84.29% | 830 (53.3%) | 312 |
| Decision Tree | 84.79% | 84.01% | 84.79% | 799 (51.3%) | 248 |
| **Random Forest** | **86.21%** | **85.73%** | **86.21%** | **841 (54.0%)** | **196** |
| Naive Bayes | 79.80% | 77.77% | 79.80% | 459 (29.5%) | 238 |
| Multilayer Perceptron | 76.74% | 82.17% | 76.74% | 19 (1.2%) | 0 |

*Baseline: always predicting `<=50K` gives 76.45% accuracy on the test set.*

## Conclusions

- **Random Forest is the best model.** It leads every metric, detects the most high-income people and has the fewest false positives.
- **Logistic Regression and Decision Tree** perform almost as well (about 84–85% accuracy) and are easier to interpret.
- **Naive Bayes** underperforms because its independence and normality assumptions do not hold for this data.
- **The Multilayer Perceptron barely learned:** its accuracy is almost the same as the baseline, mainly because the features are **not scaled** (for example, `fnlwgt` reaches 1.48 million).
- Even the best models miss about **46% of high-income people**, because they lean toward the majority class when uncertain.

## Recommendations

1. Use **Random Forest**. If explainability matters more, use Logistic Regression or a Decision Tree instead.
2. Handle the class imbalance with class weights (`weightCol`) or by tuning the decision threshold.
3. Scale the features (`StandardScaler`), especially for Logistic Regression and the Multilayer Perceptron.
4. Tune hyperparameters with `CrossValidator` and `ParamGridBuilder`.
5. Review the use of sensitive variables (sex and race) to avoid reproducing historical inequalities.
6. Validate the final model on `adult.test` (16,281 unseen records).

## How to Run

1. Download the dataset from UCI and upload `adult.data` to the Jupyter server, in the same folder as the notebook.
2. Open `MortigoSalario_1055.ipynb`. The notebook creates the HDFS folder, uploads the file and reads it from HDFS.
3. Adjust the Spark/Hadoop paths, the HDFS address and the Spark UI port to your own cluster.
4. Run all cells (**Kernel → Restart & Run All**).

> **Note:** the notebook was developed and executed on the course cluster (Spark + Hadoop HDFS) provided by the professor.
