# 📊 Distribution Tranformation

## 📌 Project Overview

This project focuses on applying **Distribution Transformation types** on two datasets: `train.csv` and `concrete_data.csv`. The aim is to transform skewed data into a more **normal (Gaussian-like) distribution** using different mathematical methods and visualize the results using Histogram and QQ Plots.

* **Function Transformations** were applied on `train.csv`
* **Power Transformations** were applied on `concrete_data.csv`

---

## 🎯 Objectives

* To reduce skewness in the dataset
* To transform data into a normal distribution
* To compare different transformation techniques
* To visualize distribution changes using plots

---

## 🛠️ Tools & Libraries Used

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Scipy

---

## 📂 Datasets

* Dataset 1: `train.csv` (Kaggle dataset)
* Dataset 2: `concrete_data.csv`

---

## ⚙️ Techniques Used

---

## 🔹 1. Function Transformations (on `train.csv`)

Applied basic mathematical transformations to reduce skewness:

### ✅ Log Transformation

* Reduces right skewness

```python
trf=FunctionTransformer(func=np.log1p)
```

---

### ✅ Reciprocal Transformation

* Useful for handling large values

```python
df['rec_col'] = 1 / df['column']
```

---

### ✅ Square Transformation

* Expands large values
* Created function (def apply_transform(transform):) for direct square transformation use
```python
apply_transform(lambda x:x)
```

---

### ✅ Square Root Transformation

* Reduces moderate skewness
* Created function (def apply_transform(transform):) for direct square root transformation use
```python
apply_transform(lambda x:x**2)
```

---

### 📊 Visualization

* Histogram → shows distribution
* QQ Plot → checks normality

---

## 🔹 2. Power Transformations (on `concrete_data.csv`)

### ✅ Box-Cox Transformation

* Works only with **positive values**
* Helps in achieving normal distribution

```python
pt=PowerTransformer(method='box-cox')

X_train_transformed=pt.fit_transform(X_train+0.000001)
X_test_transformed=pt.transform(X_test+0.000001)

pd.DataFrame({'cols':X_train.columns,        #Convert X_train into dataframe
              'box_cox_lamdas':pt.lambdas_   # Each column will be given lamda value
             })
```

---

### ✅ Yeo-Johnson Transformation

* Works with both **positive and negative values**

```python
pt1=PowerTransformer()

X_train_transformed2=pt1.fit_transform(X_train)
X_test_transformed2=pt1.transform(X_test)

lr.fit(X_train_transformed2,y_train)

y_pred2=lr.predict(X_test_transformed2)
r2_score(y_test,y_pred2)

pd.DataFrame({
    'cols':X_train.columns,
    'Yeo_Johnson_lambdas':pt1.lambdas_
})
```

---

### 📊 Visualization

* Histogram → before & after transformation
* QQ Plot → to verify normality

---

## 🔍 Key Insights

* Log and square root transformations effectively reduce skewness
* Reciprocal transformation is useful for highly skewed data
* Box-Cox provides strong normalization for positive data
* Yeo-Johnson is more flexible and widely applicable
* QQ plots help validate normal distribution

---

## 📈 Conclusion

Applying transformation techniques on both **train.csv** and **concrete_data.csv** improved the data distribution significantly, making it more suitable for Machine Learning models that assume normality.

---

## 🚀 Future Work

* Apply transformations in ML model pipelines
* Compare model accuracy before and after transformation
* Explore advanced feature engineering techniques

---
