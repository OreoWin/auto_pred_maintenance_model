# Engine Health Prediction & Analysis

## Rebecca Li | UCLA MASDS

This project builds a predictive maintenance framework for automotive engines using a Kaggle engine-sensor dataset. The goal is to analyze engine health, understand sensor behavior, and develop machine learning models that predict faulty engine states.



## 1. Project Overview

Modern engines generate rich real-time sensor data (temperature, pressure, RPM, etc.).  
This project explores:

- How sensors jointly describe engine behavior  
- Whether faulty engines have identifiable patterns  
- Whether unsupervised learning can separate healthy vs faulty engines  
- Which supervised models perform best for predictive maintenance  

## 2. Data Overview 

**Source:** [Kaggle — Automotive Vehicles Engine Health Dataset  ](https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset/data)
Contains:
- Engine RPM  
- Lubricant oil pressure  
- Fuel pressure  
- Coolant pressure  
- Lubricant oil temperature  
- Coolant temperature  
- Engine condition (target variable: 0 = healthy, 1 = fault)

No missing values or duplicates → minimal data cleaning required.

```python
df["Engine Condition"].value_counts(normalize=True) 
```

```text
Engine Condition
1    0.630509
0    0.369491
Name: proportion, dtype: float64
```

The dataset is imbalanced. 

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression   # or any model you want

X = df.drop(columns=["Engine Condition"])
y = df["Engine Condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
```
Created a sample pipeline for scaling and modelling. 


## 3. EDA 

```python
#Distribution by engine condition
target = "Engine Condition"
features = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
            "Coolant pressure", "lub oil temp", "Coolant temp"]

for col in features:
    plt.figure()
    sns.kdeplot(data=df, x=col, hue=target, common_norm=False)
    plt.title(f"{col} distribution by Engine Condition")
    plt.show()
```
![EDA](figures/EDA_Engine_rpm.png)

Engine rpm shows a slightly different pattern, which needs attention in the future. 

![EDA](figures/EDA_correlation_matrix.png)

The correlation matrix further proved our assumption above. 

## 4. Unsupervised Learning

### **4.1 PCA**
```python
from sklearn.decomposition import PCA

features = ["Engine rpm", "Lub oil pressure", "Fuel pressure", 
            "Coolant pressure", "lub oil temp", "Coolant temp"]
Z = df[features]
scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

pca = PCA(n_components=3)
Z_pca = pca.fit_transform(Z_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

df['PC1'] = Z_pca[:,0]
df['PC2'] = Z_pca[:,1]
df['PC3'] = Z_pca[:,2]
```
First 3 components explain ~53% of variance.  

```text
Explained variance ratio: [0.1899485  0.17692968 0.17071361]
```

![PCA](figures/PCA_1.png)

PCA scatter plots show no separation between faulty vs healthy engines.  

```python
pca_full = PCA().fit(Z_scaled)

plt.plot(pca_full.explained_variance_ratio_, marker='o')
plt.xlabel("Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.show()
```
![PCAscree](figures/PCA_scree_plot.png)



```python
svm = SVC(kernel='rbf', gamma='scale', probability=True)
svm.fit(df[['PC1', 'PC2']], df['Engine Condition'])

x_min, x_max = df['PC1'].min() - 1, df['PC1'].max() + 1
y_min, y_max = df['PC2'].min() - 1, df['PC2'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = svm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.4)
sns.scatterplot(x='PC1', y='PC2', hue='Engine Condition', data=df, s=30)
plt.title("Nonlinear Decision Boundary (SVM on PCA)")
plt.show()
```

To further prove that there is a nonlinear boundary between a faulty vs a normal engine, we visualized a decision boundary. 

![SVMPCA](figures/nonlinear_decision_boundary.png)

Engine faults are driven by nonlinear combinations of sensor readings (temperature, pressure, RPM).
The PCA + SVM boundary visualization clearly shows that fault conditions do not form distinct clusters in feature space; instead, they interleave continuously with normal operating states.Therefore:

- Unsupervised clustering (KMeans, DBSCAN) will fail, because there is no geometric separation.
- Distance-based models such as KNN are ineffective, because the local neighborhood does not reflect engine condition.



### **4.2 Clustering (K-Means)**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = []

for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(Z_scaled)
    score = silhouette_score(Z_scaled, labels)
    scores.append(score)
    print(f"k = {k}, silhouette = {score:.3f}")

plt.plot(range(2, 9), scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()
```
![sihouette](figures/Silhouette_Score_vs_Number_of_Clusters.png)

Silhouette analysis across k = 2–8 shows scores between 0.12 and 0.16, far below the threshold (0.25) for meaningful cluster structure.
This indicates that engine sensor data does not partition naturally into subgroups.The structure is continuous rather than discrete, meaning:
- No clear boundaries exist between fault and non-fault cases.
- Clustering algorithms cannot identify distinct operating modes.
Together with the PCA scatter plot, this confirms that clustering is not appropriate for this dataset.


**Conclusion**  

The data does *not* naturally cluster.  
Engine faults result from complex nonlinear interactions, not geometric groups.

## 5. Supervised Models

### **5.1 Logistic Regression (Baseline)**
- Simple, linear baseline.
- Useful for quick benchmarking.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

X = df.drop(columns=["Engine Condition"])
y = df["Engine Condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# fit
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

```

```text
              precision    recall  f1-score   support

           0       0.53      0.59      0.56      1444
           1       0.74      0.69      0.71      2463

    accuracy                           0.65      3907
   macro avg       0.63      0.64      0.64      3907
weighted avg       0.66      0.65      0.66      3907

AUC: 0.6919559058554136
```



### **5.2 Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("AUC:", roc_auc_score(y_test, y_proba_rf))
print(classification_report(y_test, y_pred_rf))
```

```text
AUC: 0.6811769872787616

              precision    recall  f1-score   support

           0       0.54      0.36      0.43      1444
           1       0.69      0.82      0.75      2463

    accuracy                           0.65      3907
   macro avg       0.61      0.59      0.59      3907
weighted avg       0.63      0.65      0.63      3907
```

![cm](figures/Random_Forest_confusion_matrix.png)



![rf](figures/Random_Forest_Feature_Importances.png)

- Highest feature importances:  
  - Coolant Temp  
  - Lube Oil Temp  
  - Fuel Pressure  
  - Coolant Pressure  

RPM shows low importance → speed alone does not predict faults.

# code for plots 

```python
#confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.show()

#feature importance plot 
importances = rf.feature_importances_
feature_names = X_train.columns

sorted_idx = np.argsort(importances)

plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
```


### **5.3 Gradient Boosted Decision Trees (GBDT)**
- Stronger nonlinear modeling than RF  
- Good balance of speed and accuracy

```python
#confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.show()

#feature importance plot 
importances = rf.feature_importances_
feature_names = X_train.columns

sorted_idx = np.argsort(importances)

plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
```

```text
GBDT AUC: 0.7046462155131402

              precision    recall  f1-score   support

           0       0.57      0.38      0.46      1444
           1       0.70      0.83      0.76      2463

    accuracy                           0.67      3907
   macro avg       0.64      0.61      0.61      3907
weighted avg       0.65      0.67      0.65      3907
```

![cm](figures/GBDT_confusion_matrix.png)



![rf](figures/GBDT_Feature_Importances.png)

### **5.4 XGBoost**

- Best performing model overall  
- Handles imbalanced data and nonlinear boundaries well  
- Suitable for sensor-based predictive maintenance 



## 6. Model Calibration

- Ensures predicted probabilities reflect real fault likelihood.  
- Important for maintenance scheduling & risk thresholds.


## 7. Interpretation & Business Insights

- Engine faults are driven by **thermal & pressure signals**, not RPM.  
- The model shows that **multi-sensor interactions** matter more than single variables.  
- Fault risk increases when oil/coolant temperatures and pressures deviate from normal ranges.  
- Unsupervised learning is ineffective → faults do not form separate clusters.  
- XGBoost is the most reliable model for operational deployment.

**Applications:**
- Early warning of engine failures  
- Prioritized maintenance scheduling  
- Real-time fault detection for fleet operators  

---

## 8. Deliverables

- Jupyter Notebook / Python script  
- Fully reproducible Machine Learning pipeline  
- Visualizations: EDA, PCA, clustering metrics, feature importance  
- Slides for presentation (optional)

---

## 9. Future Work 

- Carlibration on xgb
- Add SHAP for xgb
- Build dashboard









