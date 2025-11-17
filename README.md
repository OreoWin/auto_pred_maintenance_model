# Engine Health Prediction & Analysis

This project builds a predictive maintenance framework for automotive engines using a Kaggle engine-sensor dataset. The goal is to analyze engine health, understand sensor behavior, and develop machine learning models that predict faulty engine states.

---

## 1. Project Overview

Modern engines generate rich real-time sensor data (temperature, pressure, RPM, etc.).  
This project explores:

- How sensors jointly describe engine behavior  
- Whether faulty engines have identifiable patterns  
- Whether unsupervised learning can separate healthy vs faulty engines  
- Which supervised models perform best for predictive maintenance  

The final output includes:
- Exploratory Data Analysis (EDA)
- Unsupervised learning experiments (PCA, K-Means)
- Supervised classification models (Logistic Regression, Random Forest, GBDT, XGBoost)
- Model calibration & interpretability
- Business insights for engine diagnostics

---

## 2. Dataset

**Source:** Kaggle — Automotive Vehicles Engine Health Dataset  
Contains:
- Engine RPM  
- Lubricant oil pressure  
- Fuel pressure  
- Coolant pressure  
- Lubricant oil temperature  
- Coolant temperature  
- Engine condition (target variable: 0 = healthy, 1 = fault)

No missing values or duplicates → minimal data cleaning required.

---

## 3. EDA Summary

- Sensor distributions show distinct operating ranges across engines.  
- Temperatures & pressures show stronger association with faults than RPM.  
- No obvious visual separation between healthy vs faulty groups → suggests nonlinear relationships.  

---

## 4. Unsupervised Learning

### **4.1 PCA**
- First 3 components explain ~53% of variance.  
- Variance is evenly distributed → sensors capture different dimensions of engine behavior.  
- PCA scatter plots show **no separation** between faulty vs healthy engines.  
- Confirms that dimensionality reduction is **not beneficial**.

### **4.2 Clustering (K-Means)**
Silhouette scores for k = 2–8 range only **0.12–0.16**  
→ far below the threshold (~0.25) for meaningful cluster structure.

**Conclusion:**  
The data does *not* naturally cluster.  
Engine faults result from **complex nonlinear interactions**, not geometric groups.

---

## 5. Supervised Models

### **5.1 Logistic Regression (Baseline)**
- Simple, linear baseline.
- Useful for quick benchmarking.

### **5.2 Random Forest**
- Captures nonlinear interactions.  
- Highest feature importances:  
  - Coolant Temp  
  - Lube Oil Temp  
  - Fuel Pressure  
  - Coolant Pressure  

RPM shows low importance → speed alone does not predict faults.

### **5.3 Gradient Boosted Decision Trees (GBDT)**
- Stronger nonlinear modeling than RF  
- Good balance of speed and accuracy

### **5.4 XGBoost**
- Best performing model overall  
- Handles imbalanced data and nonlinear boundaries well  
- Suitable for sensor-based predictive maintenance 

---

## 6. Model Calibration

- Ensures predicted probabilities reflect real fault likelihood.  
- Important for maintenance scheduling & risk thresholds.

---

## 7. Interpretation & Business Insights

- Engine faults are driven by **thermal & pressure signals**, not RPM.  
- The model shows that **multi-sensor interactions** matter more than single variables.  
- Fault risk increases when oil/coolant temperatures and pressures deviate from normal ranges.  
- Unsupervise
