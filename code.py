import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("engine_data.csv")

#structure check 
df.shape          # (19535, 7)
df.head()
df.info()
#df.describe()
df["Engine Condition"].value_counts(normalize=True) #imbalanced


#data quality check 
df.isna().sum
df.duplicated().sum
#No data cleaning needed

#range sanity per feature BUT idk how things should looks like so just leave it here
#df[["Engine rpm", "Lub oil pressure", "Fuel pressure","Coolant pressure", "lub oil temp", "Coolant temp"]].describe().T

#--------------
#EDA
#--------------
#Distribution by engine condition
target = "Engine Condition"
features = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
            "Coolant pressure", "lub oil temp", "Coolant temp"]

for col in features:
    plt.figure()
    sns.kdeplot(data=df, x=col, hue=target, common_norm=False)
    plt.title(f"{col} distribution by Engine Condition")
    plt.show()

#ATTENTION: Engine rpm 

#correlation 
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.show()

#This dataset is MAYBE not clusterable.

#--------------
#Data Prepare
#--------------
#train/test splits 
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Engine Condition"])
y = df["Engine Condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#scaling 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--------------
#PCA 
#--------------
from sklearn.decomposition import PCA

features = ["Engine rpm", "Lub oil pressure", "Fuel pressure", 
            "Coolant pressure", "lub oil temp", "Coolant temp"]
Z = df[features]
scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

df['PC1'] = Z_pca[:,0]
df['PC2'] = Z_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Engine Condition",
    data=df,
    palette="coolwarm",
    s=40
)
plt.title("PCA of Engine Sensor Data")
plt.show()

pca_full = PCA().fit(Z_scaled)

plt.plot(pca_full.explained_variance_ratio_, marker='o')
plt.xlabel("Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.show()

#SVM decision boundary on PC1-PC2

#svm = SVC(kernel='rbf', gamma='scale', probability=True)
#svm.fit(df[['PC1', 'PC2']], df['Engine Condition'])

#x_min, x_max = df['PC1'].min() - 1, df['PC1'].max() + 1
#y_min, y_max = df['PC2'].min() - 1, df['PC2'].max() + 1
#xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
#                     np.linspace(y_min, y_max, 300))

#Z = svm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#Z = Z.reshape(xx.shape)

#plt.figure(figsize=(8,6))
#plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.4)
#sns.scatterplot(x='PC1', y='PC2', hue='Engine Condition', data=df, s=30)
#plt.title("Nonlinear Decision Boundary (SVM on PCA)")
#plt.show()

#Proof of no group structure:Silhouette Score Plot for KMeans
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

#--------------
#Baseline - Logistic Regression 
#--------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

#--------------
#Baseline - Random Forest
#--------------
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

plt.figure(figsize=(8, 5))
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

#--------------
#plain GBDT
#--------------
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(
    loss='log_loss',     
    learning_rate=0.05,   
    n_estimators=300,     
    max_depth=3,         
    subsample=0.8,        
    random_state=42
)

gbdt.fit(X_train, y_train)

y_pred_gbdt = gbdt.predict(X_test)
y_proba_gbdt = gbdt.predict_proba(X_test)[:, 1]

print("GBDT AUC:", roc_auc_score(y_test, y_proba_gbdt))
print(classification_report(y_test, y_pred_gbdt))

cm = confusion_matrix(y_test, y_pred_gbdt)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("GBDT - Confusion Matrix")
plt.show()

importances = gbdt.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importances)
plt.figure(figsize=(8,5))
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("GBDT Feature Importances")
plt.show()

#--------------
#XGBoost
#--------------
from xgboost import XGBClassifier
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)

xgb = XGBClassifier(
    n_estimators=400,          
    max_depth=4,              
    learning_rate=0.05,       
    subsample=0.8,            
    colsample_bytree=0.8,     
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,  
    random_state=42,
    n_jobs=-1,                
    tree_method="hist"        
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

print("XGBoost AUC:", roc_auc_score(y_test, y_proba_xgb))
print(classification_report(y_test, y_pred_xgb))

#--------------
#Calibration Models based on xgb
#--------------
from sklearn.calibration import calibration_curve

y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

prob_true, prob_pred = calibration_curve(y_test, y_proba_xgb, n_bins=10, strategy="quantile")
plt.plot(prob_pred, prob_true, marker='o', label='XGBoost (uncalibrated)')
plt.plot([0,1],[0,1], '--', label='Perfect calibration')
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Curve - XGBoost (before calibration)")
plt.legend()
plt.show()

#--------------
# Calibration Models based on xgb - REDESIGNED
#--------------
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Plot calibration BEFORE calibration
y_proba_xgb_uncalibrated = xgb.predict_proba(X_test)[:, 1]
prob_true_uncal, prob_pred_uncal = calibration_curve(
    y_test, y_proba_xgb_uncalibrated, n_bins=10, strategy="quantile"
)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='XGBoost (uncalibrated)', linewidth=2)
plt.plot([0, 1], [0, 1], '--', label='Perfect calibration', linewidth=2, color='gray')
plt.xlabel("Mean Predicted Probability", fontsize=12)
plt.ylabel("Fraction of Positives", fontsize=12)
plt.title("Calibration Curve - XGBoost (Before Calibration)", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#unfinished:carlibration 
