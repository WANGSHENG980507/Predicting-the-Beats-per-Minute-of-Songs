# 1. Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',None)

df1 = pd.read_csv(r"C:\Users\Alienware\Desktop\exercise\Predicting Road Accident Risk\train.csv")
print('shape of data:',df1.shape)
print(df1.head())

# 2. Basic Info & Data Quality
print(df1.info())
print(df1.describe(include='all'))

# Missing value
missing = df1.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    plt.figure(figsize=(10,6))
    missing.plot(kind='bar')
    plt.title('Missing Values')
    plt.show()
else:
    print('No null value exist')

print('Duplicated rows:',df1.duplicated().sum())
print("\n--- Descriptive Statistics ---")
print(df1.describe().T)
# One-hot
from sklearn.preprocessing import OneHotEncoder
df = pd.get_dummies(df1,drop_first=True) 
print(df.info())

# 3. Target Variable Analysis (Target: accident_risk)
target = 'accident_risk'
plt.figure(figsize=(10,6))
sns.histplot(df[target],kde=True,bins=30)
plt.title(f'distribution of {target}')
plt.show()

# 4. Feature Distributions
num_feature = df.columns.tolist()

df[num_feature].hist(bins=30,figsize=(15,12),layout=(4,4))
plt.suptitle('Feature distribution')
plt.show()

# 5. Correlation Analysis
plt.figure(figsize=(12,10))
sns.heatmap(df[num_feature].corr(),annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 6. Linear regression analyse
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
x = df.drop(columns='accident_risk')
y = df['accident_risk']
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
print(x.info())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
df_train = pd.concat([x_train,y_train],axis=1)
formula = "accident_risk ~ " + " + ".join(x_train.columns)
LR_model = smf.ols(formula=formula,data = df_train).fit()
print(LR_model.summary())
y_predicted = LR_model.predict(x_test)
print("R^2:",r2_score(y_test,y_predicted),"MSE:",mean_squared_error(y_test,y_predicted))

# cross_val_score analyze
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lm, x, y, cv=cv, scoring="r2")
print("Cross_val_score mean R2-value:", np.mean(cv_scores))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)  
y_pred_rf = rf.predict(x_test)  

print("RF R²:", r2_score(y_test, y_pred_rf))
print("RF MSE:", mean_squared_error(y_test, y_pred_rf))


y_pred_lm = lm.fit(x_train, y_train).predict(x_test)
residuals = y_test - y_pred_lm


plt.figure(figsize=(10,5))
sns.histplot(residuals, kde=True, bins=30)
plt.title("residuals distribution")
plt.show()


df_test = pd.read_csv(r"C:\Users\Alienware\Desktop\exercise\Predicting Road Accident Risk\test.csv")
df_test = pd.get_dummies(df_test,drop_first=True)
df_test[target] = rf.predict(df_test)
df_test1 = df_test['id']
df_test2 = df_test[target]
df_test = pd.concat([df_test1,df_test2],axis=1)
df_test.to_csv(r"C:\Users\Alienware\Desktop\exercise\Predicting Road Accident Risk\sample_submission.csv")
