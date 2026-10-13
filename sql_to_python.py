import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Varun@123",
    database="mydatabase"
)

print("Connected to MySql successfully!")
conn.close()


from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


engine = create_engine("mysql+mysqlconnector://aqi_user:aqi123@localhost/mydatabase")

query= "SELECT * FROM air_quality"
df =pd.read_sql(query, engine)

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df['city'].value_counts())

df['date'] =pd.to_datetime(df['date'])
df= df.sort_values(["city","date"])

#global style settings
sns.set_style("whitegrid")
sns.set_context("talk")   
plt.rcParams["figure.figsize"] =(10,6)
plt.rcParams["axes.titlesize"]= 16
plt.rcParams["axes.labelsize"]= 12

#aqi distribution
plt.figure()
sns.histplot(df['aqi'],bins=30,kde=True, color="teal")
plt.title("Distribution of AQI Levels",fontweight="bold")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()

#aqi outliers
plt.figure()
sns.boxplot(x=df['aqi'],color="orange")
plt.title("Outlier Detection in AQI",fontweight="bold")
plt.show()

#pollutants comparison
pollutants = ['pm25','pm10','no2','so2','co','o3']
df[pollutants].mean().plot(kind='bar', colormap="viridis",edgecolor="black")
plt.title("Average Pollutant Levels", fontweight="bold")
plt.xticks(rotation=45)
plt.show()

#city-wise aqi comparison
plt.figure()
sns.boxplot(x='city',  y='aqi',data=df,hue='city',  palette="Set2",legend=False)
plt.title("City-wise AQI Comparison",fontweight="bold")
plt.xticks(rotation=47)
plt.show()

#AQI trend over time
plt.figure()
palette = sns.color_palette("tab10")
for i, city in enumerate(df['city'].unique()):
    city_data =df[df['city']== city]
    plt.plot(city_data['date'],city_data['aqi'],
             label=city,
             color=palette[i],
             linewidth=2)
plt.title("AQI Trend Over Time",fontweight="bold")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.show()

#correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Between AQI & Pollutants", fontweight="bold")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

df["aqi_lag1"]= df.groupby("city")["aqi"].shift(1)
df["aqi_lag2"] =df.groupby("city")["aqi"].shift(2)
df["aqi_lag3"] =df.groupby("city")["aqi"].shift(3)

#rolling mean 
df["aqi_rolling_mean_7"] = (
    df.groupby("city")["aqi"]
      .shift(1)
      .rolling(7)
      .mean()
)

df.dropna(inplace=True)


df= pd.get_dummies(df, columns=["city"], drop_first=True)
X =df.drop(["aqi", "date"], axis=1)
y =df["aqi"]

features= [
    "pm25", "pm10","no2", "so2", "co","o3",
    "temperature","humidity",
    "aqi_lag1","aqi_lag2","aqi_lag3",
    "aqi_rolling_mean_7"
]
print(df.columns)

split_index= int(len(df)*0.8)

X_train= X.iloc[:split_index]
X_test= X.iloc[split_index:]

y_train= y.iloc[:split_index]
y_test= y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled =scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=200),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SVR": SVR()
}

results = []

for name,model in models.items():
    
    # Using scaled data for SVR
    if name =="SVR":
        model.fit(X_train_scaled, y_train)
        preds= model.predict(X_test_scaled)
    else:
        model.fit(X_train,y_train)
        preds =model.predict(X_test)
    
    mae= mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test,preds)
    
    results.append([name,mae,r2])

results_df= pd.DataFrame(results, columns=["Model","MAE","R2"])
print("\nMODEL COMPARISON")
print(results_df.sort_values("R2",ascending=False))


from sklearn.model_selection import GridSearchCV
print("\n⚙️ HYPERPARAMETER TUNING")

param_grid = {
    "n_estimators": [100,200],
    "max_depth": [None,10,20],
    "min_samples_split": [2,5]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:",grid.best_params_)
print("Best CV Score:",grid.best_score_)

best_model = grid.best_estimator_


from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import numpy as np

print("\n📊 TIME SERIES SPLIT (Random Forest)")

tscv = TimeSeriesSplit(n_splits=5)

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

r2_scores = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    
    X_train_fold, X_test_fold =X.iloc[train_index],X.iloc[test_index]
    y_train_fold, y_test_fold= y.iloc[train_index], y.iloc[test_index]
    
    rf_model.fit(X_train_fold,y_train_fold)
    preds= rf_model.predict(X_test_fold)
    
    r2= r2_score(y_test_fold, preds)
    r2_scores.append(r2)
    
    print(f"Fold {fold+1} R2:", r2)

print("Average TimeSeries R2:",np.mean(r2_scores))

best_model.fit(X_train,y_train)

#Feature importance
importances= best_model.feature_importances_

feature_importance_df= pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance",ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance",y="Feature",data=feature_importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()
print(feature_importance_df)

def health_risk(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

print("\n🌍 NEXT DAY AQI FORECAST (City-wise)")
city_conditions = {
    "Delhi": (df["city_Jaipur"] == 0) & (df["city_Lucknow"] == 0),
    "Jaipur": df["city_Jaipur"] == 1,
    "Lucknow": df["city_Lucknow"] == 1
}

for city, condition in city_conditions.items():

    city_data =df[condition]
    last= city_data.iloc[-1]

    # Very Important: Using same feature columns as training
    next_input= pd.DataFrame([last[X_train.columns].values],
                              columns=X_train.columns)

    next_prediction =best_model.predict(next_input)[0]

    print(f"{city} → AQI: {round(next_prediction,2)} | Risk Level: {health_risk(next_prediction)}")
    

import joblib
joblib.dump(best_model,"best_aqi_model.pkl")
joblib.dump(X_train.columns,"feature_columns.pkl")