##########################################################################################################
# END TO END MACHINE LEARNING PROJECT ON HITTERS DATASET
##########################################################################################################
# IMPORT NECESSARY LIBRARIES AND MODULES
# Making necessary adjustments for the representation of the dataset
##########################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from HAFTA_08.helpers.visuals import *
from HAFTA_08.helpers.eda import *
from HAFTA_08.helpers.data_prep import *

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

import warnings
warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

##########################################################################################################
# CONTENT
##########################################################################################################
# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modelling & Hyperparameter Optimization
# 4. Model Based Feature Selection
# 5. Automated Hyperparameter Optimization
# 6. Ensemble Learning
# 7. Prediction
# 8. References
##########################################################################################################
# READ THE DATASET
##########################################################################################################
df_ = pd.read_csv("HAFTA_08/hitters.csv")
df = df_.copy()
##########################################################################################################
# 1. EXPLORATORY DATA ANALYSIS
##########################################################################################################
check_df(df)    # We have 59 NA values in "Salary"

# Observe NaN values
df[df.isnull().any(axis=1)].head()

missing_values_table(df)
'''
        n_miss  ratio
Salary      59 18.320
'''
# Ratio of missing values of "Salary" is 18.32 which is high.

df.describe().T

# Distplot of target variable
sns.distplot(df.Salary)

# Grab numerical, categorical, etc columns
cat_cols, num_cols, cat_but_car, num_but_cat, binary_cols = grab_col_names(df)

# Plot all numerical columns:
plot_numerical_col(df, num_cols)

# Correlation matrix
mask = np.triu(np.ones_like(df.corr().round(2)))
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True, mask=mask)
plt.show()

##########################################################################################################
# 2. DATA PRE-PROCESSING & FEATURE ENGINEERING
##########################################################################################################
check_df(df, head=5, tail=5, quan=False)   # We have 59 NaN values in "Salary". Their ratio was > 18

# I will extract NaN values of "Salary"
df2 = df[~(df.isnull().any(axis=1))]

df2.shape   # (263, 20)

# Implement grab_col_names fo df2
cat_cols2, num_cols2, cat_but_car2, num_but_cat2, binary_cols2 = grab_col_names(df2)

# SUMMARY OF CATEGORICAL COLUMNS
for col in cat_cols2:
    cat_summary(df2, col, plot=False)

df2.describe().T

# OUTLIER ANALYSIS
for col in num_cols:
    print(col, check_outlier(df2, col))

# NOTE: I will continue with the dataset including outliers.
# Keep in mind min=0 values while feature engineering
min_zero_cols = [col for col in df2 if df2[col].min() == 0]
min_zero_cols   #  ['HmRun', 'Runs', 'RBI', 'Walks', 'CHmRun', 'PutOuts', 'Assists', 'Errors']

# YEARS
sns.displot(df2.Years)
df2["New_Years_Cat"] = pd.cut(x=df2['Years'],
                              bins=[0, 3, 6, 10, 15, 19, 24],
                              labels=["(0-3]", "(3, 6]", "(6, 10]", "(10, 15]", "(15, 19]", "(19, 24]"])

df2.groupby(['League','Division', 'New_Years_Cat']).agg({'Salary':'mean'})

df2.groupby(['New_Years_Cat']).agg({'Salary':'mean'})

df2.groupby(['Division']).agg({'Salary':'mean'})

df2.groupby(['League']).agg({'Salary':'mean'})

df2.groupby(['NewLeague']).agg({'Salary':'mean'})

df2.groupby(['Errors','Assists', 'New_Years_Cat']).agg({'Salary':'mean'})

# CREATING NEW FEATURES
# BATTING AVERAGE
# 1986-1987 season
df2["New_Batting_Avg"] = df2["Hits"]/df2["AtBat"]
# Whole career
df2["New_C_Batting_Avg"] = df2["CHits"]/df2["CAtBat"]

# PERCENTAGE OF HOME RUNS
# 1986-1987 Season
df2.loc[df2["Runs"] != 0, "New_Percentage_HmRun"] = df2["HmRun"] / df2["Runs"] # divide by 0 correction
df2.loc[df2["Runs"] == 0, "New_Percentage_HmRun"] = 0
# Whole career
df2["New_C_Percentage_HmRun"] = df2["CHmRun"]/df2["CRuns"]

# AT BATS PER HOME RUN
# 1986-1987 Season
df2.loc[df2["HmRun"] != 0, "New_AtBat_Per_HmRun"] = df2["AtBat"] / df2["HmRun"] # divide by 0 correction
df2.loc[df2["HmRun"] == 0, "New_AtBat_Per_HmRun"] = 0
# Whole career
df2.loc[df2["CHmRun"] != 0, "New_C_AtBat_Per_HmRun"] = df2["CAtBat"] / df2["CHmRun"] # divide by 0 correction
df2.loc[df2["CHmRun"] == 0, "New_C_AtBat_Per_HmRun"] = 0

# HOME RUN PER HITS
# 1986-1987 Season
df2["New_HmRun_Per_Hits"] = df2["HmRun"]/df2["Hits"]
# Whole career
df2["New_C_HmRun_Per_Hits"] = df2["CHmRun"]/df2["CHits"]

# FIELDING PERCENTAGE (FPCT)
# FPCT = (Putouts + Assists) / (Total Chances)
# Total Chances = Putouts + Assists + Errors
# 1986-1987 Season
df2.loc[(df2["PutOuts"]+df2["Assists"]+df2["Errors"]) != 0, "New_FPCT"] = (df2["PutOuts"]+df2["Assists"]) / (df2["PutOuts"]+df2["Assists"]+df2["Errors"]) # divide by 0 correction
df2.loc[(df2["PutOuts"]+df2["Assists"]+df2["Errors"]) == 0, "New_FPCT"] = 0

# PERFORMANCES OF 1986-1987 SEASON
df2["New_Perf_Hits"] = df2["Hits"]/df2["CHits"]
df2["New_Perf_AtBat"] = df2["AtBat"]/df2["CAtBat"]
df2.loc[df2["CHmRun"] != 0, "New_Perf_HmRun"] = df2["HmRun"] / df2["CHmRun"] # divide by 0 correction
df2.loc[df2["CHmRun"] == 0, "New_Perf_HmRun"] = 0
df2["New_Perf_Runs"] = df2["Runs"]/df2["CRuns"]
df2["New_Perf_RBI"] = df2["RBI"]/df2["CRBI"]
df2["New_Perf_Walks"] = df2["Walks"]/df2["CWalks"]

# AVERAGE STATISTICS OF 1986-1987 SEASON
df2["New_Avg_Hits"] = df2["CHits"]/df2["Years"]
df2["New_Avg_AtBat"] = df2["CAtBat"]/df2["Years"]
df2["New_Avg_HmRun"] = df2["CHmRun"]/df2["Years"]
df2["New_Avg_Runs"] = df2["CRuns"]/df2["Years"]
df2["New_Avg_RBI"] = df2["CRBI"]/df2["Years"]
df2["New_Avg_Walks"] = df2["CWalks"]/df2["Years"]

# HOME RUN ANALYSIS
sns.distplot(df2.HmRun)
df2["New_HmRun_Cat"] = pd.cut(x=df2['HmRun'],
                              bins=[0, 10, 20, 30, 40],
                              labels=["(0-10]", "(10-20]", "(20-30]", "(30-40]"])

df2.groupby('New_HmRun_Cat').agg({'Salary':'mean'})

# Let's see our dataset after creating new features
df3 = df2.copy()
df3.describe().T
df3.shape   # (263, 43)

df3.isnull().sum()  # There are 10 NaN values at "New_HmRun_Cat"

# Observe NaN values
df3[df3.isnull().any(axis=1)].head(10)

# These NaN values are resulting of HmRun = 0, so we can fill this value with zero ("0-10" for this categorical feature).
df3["New_HmRun_Cat"].fillna("(0-10]", inplace=True)

# As we have a category type of variables ("New_Years_Cat" and "New_HmRun_Cat"), we need to find cat_cols, etc. again:
cat_cols3 = [col for col in df3.columns if df3[col].dtypes not in ["int", "int64","float64"]]

cat_but_car3 = [col for col in df3.columns if df3[col].nunique() > 20 and df3[col].dtypes == "O"]    #[]

# num_cols
num_cols3 = [col for col in df3.columns if df3[col].dtypes in ["int", "int64","float64"]]
len(num_cols3)      # 38

# binary_cols
binary_cols3 = [col for col in df3.columns if df3[col].dtype not in [int, float] and
                df3[col].nunique() == 2]

print(f"Observations: {df3.shape[0]}")
print(f"Variables: {df3.shape[1]}")
print(f'cat_cols: {len(cat_cols3)}')
print(f'num_cols: {len(num_cols3)}')
print(f'cat_but_car: {len(cat_but_car3)}')
print(f'binary_cols: {len(binary_cols3)}')

# We will not use "Salary" column while encoding
num_cols3 = [col for col in num_cols3 if "Salary" not in col]

# LABEL ENCODING
# Label encoding for binary columns and categorical columns:
for col in cat_cols3:
    df3 = label_encoder(df3, col)

# ROBUST SCALER
rs = RobustScaler()
df3[num_cols3] = rs.fit_transform(df3[num_cols3])

df3.describe().T

# RARE ANALYSIS
rare_analyser(df3, "Salary", cat_cols3)

##########################################################################################################
# 3. MODELlING & HYPERPARAMETER OPTIMISATION
##########################################################################################################
# Selecting dependent and independent variables
y = df3[["Salary"]]
X = df3.drop("Salary", axis=1)

######################################################
# Base Models
######################################################
# First check base models and select among them
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
'''
RMSE: 308.9529 (LR) 
RMSE: 304.5512 (Ridge) 
RMSE: 303.8655 (Lasso) 
RMSE: 307.6324 (ElasticNet) 
RMSE: 295.6954 (KNN) 
RMSE: 392.2768 (CART) 
RMSE: 261.6829 (RF) 
RMSE: 444.3529 (SVR) 
RMSE: 266.4184 (GBM) 
RMSE: 288.2598 (XGBoost) 
RMSE: 280.2861 (LightGBM) 

I choose RF, GBM, LightGBM and XGBoost as they have less RMSE than other models. 
'''

################################################
# Random Forests
################################################
rf_model = RandomForestRegressor(random_state=1)

rmse = np.mean(np.sqrt(-cross_val_score(rf_model, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)} ")       # RMSE: 259.267

rf_params = {"max_depth": [2, 5, 8, 15, None],
             "max_features": [2, 3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [200, 500, 700, 1000]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=3, n_jobs=-1, verbose=False).fit(X, y.values.ravel())

rf_final = rf_model.set_params(**rf_best_grid.best_params_,
                               random_state=1).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE (After): {round(rmse, 4)} ")       # RMSE (After): 259.8212
print(f"RF best params: {rf_best_grid.best_params_}", end="\n\n")
# RF best params: {'max_depth': None, 'max_features': 3, 'min_samples_split': 2, 'n_estimators': 1000}

################################################
# GBM Model
################################################
gbm_model = GradientBoostingRegressor(random_state=1)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_model, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)} ")       # RMSE: 268.4549

gbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
              "max_depth": [2, 3, 4, 5, 8],
              "n_estimators": [100, 200, 500, 700],
              "subsample": [1, 0.3, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=3, n_jobs=-1, verbose=False).fit(X, y.values.ravel())

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_,
                               random_state=1).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE (After): {round(rmse, 4)} ")       # RMSE (After): 259.7271
print(f"GBM best params: {gbm_best_grid.best_params_}", end="\n\n")
# GBM best params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.5}

################################################
# LightGBM Model
################################################
lgbm_model = LGBMRegressor(random_state=1)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)} ")       # RMSE: 280.2861

lightgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                   "n_estimators": [200, 300, 400, 500, 700, 1000],
                   "colsample_bytree": [0.2, 0.3, 0.4, 0.5, 0.7]}

lgbm_best_grid = GridSearchCV(lgbm_model, lightgbm_params, cv=3, n_jobs=-1, verbose=False).fit(X, y.values.ravel())

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_,
                               random_state=1).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE (After): {round(rmse, 4)} ")       # RMSE (After): 281.5258
print(f"LightGBM best params: {lgbm_best_grid.best_params_}", end="\n\n")
# LightGBM best params: {'colsample_bytree': 0.3, 'learning_rate': 0.2, 'n_estimators': 200}

################################################
# XGBoost Model
################################################
xgboost_model = XGBRegressor(random_state=1)

rmse = np.mean(np.sqrt(-cross_val_score(xgboost_model, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)} ")       # RMSE: 288.2598

xgboost_params = {"learning_rate": [0.1, 0.01, 0.02, 0.03, 0.05],
                  "max_depth": [2, 3, 5, 8, 12],
                  "n_estimators": [300, 500, 700, 1000, 1500],
                  "colsample_bytree": [0.2, 0.3, 0.5, 0.8, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=3, n_jobs=-1, verbose=False).fit(X, y.values.ravel())

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_,
                               random_state=1).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(xgboost_final, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE (After): {round(rmse, 4)} ")       # RMSE (After): 255.7155
print(f"XGBoost best params: {xgboost_best_grid.best_params_}", end="\n\n")
# XGBoost best params: {'colsample_bytree': 0.3, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1500}

##########################################################################################################
# 4. MODEL BASED FEATURE SELECTION
##########################################################################################################
# FEATURE IMPORTANCE
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.title(f"Features for {type(model).__name__}")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X, num=20)
plot_importance(gbm_final, X, num=20)
plot_importance(lgbm_final, X, num=20)
plot_importance(xgboost_final, X, num=20)

# MODEL BASED FEATURE SELECTION
# Get the importances of each final model
feature_imp_rf = pd.DataFrame({'Value': rf_final.feature_importances_, 'Feature': X.columns})
feature_imp_gbm = pd.DataFrame({'Value': gbm_final.feature_importances_, 'Feature': X.columns})
feature_imp_lgbm = pd.DataFrame({'Value': lgbm_final.feature_importances_, 'Feature': X.columns})
feature_imp_xgboost = pd.DataFrame({'Value': xgboost_final.feature_importances_, 'Feature': X.columns})

# Gather them in one dataframe
feature_df = pd.merge(feature_imp_rf, feature_imp_gbm, on="Feature")
feature_df2 = pd.merge(feature_imp_lgbm, feature_imp_xgboost, on="Feature")
feature_imp_df = pd.merge(feature_df,feature_df2, on="Feature")

feature_imp_df = feature_imp_df.rename(columns = {'Value_x_x': 'RF_score',
                                                  'Value_y_x': 'GBM_score',
                                                  'Value_x_y': 'LGBM_score',
                                                  'Value_y_y': 'XGBoost_score'})
feature_imp_df = feature_imp_df[['Feature', 'RF_score', 'GBM_score', 'LGBM_score', 'XGBoost_score']]

# Standardization with MinMaxScaler
num_cols_imp = [col for col in feature_imp_df.columns if feature_imp_df[col].dtypes in ['int32', 'float32', 'float64']]
scaler = MinMaxScaler()
feature_imp_df[num_cols_imp] = scaler.fit_transform(feature_imp_df[num_cols_imp])
feature_imp_df.head()

# Calculate and sort weighted scores
feature_imp_df["Weighted_Score"] = (feature_imp_df["RF_score"]+\
                                   feature_imp_df["GBM_score"]+\
                                   feature_imp_df["LGBM_score"]+\
                                   feature_imp_df["XGBoost_score"])/4

dff = feature_imp_df[['Feature', 'Weighted_Score']]
dff.sort_values(by= 'Weighted_Score', ascending=False).head(10)

'''
 Feature  Weighted_Score
8               CHits           0.831
11               CRBI           0.760
39        New_Avg_RBI           0.745
10              CRuns           0.606
5               Walks           0.498
7              CAtBat           0.473
38       New_Avg_Runs           0.455
21  New_C_Batting_Avg           0.449
12             CWalks           0.446
9              CHmRun           0.428
'''

######################################################
# MODEL ACCORDING TO NEW FEATURE SELECTION
######################################################
# I can drop least important features and build models again and check the RMSE scores.
# I will drop features whose scores < 0.3
to_drop = dff.loc[dff[('Weighted_Score'] < 0.3]
features_to_drop = to_drop['Feature']

# Before dropping above features, I will drop high correlated features first.
# Remembering the correlation matrix, we can drop CRuns and CAtBat.
# Because correlation between CAtBat and CRuns = 0.98
# Correlation between CHits and CRuns = 0.98
# Correlation between CHits and CAtBat = 1

df4 = df3.drop('CRuns', axis=1)
df4 = df4.drop('CAtBat', axis=1)

# Modelling again
y2 = df4[["Salary"]]
X2 = df4.drop("Salary", axis=1)

models = [('RF', RandomForestRegressor(random_state=1)),
          ('GBM', GradientBoostingRegressor(random_state=1)),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor(random_state=1))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X2, y2, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

'''
RMSE: 260.9191 (RF) 
RMSE: 255.3273 (GBM) 
RMSE: 281.7351 (XGBoost) 
RMSE: 279.8737 (LightGBM) 
'''
# I will drop the least important features found above.
df5= df4.drop(features_to_drop, axis=1)

y3 = df5[["Salary"]]
X3 = df5.drop("Salary", axis=1)

# Check RMSE values:
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X3, y3, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
'''
RMSE: 254.3354 (RF) 
RMSE: 255.2996 (GBM) 
RMSE: 269.6594 (XGBoost) 
RMSE: 279.7285 (LightGBM) 
'''

##########################################################################################################
# 5. AUTOMATED HYPERPARAMETER OPTIMISATION
##########################################################################################################
rf_params = {"max_depth": [2, 5, 8, 15, None],
             "max_features": [2, 3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [200, 500, 700, 1000]}

gbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
              "max_depth": [2, 3, 4, 5, 8],
              "n_estimators": [100, 200, 500, 700],
              "subsample": [1, 0.3, 0.5, 0.7]}

lightgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                   "n_estimators": [200, 300, 400, 500, 700, 1000],
                   "colsample_bytree": [0.2, 0.3, 0.4, 0.5, 0.7]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.02, 0.03, 0.05],
                  "max_depth": [2, 3, 5, 8, 12],
                  "n_estimators": [300, 500, 700, 1000, 1500],
                  "colsample_bytree": [0.2, 0.3, 0.5, 0.8, 1]}

regressors = [("RF", RandomForestRegressor(random_state=1), rf_params),
              ("GBM", GradientBoostingRegressor(random_state=1), gbm_params),
              ('LightGBM', LGBMRegressor(random_state=1), lightgbm_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X3, y3, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X3, y3)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X3, y3, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

'''
########## RF ##########
RMSE: 254.3354 (RF)
RMSE (After): 260.5518 (RF) 
RF best params: {'max_depth': None, 'max_features': 2, 'min_samples_split': 2, 'n_estimators': 200}
########## GBM ##########
RMSE: 255.2996 (GBM) 
RMSE (After): 255.2996 (GBM) 
GBM best params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1}
########## LightGBM ##########
RMSE: 279.7285 (LightGBM) 
RMSE (After): 272.0198 (LightGBM) 
LightGBM best params: {'colsample_bytree': 0.2, 'learning_rate': 0.03, 'n_estimators': 1000}
########## XGBoost ##########
RMSE: 269.6594 (XGBoost) 
RMSE (After): 259.7992 (XGBoost) 
XGBoost best params: {'colsample_bytree': 0.3, 'learning_rate': 0.02, 'max_depth': 3, 'n_estimators': 1000}
'''

##########################################################################################################
# 6. ENSEMBLE LEARNING
##########################################################################################################
voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('GBM', best_models["GBM"]),
                                         ('LightGBM', best_models["LightGBM"]),
                                         ('XGBoost', best_models["XGBoost"])])

voting_reg.fit(X3, y3)

rmse = np.mean(np.sqrt(-cross_val_score(voting_reg, X3, y3, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)} ({name}) ")
# RMSE: 253.3332 (XGBoost)

##########################################################################################################
# 7. PREDICTION
##########################################################################################################
random_user = X.sample(1, random_state=42)
voting_reg.predict(random_user) # array([788.22318504])

# Let's see the original salary
df.iloc[148]    # 850.000

##########################################################################################################
# 8. REFERENCES
##########################################################################################################
# Data Science and Machine Learning Bootcamp, 2021, https://www.veribilimiokulu.com/
# https://www.kaggle.com/mathchi/four-different-models-for-hitters
# https://www.mlb.com/glossary