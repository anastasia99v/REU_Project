# -*- coding: utf-8 -*-

pip install catboost
pip install sklego

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline


from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklego.linear_model import LADRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14,3)
sns.set_style("darkgrid")

df_train = pd.read_csv('.../train.csv')
df_test = pd.read_csv('.../test.csv')

df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)


colms_num = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
              'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
              'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
              'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds']

for i in colms_num:
  plt.subplot(1,2,1)
  sns.histplot(data=df_train, x=df_train[i], bins=25, color='darkviolet')
  plt.title(i + ' - Train')
  plt.subplot(1,2,2)
  sns.histplot(data=df_test, x=df_test[i], bins=25, color='springgreen')
  plt.title(i + ' - Test')
  plt.savefig(f'./f{i}.png')
  plt.show();

def plot_box(df, x, y):
  plt.figure(figsize=(10,5))
  sns.boxplot(data=df, x=x, y=y)
  plt.savefig(f'./boxplot_{x}-{y}.png')
  return plt.show();

col_box = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
              'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
              'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
              'RainingDays', 'AverageRainingDays']

for i in col_box:
  plot_box(df_train, i, 'yield')

for i in col_box:
  display(df_train[i].value_counts().to_frame())
  print()
  print('-'*25)
  print()


def plot_scatter(df, x, y):
  plt.figure(figsize=(9,5))
  sns.scatterplot(data=df, x=x, y=y)
  plt.savefig(f'./scatterplot_{x}-{y}.png')
  return plt.show();

plt.figure(figsize=(10,4))
sns.histplot(data=df_train, x="yield", bins=50, color='darkviolet')
plt.title('Target (Yield)')
plt.savefig(f'./histplot_target.png')
plt.show();
print(df_train['yield'].describe())

f_col = ['fruitset','fruitmass','seeds']
for i in f_col:
  plot_scatter(df_train, i, 'yield')

sns.pairplot(df_train[['fruitset','fruitmass','seeds']], plot_kws = {'alpha': 0.25})
plt.savefig(f'./pairplot_num_cols.png')
plt.show();

def outlier_thresholds(data, col_name, q1=0.25, q3=0.75):
  quartile_1 = data[col_name].quantile(q1)
  quartile_3 = data[col_name].quantile(q3)
  c_range = quartile_3 - quartile_1
  up_limit = quartile_3 + 1.5 * c_range
  low_limit = quartile_1 - 1.5 * c_range
  return low_limit, up_limit


def check_out(data, col_name):
  low_limit, up_limit = outlier_thresholds(data, col_name)
  if data[(data[col_name] > up_limit) | (data[col_name] < low_limit)].any(axis=None):
    return True
  else:
    return False


def replace_by_thresholds(data, var):
  low_limit, up_limit = outlier_thresholds(data, var)
  data.loc[(data[var] < low_limit), var] = low_limit
  data.loc[(data[var] > up_limit), var] = up_limit

for i in df_train.columns:
  print(i, check_out(df_train, i))

outliers_colms = ['honeybee','bumbles','andrena','osmia','RainingDays',
                  'AverageRainingDays','fruitset','fruitmass','seeds','yield']
for i in outliers_colms:
  replace_by_thresholds(df_train, i)


for i in df_train.columns:
  print(i, check_out(df_train, i))

for i in col_box:
  display(df_train[i].value_counts().to_frame())
  print()
  print('-'*25)
  print()

plt.figure(figsize=(15,8))
sns.heatmap(df_train.corr(method="spearman").round(2), 
            mask=np.triu(np.ones_like(df_train.corr(method="spearman").round(2), dtype=bool)),
            cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
            vmin=-1, 
            vmax=+1,
            center=0,
            linecolor='yellow',
            square=True,
            linewidths=0.5,
            cbar=True,
            annot=True)
plt.title('Spearman Correlation Matrix')
plt.savefig(f'./Spearman_Corr.png')
plt.show();


plt.figure(figsize=(15,8))
sns.heatmap(df_train.corr(method="pearson").round(2), 
            mask=np.triu(np.ones_like(df_train.corr(method="pearson").round(2), dtype=bool)),
            cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
            vmin=-1, 
            vmax=+1,
            center=0,
            linecolor='yellow',
            square=True,
            linewidths=0.5,
            cbar=True,
            annot=True)
plt.title('Pearson Correlation Matrix')
plt.savefig(f'./Pearson_Corr.png')
plt.show();


total_density_train = df_train['honeybee'] + df_train['bumbles'] + df_train['andrena'] + df_train['osmia']
total_density_test = df_test['honeybee'] + df_test['bumbles'] + df_test['andrena'] + df_test['osmia']

df_train['TempRange'] = df_train['MaxOfUpperTRange'] - df_train['MinOfLowerTRange']
df_train['TotalBeeDens'] = df_train['honeybee'] + df_train['bumbles'] + df_train['andrena'] + df_train['osmia']
df_train['HoneybeeDominance'] = df_train['honeybee'] / total_density_train
df_train['BumblesBeeDominance'] = df_train['bumbles'] / total_density_train
df_train['AndrenaBeeDominance'] = df_train['andrena'] / total_density_train
df_train['OsmiaBeeDominance'] = df_train['osmia'] / total_density_train
df_train['RainIntensity'] = df_train['AverageRainingDays'] / df_train['RainingDays']

df_test['TempRange'] = df_test['MaxOfUpperTRange'] - df_test['MinOfLowerTRange']
df_test['TotalBeeDens'] = df_test['honeybee'] + df_test['bumbles'] + df_test['andrena'] + df_test['osmia']
df_test['HoneybeeDominance'] = df_test['honeybee'] / total_density_test
df_test['BumblesBeeDominance'] = df_test['bumbles'] / total_density_test
df_test['AndrenaBeeDominance'] = df_test['andrena'] / total_density_test
df_test['OsmiaBeeDominance'] = df_test['osmia'] / total_density_test
df_test['RainIntensity'] = df_test['AverageRainingDays'] / df_test['RainingDays']


train = df_train.copy()

x = add_constant(train.drop(['yield'], axis=1))
ml = OLS(train['yield'], x).fit()
p_vls = pd.DataFrame(ml.pvalues)
p_vls.reset_index(inplace=True)
p_vls.rename(columns={0: 'p_value', 'index': 'features'}, inplace=True)
p_vls.style.background_gradient(cmap='coolwarm')

"""# Building Model"""

colms_to_cat = ['clonesize', 'bumbles', 'osmia',
              'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange', 
              'MinOfLowerTRange', 'AverageOfLowerTRange',
              'RainingDays', 'AverageRainingDays', 'TempRange']

X = df_train.drop(['yield', 'honeybee', 'MaxOfLowerTRange', 'andrena','AndrenaBeeDominance'], axis=1)
y = df_train['yield']

for i in colms_to_cat:
  X[i] = X[i].astype('category')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.2, random_state=42)


def get_preprocessed(train, val, test):
  rob_sc = StandardScaler()
  train = rob_sc.fit_transform(train)
  val = rob_sc.transform(val)
  test = rob_sc.transform(test)
  return train, val, test

X_train, X_val, X_test = get_preprocessed(X_train, X_val, X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

predicton_val = linear_model.predict(X_val)
print(f'Accuracy MAE: {mean_absolute_error(y_val, predicton_val)}')


predicton_test = linear_model.predict(X_test)
print(f'Accuracy MAE: {mean_absolute_error(y_test, predicton_test)}')


"""# Fine-Tuning"""

model_params = {

    'catboost_regression': {
        'model': CatBoostRegressor(),
        'params': {
            'iterations': [1000],
            'learning_rate': [0.01, 0.1],
            'depth': [6, 16],
            'l2_leaf_reg': [0, 0.01, 0.1],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bylevel': [0.5, 0.7, 0.9],
            'random_strength': [0.05, 0.1, 0.2],
            'loss_function': ['MAE'],
            'verbose': [False],
            'grow_policy': ["Lossguide"]
        }
    },

    'lgbm_regression': {
        'model': lgb.LGBMRegressor(),
        'params': {
            'n_estimators': [1000],
            'learning_rate': [0.01, 0.1],
            'max_depth': [5, 6, 7],
            'min_child_weight': [5, 15, 20],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'reg_alpha': [0, 0.1, 0.2, 0.3, 1],
            'reg_lambda': [0, 0.1, 0.2, 0.3, 1],
            'num_leaves': [12, 20, 32, 64],
            'objective': ["mae"],
            'force_col_wise': [True],
        }
    },

    'xgbboost_regression': {
        'model': xgb.XGBRegressor(),
        'params': {
             'n_estimators': [1000],
             'learning_rate': [0.01, 0.1],
             'max_depth': [5, 6, 7],
             'min_child_weight': [5, 10, 20],
             'subsample': [0.5, 0.7, 0.9],
             'colsample_bytree': [0.5, 0.7, 0.9],
             'gamma': [0, 0.1, 0.2, 1],
             'reg_alpha': [0, 0.1, 0.2, 1],
             'reg_lambda': [0, 0.1, 0.2, 1],
             'objective': ['reg:squarederror', 'reg:pseudohubererror'],
             'eval_metric': ['mae'],
        }
    },
    'gradientboost_regression': {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': [250],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10, 16],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': [1.0, 'sqrt', 'log2'],
            'subsample': [0.5, 0.7, 0.9, 1.0],
            'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
            'alpha': [0.1, 0.5, 0.9]
        }
    }
}

scores = []
best_estimators = {}
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
for algo, mp, in model_params.items():
  gs = RandomizedSearchCV(mp['model'], param_distributions=mp['params'], cv=cv, return_train_score=False)
  gs.fit(X_train, y_train)
  scores.append({
      'model': algo,
      'best_score': gs.best_score_,
      'best_params': gs.best_params_
  })
  best_estimators[algo] = gs.best_estimator_

df_models = pd.DataFrame(scores, columns=['model','best_score','best_params'])


display(best_estimators['catboost_regression'].score(X_val, y_val))
display(best_estimators['lgbm_regression'].score(X_val, y_val))
display(best_estimators['xgbboost_regression'].score(X_val, y_val))
display(best_estimators['gradientboost_regression'].score(X_val, y_val))

best_reg1 = best_estimators['catboost_regression']
best_reg2 = best_estimators['lgbm_regression']
best_reg3 = best_estimators['xgbboost_regression']
best_reg4 = best_estimators['gradientboost_regression']

def show_best_pred(X_test, y_test, best_reg, name):
  best_reg_pred = best_reg.predict(X_test)
  print(f'MAE of {name}: {round(mean_absolute_error(y_test, best_reg_pred), 2)}')
  print(f'\nR2 Score of {name}: {r2_score(y_test, best_reg_pred)}')

show_best_pred(X_test, y_test, best_reg1, 'Catboost Regression')

show_best_pred(X_test, y_test, best_reg2, 'LightGBM Regression')

show_best_pred(X_test, y_test, best_reg3, 'XGBRegression')

show_best_pred(X_test, y_test, best_reg4, 'GradientBoostingRegressor')

def plot_important_features(ft_import, names, model):
  f_importance = np.array(ft_import)
  f_names = np.array(names)
  d = {'feature_importance': f_importance, 'feature_names': f_names}
  df = pd.DataFrame(d)
  df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
  plt.figure(figsize=[10,8])
  sns.barplot(x=df['feature_importance'], y=df['feature_names'], palette='RdBu')
  plt.ylabel('Feature Names')
  plt.xlabel('Feature Importance, %')
  plt.title(f'Feature Importance of {model} Model')
  plt.show()

plot_important_features(best_reg1.feature_importances_, X.columns, 'CatboostRegression')
plot_important_features(best_reg2.feature_importances_, X.columns, 'LightGBRegression')
plot_important_features(best_reg3.feature_importances_, X.columns, 'XGBRegression')
plot_important_features(best_reg4.feature_importances_, X.columns, 'GradientBoostingRegressor')

"""# Stacking"""

estimators = [
    ('p_catboost', best_reg1),
    ('p_lgbm', best_reg2),
]

stack_regressor = StackingRegressor(estimators=estimators, 
                                          final_estimator=LADRegression(alpha=0.001))

stack_regressor

stack_regressor_model = stack_regressor.fit(X_train, y_train)

print('Validation Data')
print()
show_best_pred(X_val, y_val, stack_regressor_model, 'Stacking Regressor')
print()
print('-'*20)
print('Test Data')
print()
show_best_pred(X_test, y_test, stack_regressor_model, 'Stacking Regressor')