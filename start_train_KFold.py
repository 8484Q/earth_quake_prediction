import os
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

start_time = time.time()
print(os.listdir("./"))

"""
load from 90% gap data
"""
train_x1 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x1.csv',dtype=np.float32)
train_x2 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x2.csv',dtype=np.float32)
train_x3 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x3.csv',dtype=np.float32)
train_x4 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x4.csv',dtype=np.float32)
train_x5 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x5.csv',dtype=np.float32)
train_x6 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x6.csv',dtype=np.float32)
train_x7 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/x7.csv',dtype=np.float32)

y1 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y1.csv')
y2 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y2.csv')
y3 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y3.csv')
y4 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y4.csv')
y5 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y5.csv')
y6 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y6.csv')
y7 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/y7.csv')

test_x1 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x1.csv')
test_x2 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x2.csv')
test_x3 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x3.csv')
test_x4 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x4.csv')
test_x5 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x5.csv')
test_x6 = pd.read_csv('./dataset/90%_overlap_peak_0.5cor/test_x6.csv')

# train_x1 = train_x1.dropna()
# y1 = y1.dropna()


df_train = [train_x1, train_x2, train_x4, train_x5, train_x6, train_x7]
df_valid = [train_x3]
df_train_y = [y1, y2, y4, y5, y6, y7]
df_valid_y = [y3]
df_test = [test_x1, test_x2, test_x3, test_x4, test_x5, test_x6]

train_x_all = pd.concat(df_train)
valid_x_all = pd.concat(df_valid)
train_y_all = pd.concat(df_train_y)
valid_y_all = pd.concat(df_valid_y)
test_all = pd.concat(df_test)


train_x_all.drop(labels=['Unnamed: 0','seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
valid_x_all.drop(labels=['Unnamed: 0','seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
train_y_all.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
valid_y_all.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
test_all = test_all.rename(columns = {'Unnamed: 0':'seg_id'})
test_all = test_all.set_index('seg_id')

"""
scaler data
"""
scaler = StandardScaler()
scaler.fit(train_x_all)
scaled_train_X = pd.DataFrame(scaler.transform(train_x_all), columns=train_x_all.columns)
scaled_valid_X = pd.DataFrame(scaler.transform(valid_x_all), columns=valid_x_all.columns)
scaled_test_X = pd.DataFrame(scaler.transform(test_all), columns=test_all.columns)

print ('before pearsonr is: ',scaled_train_X.shape)

"""
use pearsonr method to drop un-related column
"""
pcol = []
pcor = []
pval = []
y = train_y_all['time_to_failure'].values

for col in scaled_train_X.columns:
    pcol.append(col)
    pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
    pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))

df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
df.sort_values(by=['cor', 'pval'], inplace=True)
df.dropna(inplace=True)
#df = df.loc[df['pval'] < 0.05]
#df[df['col'] == "FFT_Mag_10q15000"]
#df = df[(df.pval < 0.05) & (abs(df.cor)>=0.5)]
#df = df[(df.pval < 0.05)]
#df = df[(abs(df.cor)>=0.5)]
df = df[(abs(df.cor)>=0.55)]
#df = df[(abs(df.cor)>=0.6)]
#df = df[(abs(df.cor)>=0.65)]
#df = df[(abs(df.cor)>=0.7)]

drop_cols = []
for col in scaled_train_X.columns:
    if col not in df['col'].tolist():
        drop_cols.append(col)

scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
scaled_valid_X.drop(labels=drop_cols, axis=1, inplace=True)
scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

print ('after pearsonr is: ',scaled_train_X.shape)

"""
build model
"""
params = {'num_leaves': 31,
         'objective':'regression',
         'min_data_in_leaf': 200,
         'max_depth': 12,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         'max_bin':128,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": -1,
         "random_state": 21}

n_fold = 6
folds = KFold(n_splits=n_fold, shuffle=False)#, random_state=21)
train_columns = scaled_train_X.columns.values
best_iteration_all = list()
validation_mae_all = list()

oof = np.zeros(len(scaled_train_X))
valid_predictions = np.zeros(len(scaled_valid_X))
predictions = np.zeros(len(scaled_test_X))
feature_importance_df = pd.DataFrame()

print('train size is: ', len(scaled_train_X))
print('validation size is: ', len(scaled_valid_X))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y_all.values)):
    strLog = "fold {}".format(fold_)
    print(strLog)

    X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
    y_tr, y_val = train_y_all.iloc[trn_idx], train_y_all.iloc[val_idx]

    model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
    model.fit(X_tr,
              y_tr,
              eval_set=[(X_tr, y_tr), (X_val, y_val)],
              eval_metric='mae',
              verbose=1000,
              #num_boost_round=700,
              early_stopping_rounds=200)
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
    # feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # predictions
    predictions += model.predict(scaled_test_X, num_iteration=model.best_iteration_) / folds.n_splits
    valid_predictions += model.predict(scaled_valid_X, num_iteration=model.best_iteration_) / folds.n_splits

    validation_mae = mean_absolute_error(valid_predictions, valid_y_all)
    validation_mae_all.append(validation_mae)
    best_iteration_all.append(model.best_iteration_)

    print ('Test Dataset MAE: ', str(validation_mae))
print ('all best iteration: ', best_iteration_all)
print ('all mae: ', validation_mae_all)

cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:200].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('./output/lgbm_importances.png')

submission = pd.read_csv('./dataset/sample_submission.csv', index_col='seg_id')
submission.time_to_failure = predictions
submission.to_csv('./output/submission.csv',index=True)

end_time = time.time()
print('totol run ', str(end_time - start_time))