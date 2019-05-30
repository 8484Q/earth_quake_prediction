#https://www.kaggle.com/tocha4/lanl-master-s-approach

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from catboost import Pool, CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

print (os.listdir("./"))

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

df_train = [train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_x7]
df_train_y = [y1, y2, y3, y4, y5, y6, y7]
df_test = [test_x1, test_x2, test_x3, test_x4, test_x5, test_x6]

train_x_all = pd.concat(df_train)
train_y_all = pd.concat(df_train_y)
test_all = pd.concat(df_test)

train_x_all.drop(labels=['Unnamed: 0','seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
train_y_all.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
test_all = test_all.rename(columns = {'Unnamed: 0':'seg_id'})
test_all = test_all.set_index('seg_id')

"""
scaler data
"""
scaler = StandardScaler()
scaler.fit(train_x_all)
scaled_train_X = pd.DataFrame(scaler.transform(train_x_all), columns=train_x_all.columns)
scaled_test_X = pd.DataFrame(scaler.transform(test_all), columns=test_all.columns)

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
df = df[(abs(df.cor)>=0.5)]
#df = df[(abs(df.cor)>=0.55)]
#df = df[(abs(df.cor)>=0.6)]
#df = df[(abs(df.cor)>=0.65)]
#df = df[(abs(df.cor)>=0.7)]

drop_cols = []
for col in scaled_train_X.columns:
    if col not in df['col'].tolist():
        drop_cols.append(col)

scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

"""
build model
"""
X_train, X_test, y_train, y_test = train_test_split(scaled_train_X, train_y_all,
                                                                        test_size=0.3,
                                                                        random_state=21)
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=False, random_state=21)
fold_idxs = []
best_iteration_all = list()
test_mae_all = list()
train_columns = scaled_train_X.columns.values
oof = np.zeros(len(X_train))
train_score = list()
test_predictions = np.zeros(len(X_test))
predictions = np.zeros(len(scaled_test_X))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train.values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    fold_idxs.append(val_idx)

    X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    model = CatBoostRegressor(n_estimators=10000, verbose=-1, objective="MAE", loss_function="MAE",
                              boosting_type="Ordered", task_type="CPU",thread_count=-1 )#,learning_rate=0.01)
    model.fit(X_tr,
              y_tr,
              eval_set=[(X_val, y_val)],
              #eval_metric='mae',
              verbose=5000,
              early_stopping_rounds=100)
    oof[val_idx] = model.predict(X_val)

    # feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # predictions
    predictions += model.predict(scaled_test_X) / folds.n_splits
    test_predictions += model.predict(X_test) / folds.n_splits
    best_iteration_all.append(model.best_iteration_)
    test_mae = mean_absolute_error(test_predictions, y_test)
    test_mae_all.append(test_mae)
    train_score.append(model.best_score_['learn']["MAE"])
    print('Test Dataset MAE: ', str(test_mae))

cv_score = mean_absolute_error(y_train, oof)
print(f"After {n_fold} test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}", end=" ")
print ('all mae: ', test_mae_all)

# today = str(datetime.date.today())
# submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
#
# submission["time_to_failure"] = predictions
# submission.to_csv(f'CatBoost_{today}_test_{cv_score:.3f}_train_{np.mean(train_score):.3f}.csv', index=False)
# submission.head()

"""
final submission
"""
# Scirpus_prediction = pd.read_csv("../input/andrews-new-script-plus-a-genetic-program-model/gpI.csv")
# Scirpus_prediction.head()
#
# today = str(datetime.date.today())
# submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
#
# submission["time_to_failure"] = (predictions+NN_predictions+Scirpus_prediction.time_to_failure.values)/3
# submission.to_csv(f'FINAL_{today}_submission.csv', index=False)
# submission.head()