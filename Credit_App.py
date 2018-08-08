from sklearn.model_selection import GridSearchCV
import lightgbm
import credit_dep as cd
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn as sk

# Change the max columns limit to 1000 for display in console
pd.set_option('max_columns', 1000)

PREV_APP_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\previous_application\\previous_application.csv"
TRAIN_META_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_train\\application_train_meta.csv"
TEST_META_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_train\\application_test_meta.csv"
TRAIN_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_train\\application_train.csv"
TEST_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_test\\application_test.csv"
#BUREAU_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\bureau\\bureau_samplebig.csv"
OUTPUT_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\test_y.csv"
#BUREAU_BALANCE_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\bureau_balance\\bureau_balance_samplebig.csv"

# Step 1 - Read the files
test_df = cd.read_file(TEST_FILENAME)
train_df = cd.read_file(TRAIN_FILENAME)
output = test_df["SK_ID_CURR"]

print("Training Set Size", train_df.shape)
print("Test_Set Size", test_df.shape)

train_df = train_df[train_df["CODE_GENDER"] != "XNA"]
train_df = train_df[train_df["NAME_INCOME_TYPE"]!="Maternity leave"]

train_df = train_df[train_df["NAME_FAMILY_STATUS"]!="Unknown"]

print("Training Set Size", train_df.shape)

#print(train_df.iloc[:,1].unique())
#print(train_df.select_dtypes(include=[np.float64]).columns.values)

# Step 2 - clean the data frames i.e. clean NA values, covert the object data type to Numeric
num_train_null_cols = cd.assess(train_df, write_to_file = False )
print(num_train_null_cols, " columns have null values out of total train set", train_df.shape[1],". Cleaning them.....press any key to continue")
train_df = cd.cleanse(train_df)
print("Training Set Size", train_df.shape)


num_test_null_cols = cd.assess(test_df, write_to_file = False, FILE_NAME=TEST_META_FILENAME )
print(num_test_null_cols, " columns have null values out of total test set", test_df.shape[1],". Cleaning them.....press any key to continue")
test_df = cd.cleanse(test_df)
print("Test_Set Size", test_df.shape)

train_df, test_df = cd.clean_NA(train_df, test_df)

print(train_df.shape)
print(test_df.shape)
print(train_df.isna().sum().sum())
print(test_df.isna().sum().sum())

# Step 3 - Label encode the category columns. You can also use the hot encoding using function rep_cat_columns
# alternate method - train_df, test_df = cd.repl_catg_columns(train_df, test_df)
train_df = cd.label_encode(train_df)
test_df = cd.label_encode(test_df)

#train_df = cd.binary_encoding(train_df)
#input()

print(train_df.shape)
print(test_df.shape)


# Step 4 - combine with data the bureau tables and bureal balance table
x = cd.get_cleaned_bureau_data()

train_df = train_df.merge(x, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Training Set Shape after merging with Bureau", train_df.shape)

print("Test Set Shape before merging with Bureau", test_df.shape)
test_df = test_df.merge(x, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Test Set Shape after merging with Bureau", test_df.shape)

print(train_df.shape)
train_df = train_df.fillna(0)
#train_df.pop("TARGET")
print(train_df.isna().sum().sum())


print(test_df.isna().sum().sum())
test_df = test_df.fillna(0)
print(train_df.shape)
print(test_df.shape)
print(test_df.columns)

# Step 5: Join with Previous Applications Balances DAta
print(train_df.shape)
#print(dtrain_df.columns[:5])

agg_pa_df =  cd.get_prev_app_data()

ctrain_df = train_df.merge(agg_pa_df, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Training Set Shape after merging with Bureau", ctrain_df.shape)
print(ctrain_df.isna().sum().sum())
train_df = ctrain_df.fillna(0)
print(train_df.isna().sum().sum())

test_df = test_df.merge(agg_pa_df, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Test Set Shape after merging with Bureau", test_df.shape)
print(test_df.isna().sum().sum())
test_df = test_df.fillna(0)
print(test_df.isna().sum().sum())

data_df =  cd.get_prev_bal_data()
print(data_df.head())
print(train_df.shape)
train_df = train_df.merge(data_df, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Training Set Shape after merging with Bureau", train_df.shape)
print(train_df.isna().sum())
train_df = train_df.fillna(0)
print(train_df.shape)
test_df = test_df.merge(data_df, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Test Set Shape after merging with Bureau", test_df.shape)
print(test_df.isna().sum().sum())
test_df = test_df.fillna(0)


cc_bal_df = cd.get_cred_file_data()

train_df = train_df.merge(cc_bal_df, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Training Set Shape after merging with Cred Bal", train_df.shape)
print(train_df.isna().sum().sum())
train_df = train_df.fillna(0)
print(train_df.shape)
test_df = test_df.merge(cc_bal_df, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Test Set Shape after merging with Cred Bal", test_df.shape)
print(test_df.isna().sum().sum())
test_df = test_df.fillna(0)

in_pay_df = cd.get_install_payments_data()

train_df = train_df.merge(in_pay_df, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Training Set Shape after merging with install payments", train_df.shape)
print(train_df.isna().sum().sum())
train_df = train_df.fillna(0)
print(train_df.shape)

test_df = test_df.merge(in_pay_df, how='left', left_on='SK_ID_CURR', right_on='SK_ID_CURR')
print("Test Set Shape after merging with Install Payments", test_df.shape)
print(test_df.isna().sum().sum())
test_df = test_df.fillna(0)



#train_df, test_df = cd.corr_extract(train_df,test_df)


ctrain_df = train_df.drop("SK_ID_CURR", axis = 1)
ctest_df = test_df.drop("SK_ID_CURR", axis = 1)
print(train_df.shape)
print(ctrain_df.shape)
print(test_df.shape)
print(ctest_df.shape)
#train_df.pop("SK_ID_CURR")
#test_df.pop("SK_ID_CURR")

#Step 6. Idenfify the cols that are not coorelated using pearson coeff. Remove non required colsCorrelation Data
#dtrain_df, dtest_df=  cd.corr_extract(train_df.iloc[:,1:], test_df.iloc[:,1:])
#dtrain_df, dteste_df = cd.corr
#print(cor_train.shape)



y_total_train = ctrain_df["TARGET"]
ctrain_df.pop("TARGET")
print(y_total_train.shape)

#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(ctrain_df)
#scaler = StandardScaler().fit(ctrain_df)

pre_train = scaler.transform(ctrain_df)
pre_test = scaler.transform(ctest_df)


#print(cat_train_df.shape)
#print(cat_train_df.select_dtypes(include=[np.object]).columns.values)

#pre_train = preprocessing.normalize(cat_train_df)
#pre_test = preprocessing.normalize(cat_test_df)
#print(pre_test.shape)

# Split Data
x_train, x_test, y_train, y_test = train_test_split(pre_train, y_total_train, test_size=0.3, random_state=0)



#poly = PolynomialFeatures(degree = 2)
#x_poly = poly.fit_transform(x_train)
#print(x_poly.shape)

#lm = sk.linear_model.LogisticRegression(C = 5, penalty='l2', solver = 'saga', max_iter = 450)

#lm = GaussianNB()
#import xgboost
#from xgboost import XGBClassifier
#lm = XGBClassifier()

parameters = {'n_estimators':[75,100,125,150,200], 'learning_rate':[0.01, 0.1,0.5, 1, 5], 'max_depth':[2,4,6,8], 'max_features':[0.2,0.4,0.6,0.8]}
lm1 = GradientBoostingClassifier()#n_estimators=100, learning_rate=0.1, max_depth=4, max_features=0.2)
lm = GridSearchCV(lm1, param_grid=parameters, scoring='roc_auc')
#lm = sk.ensemble.RandomForestClassifier(n_estimators=20, max_features=0.5)
#lm = sk.neural_network.MLPClassifier(hidden_layer_sizes=(200,200, 200), alpha=0.01, activation='relu')
#lm = sk.tree.DecisionTreeClassifier(max_depth=30)
#lm = sk.svm.SVC(C=5.0, kernel='rbf')
#print(lm)

#lm = sk.neighbors.KNeighborsClassifier(n_neighbors=11)
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import random
from sklearn.metrics import roc_curve, auc
#from sklearn import neural_network
#lm = neural_network.MLPClassifier(hidden_layer_sizes={100,100,100}, alpha=0.01,  activation='relu')

#random_state = np.random.RandomState(0)
#lm =  OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=random_state))

print("Training the model now")
lm.fit(x_train, y_train)
#x_train_dataset = lightgbm.Dataset(x_train, label = y_train)
#x_test_dataset =  lightgbm.Dataset(x_test, label = y_test)
#parameters = {   'application': 'binary', 'objective': 'binary', 'metric': 'auc', 'is_unbalance': 'true', 'boosting': 'gbdt', 'num_leaves': 200, 'feature_fraction': 0.5,
#    'bagging_fraction': 0.5, 'bagging_freq': 20, 'learning_rate': 0.5, 'verbose': 0 } #'num_leaves': 31

#lm = lightgbm.train(parameters, x_train_dataset, num_boost_round=5000, early_stopping_rounds=1000, valid_sets= x_test_dataset)

print("Training completed. let's predict the test set now", )

print(lm.best_params_)
print("Best Score")
print(lm.best_score_)

predict_y = lm.predict(x_test)
#predict_y_prob = lm.decision_function(x_test)
predict_y_prob = lm.predict_log_proba(x_test)
print("Prediction completed on test set. let's check the accuracy", )
print("Confusion Metrics - Test Set", sk.metrics.confusion_matrix(y_test,predict_y))
print("F1 Score - Test Set", sk.metrics.f1_score(y_test, predict_y) )
print("ROC Score - Test Set ", sk.metrics.roc_auc_score(y_test, predict_y_prob[:,1]))


print("Let's check the accuracy for training set:", )
predict_train_y = lm.predict(x_train)
print("Confusion Metrics - Train Set", sk.metrics.confusion_matrix(y_train,predict_train_y))
print("F1 Score Train Set", sk.metrics.f1_score(y_train, predict_train_y))
print("ROC - Training Set ",sk.metrics.roc_auc_score(y_train, lm.predict_log_proba(x_train)[:,1]) )
#print("ROC - Training Set ",sk.metrics.roc_auc_score(y_train, lm.decision_function(x_train)) )

## Final Validation
#predict_val_y = lm.predict(pre_test)
#print(test_df.iloc[0:5,0])
#predict_val_prob = lm.decision_function(pre_test)
predict_val_prob = lm.predict_proba(pre_test)
#print(predict_val_prob.shape)
#print(predict_val_prob[0:5])
predict_val_prob_df = pd.Series(data = predict_val_prob[:,1])
print(predict_val_prob_df.shape)
#output = pd.DataFrame(predict_y_prob)
final_output = pd.concat((output, predict_val_prob_df), axis = 1)
print(final_output.shape)
#final_output.to_csv(OUTPUT_FILENAME, header=['SK_ID_CURR', 'TARGET'])


