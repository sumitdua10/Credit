import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn as sk

# Change the max columns limit to 1000 for display in console
pd.set_option('max_columns', 1000)

TRAIN_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_train\\application_train.csv"
TEST_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_test\\application_test.csv"
BUREAU_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\bureau\\bureau_samplebig.csv"
OUTPUT_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\test_y.csv"
BUREAU_BALANCE_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\bureau_balance\\bureau_balance_samplebig.csv"

def read_file(TRAIN_FILENAME):
    train_df = pd.read_csv(TRAIN_FILENAME)
    print(train_df.shape)
    return train_df


def cleanse(data_df):
    #print(test_df["CODE_GENDER"].unique())
    #data_df = test_df
    data_df.loc[data_df.CODE_GENDER == 'M', 'CODE_GENDER'] = 1
    print("Code Gender 1 Done")
    data_df.loc[data_df.CODE_GENDER == 'F', 'CODE_GENDER'] = 0
    print("Code Gender 0 Done")
    # data_df.loc[data_df.CODE_GENDER == 'XNA', 'CODE_GENDER'] = np.nan
    #data_df["CODE_GENDER"].
    print(data_df.shape)
    data_df["CODE_GENDER"] = data_df["CODE_GENDER"].astype(int)
    print("Type Conversion Done")

    data_df.loc[data_df.FLAG_OWN_CAR == 'Y', 'FLAG_OWN_CAR'] = 1
    data_df.loc[data_df.FLAG_OWN_CAR == 'N', 'FLAG_OWN_CAR'] = 0
    data_df["FLAG_OWN_CAR"] = data_df["FLAG_OWN_CAR"].astype(int)
    print(data_df["FLAG_OWN_CAR"].isnull().sum())

    data_df.loc[data_df.FLAG_OWN_REALTY == 'Y', 'FLAG_OWN_REALTY'] = 1
    data_df.loc[data_df.FLAG_OWN_REALTY == 'N', 'FLAG_OWN_REALTY'] = 0
    data_df["FLAG_OWN_REALTY"] = data_df["FLAG_OWN_REALTY"].astype(int)
    print(data_df["FLAG_OWN_REALTY"].isnull().sum())

    # data_df.groupby(by="EMERGENCYSTATE_MODE", axis=0).count()
    data_df.loc[data_df["EMERGENCYSTATE_MODE"].isna(), "EMERGENCYSTATE_MODE"] = 'N'
    data_df.loc[data_df.EMERGENCYSTATE_MODE == 'Yes', 'EMERGENCYSTATE_MODE'] = 1
    data_df.loc[data_df.EMERGENCYSTATE_MODE == 'No', 'EMERGENCYSTATE_MODE'] = 0
    data_df.loc[data_df.EMERGENCYSTATE_MODE == 'Y', 'EMERGENCYSTATE_MODE'] = 1
    data_df.loc[data_df.EMERGENCYSTATE_MODE == 'N', 'EMERGENCYSTATE_MODE'] = 0

    data_df["EMERGENCYSTATE_MODE"] = data_df["EMERGENCYSTATE_MODE"].astype(int)

    data_df[data_df.select_dtypes(include=[np.object]).columns.values] = data_df[
        data_df.select_dtypes(include=[np.object]).columns.values].fillna("Others")

    data_df.pop("WEEKDAY_APPR_PROCESS_START")

    data_df[data_df.select_dtypes(include=[np.float64]).columns.values] = data_df[
        data_df.select_dtypes(include=[np.float64]).columns.values].fillna(0)
    print(data_df[data_df.select_dtypes(include=[np.float64]).columns.values].isna().sum())

    data_df[data_df.select_dtypes(include=[np.float]).columns.values] = data_df[
        data_df.select_dtypes(include=[np.float]).columns.values].fillna(0)
    print(data_df[data_df.select_dtypes(include=[np.float]).columns.values].isna().sum())

    data_df[data_df.select_dtypes(include=[np.int64]).columns.values] = data_df[
        data_df.select_dtypes(include=[np.int64]).columns.values].fillna(0)
    print(data_df[data_df.select_dtypes(include=[np.int64]).columns.values].isna().sum())

    data_df.pop("SK_ID_CURR")
    return data_df

def repl_catg_columns(tr_df, te_df):
    tr_data_types = tr_df.dtypes
    #te_data_types = te_df.dtypes
    #print(type(tr_data_types))
    #if (tr_data_types == te_data_types):
     #   print("Data Types Match")
    #else:
      #  raise (Exception)
    tr_concat_df = pd.DataFrame(tr_df)
    te_concat_df = pd.DataFrame(te_df)
    # print(concat_df)
    counter_list = []
    for counter, col in enumerate(tr_data_types, 0):
        print(col, " ", counter)
        if col == "object":
            #if [tr_df.iloc[:,counter].unique()] != [te_df.iloc[:,counter].unique()]:
            #    print("Object data type identified ", tr_concat_df.columns[counter],  " at column ", counter)

            # dummy_df = pd.get_dummies(df[cols[counter]])
            # dummy_df = pd.get_dummies((df.iloc[:,2]))
            # print(dummy_df)
            tr_dummy_df = pd.get_dummies(tr_df.iloc[:, counter])
            te_dummy_df = pd.get_dummies(te_df.iloc[:, counter])
            #print(tr_dummy_df.shape)
            #print(te_dummy_df.shape)
            # print(dummy_df)
            # concat_df.pop(df[cols[counter]])#, inplace = True)
            counter_list.append(counter)
            # concat_df = concat_df.drop(concat_df.columns[counter], axis = 1)

            # concat_df = concat_df.pop(df.iloc[:,counter])  # , inplace = True)
            # concat_df.drop(cols[counter])
            tr_concat_df = pd.concat((tr_concat_df, tr_dummy_df), axis=1)
            te_concat_df = pd.concat((te_concat_df, te_dummy_df), axis=1)

            # print(concat_df.shape)
    tr_concat_df = tr_concat_df.drop(tr_concat_df.columns[counter_list], axis=1)
    te_concat_df = te_concat_df.drop(te_concat_df.columns[counter_list], axis=1)

    return tr_concat_df, te_concat_df


test_df = read_file(TEST_FILENAME)
train_df = read_file(TRAIN_FILENAME)

train_df = train_df[train_df["CODE_GENDER"] != "XNA"]
print(train_df.shape)
#print(train_df.iloc[:,1].unique())


cl_test_df = cleanse(test_df)
cl_train_df = cleanse(train_df)

print(cl_train_df.shape)
print(cl_test_df.shape)

print(cl_train_df["NAME_FAMILY_STATUS"].unique())

cl_train_df = cl_train_df[cl_train_df["NAME_INCOME_TYPE"]!="Maternity leave"]
print(cl_train_df.shape)

cl_train_df = cl_train_df[cl_train_df["NAME_FAMILY_STATUS"]!="Unknown"]
print(cl_train_df.shape)


y_train = cl_train_df["TARGET"]
print(y_train.shape)

cl_train_df.pop("TARGET")
print(cl_train_df.shape)
print(cl_test_df.shape)

cat_train_df, cat_test_df = repl_catg_columns(cl_train_df, cl_test_df)

print(cat_train_df.shape)
#cat_val_df = repl_catg_columns(cl_test_df)
print(cat_test_df.shape)

#x = cat_train_df.corr()
#print(x.shape)
#print(type(x))

#print(x.iloc[:2,2])
"""
#print(cat_train_df.shape)
#print(cat_train_df.select_dtypes(include=[np.object]).columns.values)

# Correlation Data
x = cat_train_df.orr
print(type(x))
print(x.shape)
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
pre_train = scaler.fit_transform(cat_train_df)
pre_test = scaler.transform(cat_test_df)

#pre_train = preprocessing.normalize(cat_train_df)
#pre_test = preprocessing.normalize(cat_test_df)
#print(pre_test.shape)


#print(pre_test[1:5,6])

#cor_train = pre_train.corr()
#print(cor_train.shape)

# Split Data

x_train, x_test, y_train, y_test = train_test_split(pre_train, y_train, test_size=0.3)
#poly = PolynomialFeatures(degree = 2)
#x_poly = poly.fit_transform(x_train)
#print(x_poly.shape)

lm = sk.linear_model.LogisticRegression(C = 5, penalty='l2', solver = 'saga', max_iter = 250)
#lm = sk.tree.DecisionTreeClassifier(max_depth=30)
#lm = sk.svm.SVC(C=5.0, kernel='rbf')
print(lm)

#lm = sk.neighbors.KNeighborsClassifier(n_neighbors=11)
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import random
from sklearn.metrics import roc_curve, auc
from sklearn import neural_network
#lm = neural_network.MLPClassifier(hidden_layer_sizes=64)

#random_state = np.random.RandomState(0)
#lm =  OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=random_state))

print("Training the model now")
lm.fit(x_train, y_train)
print("Training completed. let's predict the test set now", )

#print(lm)
#input()

predict_y = lm.predict(x_test)
predict_y_prob = lm.decision_function(x_test)

print("Prediction completed on test set. let's check the accuracy", )
print("Accuracy test set",accuracy_score(y_test, predict_y) )

print("Confusion Metrics - Test Set", sk.metrics.confusion_matrix(y_test,predict_y))
print("F1 Score - Test Set", sk.metrics.f1_score(y_test, predict_y) )

print("ROC Score - Test Set ", sk.metrics.roc_auc_score(y_test, predict_y))


predict_train_y = lm.predict(x_train)
print("Prediction completed for train set. let's check the accuracy", )
print("Accuracy on training set",accuracy_score(y_train, predict_train_y) )

#print(predict_y.shape)
#print(predict_y_prob.shape)

#print(predict_y_prob[600:600][:])
print((predict_y[0:93000]==1).sum())
print((np.array(y_test)[0:93000]==1).sum())

#print(accuracy_score(y_test, predict_y))
#print(sk.metrics.f1_score(y_test, predict_y))

#print(accuracy_score(y_train, predict_train_y))
##print(sk.metrics.f1_score(y_train, predict_train_y))
print("ROC - Training Set ", sk.metrics.roc_auc_score(y_train, predict_train_y))

print((predict_train_y[:]==1).sum())
print((np.array(y_train)[:]==1).sum())



## Final Validation
predict_val_y = lm.predict(pre_test)
predict_val_prob = lm.predict_proba(pre_test)
output = pd.DataFrame(predict_y_prob)
output.to_csv(OUTPUT_FILENAME)
print(output.shape)
print(pre_test.shape)
print(predict_val_y.shape)

