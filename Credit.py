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


#Change the max columns limit to 1000 for display in console
pd.set_option('max_columns', 1000)

TRAIN_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_train\\application_train.csv"
TEST_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_test\\application_test.csv"
BUREAU_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\bureau\\bureau_samplebig.csv"
BUREAU_BALANCE_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\bureau_balance\\bureau_balance_samplebig.csv"

def change(TRAIN_FILENAME):
    TRAIN_FILENAME = "ABC"
    return TRAIN_FILENAME

tt = change(TRAIN_FILENAME)
print(tt)
print(TRAIN_FILENAME)


train_df, test_df  = read_files()

def read_files():
    train_df = pd.read_csv(TRAIN_FILENAME)
    print(train_df.shape)

    test_df = pd.read_csv(TEST_FILENAME)
    print(test_df.shape)
    return train_df, test_df


"""
bureau_df = pd.read_csv(BUREAU_FILENAME)
print(bureau_df.shape)

bureau_balance_df = pd.read_csv(BUREAU_BALANCE_FILENAME)
print(bureau_balance_df.shape)

#train_b_df = pd.merge(train_df, bureau_df, how='inner', left_on = 'SK_ID_CURR', right_on= 'SK_ID_CURR')
#train_b_df = pd.merge(train_df, bureau_df, by = 'SK_ID_CURR')
#train_b_df1 = train_df.merge(bureau_df, how='left',  left_on = 'SK_ID_CURR', right_on= 'SK_ID_CURR')
#print(train_b_df1.shape)
#print(train_df.shape)

#Summarize Bureau Balance File
print(bureau_balance_df.head())
#Filter on only OverDue Accounts. Rest are fine.
bb_DPD_balance_df = bureau_balance_df[bureau_balance_df.STATUS.isin(('1','2','3'))]# == '1' or bureau_balance_df.STATUS =='2']# | bureau_balance_df.STATUS == '2' | bureau_balance_df.STATUS == '3'] #.groupby(by='status')
print(bb_DPD_balance_df.head())
bb_DPD_balance_df.pop("MONTHS_BALANCE")

print(bb_DPD_balance_df.head())

bb_DPD_balance_df = bb_DPD_balance_df.groupby(by="SK_ID_BUREAU").max()
print(bb_DPD_balance_df.head())
bb_DPD_balance_df = bb_DPD_balance_df.reset_index()
print(bb_DPD_balance_df.head())

bb_DPD_balance_df.columns = ("SK_ID_BUREAU", "DPD_STATUS")

#Clean Bureau Balance of NAs. There are no NA Values
print(bb_DPD_balance_df.dtypes)
# Merge Bureau Balance with Bureau file by left outer join on Bureau balance. There should be same no. of rows as of Bureau file
bb_df = bureau_df.merge(bb_DPD_balance_df, how='left', left_on = 'SK_ID_BUREAU', right_on = 'SK_ID_BUREAU')
"""

"""
#print(bb_df.head(10))
print(bb_df.shape)
#print(bb_df.dtypes)
#print(bb_df.describe(include='all'))
showNA(bb_df)


#print(bb_df["DPD_STATUS"].isnull().sum())
bb_df["DPD_STATUS"].fillna("0", inplace = True)
print(bb_df.shape)
bb_df = repl_catg_columns(bb_df)
print(bb_df.shape)
showNA(train_df)
print(train_df.shape)
removeNA(df, 0.8)
train_df.fillna("0", inplace = True)
print(train_df.select_dtypes(include = 'int').columns)
print(train_df.shape)
x = train_df.corr()
print(type(x))
print(x.shape)

"""
#Cleaning of data. Covert 4 columns to integer
#print(train_df.dtypes)
#test_df = pd.DataFrame()
cl_train_df = cleanse(train_df)
cl_test_df = cleanse(test_df)
print(cl_test_df.shape)
"""
print(train_df.select_dtypes(include=[np.number]).shape)
print(train_df[train_df.select_dtypes(include=[np.object]).columns.values].isna().sum())

train_df.loc[train_df.CODE_GENDER == 'M', 'CODE_GENDER'] = 1
train_df.loc[train_df.CODE_GENDER == 'F', 'CODE_GENDER'] = 0
train_df["CODE_GENDER"] = train_df["CODE_GENDER"].astype(int)

train_df.loc[train_df.FLAG_OWN_CAR == 'Y', 'FLAG_OWN_CAR'] = 1
train_df.loc[train_df.FLAG_OWN_CAR == 'N', 'FLAG_OWN_CAR'] = 0
train_df["FLAG_OWN_CAR"] = train_df["FLAG_OWN_CAR"].astype(int)
print(train_df["FLAG_OWN_CAR"].isnull().sum())

train_df.loc[train_df.FLAG_OWN_REALTY == 'Y', 'FLAG_OWN_REALTY'] = 1
train_df.loc[train_df.FLAG_OWN_REALTY == 'N', 'FLAG_OWN_REALTY'] = 0
train_df["FLAG_OWN_REALTY"] = train_df["FLAG_OWN_REALTY"].astype(int)
print(train_df["FLAG_OWN_REALTY"].isnull().sum())

#train_df.groupby(by="EMERGENCYSTATE_MODE", axis=0).count()
train_df.loc[train_df["EMERGENCYSTATE_MODE"].isna(), "EMERGENCYSTATE_MODE"] = 'N'
train_df.loc[train_df.EMERGENCYSTATE_MODE == 'Yes', 'EMERGENCYSTATE_MODE'] = 1
train_df.loc[train_df.EMERGENCYSTATE_MODE == 'No', 'EMERGENCYSTATE_MODE'] = 0
train_df.loc[train_df.EMERGENCYSTATE_MODE == 'Y', 'EMERGENCYSTATE_MODE'] = 1
train_df.loc[train_df.EMERGENCYSTATE_MODE == 'N', 'EMERGENCYSTATE_MODE'] = 0

train_df["EMERGENCYSTATE_MODE"] = train_df["EMERGENCYSTATE_MODE"].astype(int)

train_df[train_df.select_dtypes(include=[np.object]).columns.values] = train_df[train_df.select_dtypes(include=[np.object]).columns.values].fillna("Others")

train_df.pop("WEEKDAY_APPR_PROCESS_START")
train_df.pop("SK_ID_CURR")

#Covert the category columns to dummy columns
print(train_df.select_dtypes(include=[np.float]).columns.values)
train_df[train_df.select_dtypes(include=[np.float64]).columns.values] = train_df[train_df.select_dtypes(include=[np.float64]).columns.values].fillna(0)
print(train_df[train_df.select_dtypes(include=[np.float64]).columns.values].isna().sum())

train_df[train_df.select_dtypes(include=[np.float]).columns.values] = train_df[train_df.select_dtypes(include=[np.float]).columns.values].fillna(0)
print(train_df[train_df.select_dtypes(include=[np.float]).columns.values].isna().sum())


train_df[train_df.select_dtypes(include=[np.int64]).columns.values] = train_df[train_df.select_dtypes(include=[np.int64]).columns.values].fillna(0)
print(train_df[train_df.select_dtypes(include=[np.int64]).columns.values].isna().sum())

print(train_df.shape)
"""
y = train_df.iloc[:,0]
print(y.shape)

y_val = test_df.iloc[:,0]

train_df.pop("SK_ID_CURR")
cl_test_df.pop("SK_ID_CURR")



cat_train_df = repl_catg_columns(train_df)
print(cat_train_df.shape)
cat_val_df = repl_catg_columns(cl_test_df)
print(cat_val_df.shape)

print(cat_train_df.shape)
print(cat_train_df.select_dtypes(include=[np.object]).columns.values)

#Correlation Data
x = cat_train_df.corr()
print(type(x))
print(x.shape)



# Split Data

x_train,  x_test, y_train, y_test = train_test_split(cat_train_df, y, test_size=0.3)

lm = sk.linear_model.LogisticRegression()

print("Training the model now")
lm.fit(x_train, y_train)

print("Training completed. let's predict the test set now", )
print(lm)
input()

predict_y = lm.predict(x_test)
print("Prediction completed. let's check the accuracy", )
print(accuracy_score(y_test, predict_y))


def cleanse(train_df):
    train_df.loc[train_df.CODE_GENDER == 'M', 'CODE_GENDER'] = 1
    train_df.loc[train_df.CODE_GENDER == 'F', 'CODE_GENDER'] = 0
    #train_df.loc[train_df.CODE_GENDER == 'XNA', 'CODE_GENDER'] = np.nan
    train_df = train_df[train_df["CODE_GENDER"] != "XNA"]
    print(train_df.shape)
    train_df["CODE_GENDER"] = train_df["CODE_GENDER"].astype(int)

    train_df.loc[train_df.FLAG_OWN_CAR == 'Y', 'FLAG_OWN_CAR'] = 1
    train_df.loc[train_df.FLAG_OWN_CAR == 'N', 'FLAG_OWN_CAR'] = 0
    train_df["FLAG_OWN_CAR"] = train_df["FLAG_OWN_CAR"].astype(int)
    print(train_df["FLAG_OWN_CAR"].isnull().sum())

    train_df.loc[train_df.FLAG_OWN_REALTY == 'Y', 'FLAG_OWN_REALTY'] = 1
    train_df.loc[train_df.FLAG_OWN_REALTY == 'N', 'FLAG_OWN_REALTY'] = 0
    train_df["FLAG_OWN_REALTY"] = train_df["FLAG_OWN_REALTY"].astype(int)
    print(train_df["FLAG_OWN_REALTY"].isnull().sum())

    # train_df.groupby(by="EMERGENCYSTATE_MODE", axis=0).count()
    train_df.loc[train_df["EMERGENCYSTATE_MODE"].isna(), "EMERGENCYSTATE_MODE"] = 'N'
    train_df.loc[train_df.EMERGENCYSTATE_MODE == 'Yes', 'EMERGENCYSTATE_MODE'] = 1
    train_df.loc[train_df.EMERGENCYSTATE_MODE == 'No', 'EMERGENCYSTATE_MODE'] = 0
    train_df.loc[train_df.EMERGENCYSTATE_MODE == 'Y', 'EMERGENCYSTATE_MODE'] = 1
    train_df.loc[train_df.EMERGENCYSTATE_MODE == 'N', 'EMERGENCYSTATE_MODE'] = 0

    train_df["EMERGENCYSTATE_MODE"] = train_df["EMERGENCYSTATE_MODE"].astype(int)

    train_df[train_df.select_dtypes(include=[np.object]).columns.values] = train_df[train_df.select_dtypes(include=[np.object]).columns.values].fillna("Others")

    train_df.pop("WEEKDAY_APPR_PROCESS_START")

    train_df[train_df.select_dtypes(include=[np.float64]).columns.values] = train_df[train_df.select_dtypes(include=[np.float64]).columns.values].fillna(0)
    print(train_df[train_df.select_dtypes(include=[np.float64]).columns.values].isna().sum())

    train_df[train_df.select_dtypes(include=[np.float]).columns.values] = train_df[train_df.select_dtypes(include=[np.float]).columns.values].fillna(0)
    print(train_df[train_df.select_dtypes(include=[np.float]).columns.values].isna().sum())


    train_df[train_df.select_dtypes(include=[np.int64]).columns.values] = train_df[train_df.select_dtypes(include=[np.int64]).columns.values].fillna(0)
    print(train_df[train_df.select_dtypes(include=[np.int64]).columns.values].isna().sum())

    return train_df


"""
print(train_df[train_df.select_dtypes(include=[np.float64]).columns.values].isna().sum())
train_df.pop("WEEKDAY_APPR_PROCESS_START")
print(train_df.shape)

#print(train_df["FLAG_OWN_REALTY"].isnull().sum())

#print(train_df["EMERGENCYSTATE_MODE"].isnull().sum())
#print(train_df.shape)
#train_df["NAME_TYPE_SUITE"].fillna("Others")
showNA(train_df)



#bb_df["DPD_STATUS"] = bb_df["DPD_STATUS"].astype('int')
#pd.to_numeric(bb_df["DPD_STATUS"])
#print(bb_df["CREDIT_ACTIVE"].unique())

i=0
print(bb_df.loc[6].isnull().sum())
for i in range(bb_df.shape[1]):
    #if (bb_df.iloc[:,i].isnull().sum() / bb_df.iloc[:,i].count())
    print("column no ", i ," ", bb_df.iloc[:,i].isnull().sum())

print(bb_df.dtypes)
"""



"""
df = pd.DataFrame(np.random.randn(5,4))
df.iloc[1:3,1] = np.nan
df.iloc[4,1:3] = np.nan

print(df)
df["b"] = "Active"
df["c"] = "INR"

df.iloc[0,4] = "Closed"
dfn = pd.get_dummies(df.iloc[:,1])
print(dfn)
df.pop("b")
dfnew = pd.concat([df, dfn], axis=1)
print(dfnew)
df_new = 
df.columns = [["sk_id", "b", "c", "d", "e", "f"]]
dfn = df.drop(df.columns[[0,1]], axis=1)

print(df.iloc[:,3].count())
df_new = removeNA(df, 0.5)
print(df_new)
print(df.shape[1])
"""
#print(df)

def showNA(df):
    df_g = pd.DataFrame(np.zeros(shape = (df.shape[1],2)))
    df_g.iloc[:,0] = df.columns
    for i in range(df_g.shape[0]):
        df_g.iloc[i,1] = df.iloc[:, i].isnull().sum()
    #print(df_g)
    df_g.plot(kind = "bar")
    plt.show()
    #print(df_g) #df[].plot(kind = 'Bar')


def repl_catg_columns(tr_df, te_df):
    tr_data_types = tr_df.dtypes
    te_data_types = te_df.dypes
    if(tr_data_types.values() == te_data_types.values()):
        print("Data Types Match")
    else:
        raise (Exception)

    tr_concat_df = pd.DataFrame(tr_df)
    #print(concat_df)
    counter_list = []
    for counter, col in enumerate (tr_data_types, 0):
        print(col, " ", counter)
        if col == "object":
            print("Object data type identified ", col, " at column ", counter)
            #dummy_df = pd.get_dummies(df[cols[counter]])
            #dummy_df = pd.get_dummies((df.iloc[:,2]))
            #print(dummy_df)
            dummy_df = pd.get_dummies(df.iloc[:,counter])
            #print(dummy_df)
            #concat_df.pop(df[cols[counter]])#, inplace = True)
            counter_list.append(counter)
            #concat_df = concat_df.drop(concat_df.columns[counter], axis = 1)

            #concat_df = concat_df.pop(df.iloc[:,counter])  # , inplace = True)
            #concat_df.drop(cols[counter])
            concat_df = pd.concat((concat_df,dummy_df), axis=1)
            #print(concat_df.shape)
    concat_df = concat_df.drop(concat_df.columns[counter_list], axis=1)
    return concat_df

print(df)
concat_df = rep_cat_columns(df)
print(concat_df)


def removeNA(df, NA_threshold):
    rows = df.shape[0]
    cols_delete = []
    for i in range(df.shape[1]):
        NA_rows = df.iloc[:,i].isnull().sum()
        if (NA_rows / rows) >= NA_threshold:
         #       #df_new = df.pop(df.iloc[:,i])
                print("Removing column no. ", i, " as NA percentage is ",(NA_rows / rows ))
                cols_delete.append(i)
        #print(NA_rows)
        #print(cols_delete)

    #for i in cols_delete:
    df_new =df.drop(df.columns[cols_delete], axis = 1)
    return df_new


concatnew = removeNA(concat_df,NA_threshold= 0.1)
print(concatnew)




