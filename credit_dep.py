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
import sklearn as sk

# Change the max columns limit to 1000 for display in console
pd.set_option('max_columns', 1000)

TRAIN_META_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_train\\application_train_meta.csv"
TEST_META_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_train\\application_test_meta.csv"
TRAIN_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_train\\application_train.csv"
TEST_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\application_test\\application_test.csv"
BUREAU_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\bureau\\bureau.csv"
OUTPUT_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\test_y.csv"
BUREAU_BALANCE_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\bureau_balance\\bureau_balance.csv"
PREV_APP_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\previous_application\\previous_application.csv"
PREV_BAL_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\POS_CASH_balance.CSV\\POS_CASH_balance.csv"
CRED_BAL_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\credit card balance\\credit_card_balance.csv"
INST_PMT_FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Kaggle Credit\\installments payments\\installments_payments.csv"

def get_install_payments_data():
    in_pay_df = pd.read_csv(INST_PMT_FILENAME)
    print("Sahape after reading installment payment file ", in_pay_df.shape)
    in_pay_df['PMT_LATE_DAYS'] = in_pay_df['DAYS_ENTRY_PAYMENT'] - in_pay_df['DAYS_INSTALMENT']
    in_pay_df.loc[in_pay_df['PMT_LATE_DAYS']<0, 'PMT_LATE_DAYS'] = 0
    in_pay_df['AMOUNT_DUE'] =  in_pay_df['AMT_INSTALMENT'] - in_pay_df['AMT_PAYMENT']
    in_pay_df.loc[in_pay_df['AMOUNT_DUE'] < 0, 'AMOUNT_DUE'] =  0

    in_pay_df = in_pay_df[['SK_ID_CURR', 'PMT_LATE_DAYS', 'AMOUNT_DUE']]
    in_pay_df = in_pay_df.groupby(by = 'SK_ID_CURR').agg(np.sum)
    print("Shape after aggregation of installment payment file", in_pay_df.shape)
    print(in_pay_df.head())
    return in_pay_df



def get_cred_file_data():
    cc_bal_df = pd.read_csv(CRED_BAL_FILENAME)
    print("Balance file shape is ", cc_bal_df.shape)
    print(" Shape of credit balance file is ", cc_bal_df.shape)
    # print(data_df.columns)
    cc_bal_df = cc_bal_df[['SK_ID_CURR', 'SK_DPD_DEF']]

    cc_bal_df = cc_bal_df.groupby(by='SK_ID_CURR').agg(np.sum)
    print("Shape after aggregation ", cc_bal_df.shape)
    print(cc_bal_df.head())
    return cc_bal_df
    #cc_bal_df.pop("")

from sklearn.feature_extraction import FeatureHasher
#import category_encoders.binary.BinaryEncoder

def hash_encoding(data_df1, data_df2):
    #data_df1 = train_df.copy()
    #data_df2 = test_df.copy()
    i = 0
    #for col in ('NAME_INCOME_TYPE'): #data_df1.select_dtypes(include=[np.object]).columns:
    #if(True):
    print(data_df1.isnull().sum().sum())
    print(data_df1.head())
    print(data_df1.shape)
    #print(train_df.shape)
    col = 'NAME_INCOME_TYPE'
    print(data_df1[col].tail())
    #col = 'NAME_EDUCATION_TYPE'
    print(data_df1[col].isna().sum())
    print(data_df1[col].unique())
    #data_df1[col] = data_df1[col].fillna('unknown')
    #fh = FeatureHasher(n_features=int(len(data_df1[col].unique())/2), input_type='string')
    #fh = BinaryEncoder()
    hashed_features = fh.fit_transform(data_df1[col])
    hashed_features = hashed_features.toarray()
    data_df1 = pd.concat([data_df1, pd.DataFrame(hashed_features)], axis=1)
    data_df1.pop(col)
    print(data_df1.head)
    print(data_df1.columns)

    print("Total object cols ", data_df1.select_dtypes(include=[np.object]).shape[1])
    print("Total object cols ", data_df1.select_dtypes(include=[np.object]).columns)


    print("Total object cols ", data_df2.select_dtypes(include=[np.object]).shape[1])
    print("Total object cols ", data_df2.select_dtypes(include=[np.object]).columns)
    print(data_df2['NAME_CONTRACT_TYPE'].unique())

    print(int(len(data_df1['NAME_INCOME_TYPE'].unique())/2))
    print(data_df1['OCCUPATION_TYPE'].unique())

    fh = FeatureHasher(n_features=6, input_type='string')



    #fh = FeatureHasher(n_features=6, input_type='string')

    i = 0
    #for col in test_df.select_dtypes(include=[np.object]).columns:
     #   le.fit(test_df[col])
      #  test_df[col] = le.transform(test_df[col])
       # i += 1
    #print("Total cols label encoded - ", i)
    #print("Total object cols after labelencoding ", test_df.select_dtypes(include=[np.object]).shape[1])


def get_prev_app_data():
    print("Reading Previous Applications File name......")
    pa_df = pd.read_csv(PREV_APP_FILENAME)
    print("Size of previous app file name is ", pa_df.shape)
    pa_df = pa_df[pa_df['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y']
    print(pa_df['NAME_CONTRACT_STATUS'].value_counts())
    cols_to_retain  = ['SK_ID_CURR', 'AMT_CREDIT', 'NAME_CONTRACT_STATUS', 'CNT_PAYMENT']
    pa_df = pa_df[cols_to_retain]
    print(pa_df.isna().sum())
    pa_df.loc[pa_df['NAME_CONTRACT_STATUS']== 'Approved', 'NAME_CONTRACT_STATUS'] =  '0'
    pa_df.loc[pa_df['NAME_CONTRACT_STATUS'] == 'Canceled', 'NAME_CONTRACT_STATUS'] = '0'
    pa_df.loc[pa_df['NAME_CONTRACT_STATUS'] == 'Refused', 'NAME_CONTRACT_STATUS'] = '2'
    pa_df.loc[pa_df['NAME_CONTRACT_STATUS'] == 'Unused offer', 'NAME_CONTRACT_STATUS'] = '0'
    #pa_df['NAME_CONTRACT_STATUS'] =
    print(pa_df.dtypes)
    pa_df['NAME_CONTRACT_STATUS'] = pa_df['NAME_CONTRACT_STATUS'].astype(float)
    agg_pa_df = pa_df.groupby(by = "SK_ID_CURR").agg({'AMT_CREDIT':'sum', 'NAME_CONTRACT_STATUS': 'sum', 'CNT_PAYMENT': 'mean'})
    print("After agg shape ", agg_pa_df.shape)
    print(agg_pa_df.head())
    agg_pa_df = agg_pa_df.reset_index()
    agg_pa_df = agg_pa_df[['SK_ID_CURR', 'AMT_CREDIT', 'NAME_CONTRACT_STATUS']]
    agg_pa_df = agg_pa_df.fillna(0)
    print(agg_pa_df.isna().sum())
    return agg_pa_df


    #cl_pa_df = pa_df[]

def corr_extract(train_df, test_df):

    corr_m = train_df.iloc[:,1:].corrwith(train_df["TARGET"])
    #corr_m = x_train.corrwith(y_train)
    print(corr_m.shape)
    #print(corr_m.columns[:5])
    input()

    corr_m = corr_m.reset_index()
    print(corr_m.shape)
    print(corr_m[:20])

    corr_m[0] = np.where(corr_m[0]<0, corr_m[0] * -1, corr_m[0])
    corr_m = corr_m.sort_values(by = [0])
    corr_m = corr_m.reset_index()
    corr_m = corr_m[corr_m['index'] != 'TARGET']
    print(corr_m.head())
    cols_to_delete = list(corr_m.loc[:9,'index'])
    #cols_to_delete = corr_m[(corr_m.iloc[:,1] >= -0.005) & (corr_m.iloc[:,1] <=  0.005)]['index']
    print(cols_to_delete)
    #cols_to_delete = corr_m.columns[0:10]['index']
    #print(cols_to_delete[0:10])

    print(len(list(cols_to_delete)))
    print(cols_to_delete)
    train_df = train_df.drop(cols_to_delete, axis = 1)
    print(test_df.shape)
    test_df= test_df.drop(cols_to_delete, axis=1)
    return train_df, test_df
    print(corr_m.shape)
    # print(train_df["NAME_FAMILY_STATUS"].unique())

    #for col in list(cols_to_delete):
     #   if(col != 'SK_ID_CURR'):
      #      train_df.pop(col)

    #for col in list(cols_to_delete):
     #   if (col != 'SK_ID_CURR'):
      #      test_df.pop(col)

    #print(train_df.shape)
    #print(test_df.shape)
    #return train_df, test_df



def get_prev_bal_data():
    data_df = pd.read_csv(PREV_BAL_FILENAME)
    print(" Shape of prev app balance file is ", data_df.shape)
    #print(data_df.columns)
    data_df = data_df[['SK_ID_CURR', 'SK_DPD_DEF']]
    data_df = data_df[['SK_ID_CURR', 'SK_DPD_DEF']]
    data_df = data_df.groupby(by = 'SK_ID_CURR').agg(np.sum)
    print("Shape after aggregation ", data_df.shape)
    return data_df




def label_encode(test_df):
    le = preprocessing.LabelEncoder()
    print("Total object cols ", test_df.select_dtypes(include=[np.object]).shape[1])
    i=0
    for col in test_df.select_dtypes(include=[np.object]).columns:
        le.fit(test_df[col])
        test_df[col]= le.transform(test_df[col])
        i+=1
    print("Total cols label encoded - ", i)
    print("Total object cols after labelencoding ", test_df.select_dtypes(include=[np.object]).shape[1])
    return test_df

def assess(data_df, write_to_file = False, FILE_NAME = None):
    x = data_df[data_df.columns.values].isna().sum()

    #print(type(x))
    y= data_df.dtypes
    #print(x.shape)
    z=pd.concat([x,y],axis=1)
    if(write_to_file):
        z.to_csv(FILE_NAME)
    z.columns = [ 'NA_Values', 'DType']
    z_filtered = z[z['NA_Values']!=0]
    return z_filtered.shape[0]

def read_file(TRAIN_FILENAME):
    train_df = pd.read_csv(TRAIN_FILENAME)
    print(train_df.shape)
    return train_df



def cleanse(data_df):
    #print(test_df["CODE_GENDER"].unique())
    #data_df = test_df
    print("Inside Function")
    data_df.loc[data_df.CODE_GENDER == 'M', 'CODE_GENDER'] = 1
    print("Code Gender 1 Done")
    data_df.loc[data_df.CODE_GENDER == 'F', 'CODE_GENDER'] = 0
    print("Code Gender 0 Done")
    # data_df.loc[data_df.CODE_GENDER == 'XNA', 'CODE_GENDER'] = np.nan
    #data_df["CODE_GENDER"].
    #print(data_df.shape)
    data_df["CODE_GENDER"] = data_df["CODE_GENDER"].astype(int)
    #print("Type Conversion of field Code Gender Done")

    data_df.loc[data_df.NAME_CONTRACT_TYPE == 'Cash loans', 'NAME_CONTRACT_TYPE'] = 1

    data_df.loc[data_df.NAME_CONTRACT_TYPE == 'Revolving loans', 'NAME_CONTRACT_TYPE'] = 0

    # data_df.loc[data_df.CODE_GENDER == 'XNA', 'CODE_GENDER'] = np.nan
    # data_df["CODE_GENDER"].
    # print(data_df.shape)
    data_df["NAME_CONTRACT_TYPE"] = data_df["NAME_CONTRACT_TYPE"].astype(int)
    print(" Name contract type done.")

    data_df.loc[data_df.FLAG_OWN_CAR == 'Y', 'FLAG_OWN_CAR'] = 1
    data_df.loc[data_df.FLAG_OWN_CAR == 'N', 'FLAG_OWN_CAR'] = 0
    data_df["FLAG_OWN_CAR"] = data_df["FLAG_OWN_CAR"].astype(int)
    #print(data_df["FLAG_OWN_CAR"].isnull().sum())

    data_df.loc[data_df.FLAG_OWN_REALTY == 'Y', 'FLAG_OWN_REALTY'] = 1
    data_df.loc[data_df.FLAG_OWN_REALTY == 'N', 'FLAG_OWN_REALTY'] = 0
    data_df["FLAG_OWN_REALTY"] = data_df["FLAG_OWN_REALTY"].astype(int)
    #print(data_df["FLAG_OWN_REALTY"].isnull().sum())

    pop_cols1 = ['AMT_GOODS_PRICE',	'NAME_TYPE_SUITE',	'REGION_POPULATION_RELATIVE',	'OWN_CAR_AGE',	'FLAG_MOBIL',	'REGION_RATING_CLIENT',	'WEEKDAY_APPR_PROCESS_START',	'HOUR_APPR_PROCESS_START',
                 'BASEMENTAREA_AVG',	'COMMONAREA_AVG',	'ELEVATORS_AVG',	'ENTRANCES_AVG',	'FLOORSMIN_AVG',	'LIVINGAPARTMENTS_AVG',	'LIVINGAREA_AVG',	'NONLIVINGAPARTMENTS_AVG',
                 'NONLIVINGAREA_AVG',	'BASEMENTAREA_MODE',	'COMMONAREA_MODE',	'ELEVATORS_MODE',	'ENTRANCES_MODE',	'FLOORSMIN_MODE',	'LANDAREA_MODE',	'LIVINGAPARTMENTS_MODE',
                 'NONLIVINGAPARTMENTS_MODE',	'NONLIVINGAREA_MODE',	'BASEMENTAREA_MEDI',	'COMMONAREA_MEDI',	'ELEVATORS_MEDI',	'ENTRANCES_MEDI',	'FLOORSMIN_MEDI',	'LIVINGAPARTMENTS_MEDI',
                 'LIVINGAREA_MEDI',	'NONLIVINGAPARTMENTS_MEDI',	'NONLIVINGAREA_MEDI',	'FONDKAPREMONT_MODE',	'EMERGENCYSTATE_MODE',]

    print(data_df.shape)
    data_df = data_df.drop(pop_cols1, axis = 1)
    print(data_df['OCCUPATION_TYPE'].isna().sum())
    print(data_df['NAME_INCOME_TYPE'].isna().sum())
    data_df['OCCUPATION_TYPE'] = np.where(data_df['NAME_INCOME_TYPE'] == 'Pensioner', 'Pensioner', data_df['OCCUPATION_TYPE'])
    #data_df['NAME_TYPE_SUITE'] = data_df['NAME_TYPE_SUITE'].fillna('Unaccompanied')
    data_df['OCCUPATION_TYPE'] = data_df['OCCUPATION_TYPE'].fillna('Unknown')
    data_df['HOUSETYPE_MODE'] = data_df['HOUSETYPE_MODE'].fillna('block of flats')
    data_df['WALLSMATERIAL_MODE'] = data_df['WALLSMATERIAL_MODE'].fillna('Unknown')
    #data_df['EMERGENCYSTATE_MODE']= data_df['EMERGENCYSTATE_MODE'].fillna('No')
    print(data_df.select_dtypes(include =['object']).columns)
    data_df['REG_REGION_NOT_LIVE'] = data_df['REG_REGION_NOT_LIVE_REGION'] * data_df['REG_CITY_NOT_LIVE_CITY']
    data_df['REG_REGION_NOT_WORK'] = data_df['REG_REGION_NOT_WORK_REGION'] * data_df['REG_CITY_NOT_WORK_CITY']
    data_df['LIVE_REGION_NOT_WORK'] = data_df['LIVE_REGION_NOT_WORK_REGION'] * data_df['LIVE_CITY_NOT_WORK_CITY']

    pop_cols2 = ['REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY']

    data_df = data_df.drop(pop_cols2, axis=1)

    data_df["FLAG_DOCUMENT"] = data_df["FLAG_DOCUMENT_2"] + data_df["FLAG_DOCUMENT_3"] + data_df["FLAG_DOCUMENT_4"] + data_df["FLAG_DOCUMENT_5"] + data_df["FLAG_DOCUMENT_6"] + data_df["FLAG_DOCUMENT_7"] + data_df["FLAG_DOCUMENT_8"] + data_df["FLAG_DOCUMENT_9"] + data_df["FLAG_DOCUMENT_10"] + data_df["FLAG_DOCUMENT_11"] + data_df["FLAG_DOCUMENT_12"] + data_df["FLAG_DOCUMENT_13"] + data_df["FLAG_DOCUMENT_14"] +data_df["FLAG_DOCUMENT_15"] + data_df["FLAG_DOCUMENT_16"] + data_df["FLAG_DOCUMENT_17"] + data_df["FLAG_DOCUMENT_18"] + data_df["FLAG_DOCUMENT_19"] + data_df["FLAG_DOCUMENT_20"] + data_df["FLAG_DOCUMENT_21"]

    pop_cols3 = ['FLAG_DOCUMENT_2',	'FLAG_DOCUMENT_3',	'FLAG_DOCUMENT_4',	'FLAG_DOCUMENT_5',	'FLAG_DOCUMENT_6',	'FLAG_DOCUMENT_7',	'FLAG_DOCUMENT_8',	'FLAG_DOCUMENT_9',	'FLAG_DOCUMENT_10',	'FLAG_DOCUMENT_11',	'FLAG_DOCUMENT_12',	'FLAG_DOCUMENT_13',	'FLAG_DOCUMENT_14',	'FLAG_DOCUMENT_15',	'FLAG_DOCUMENT_16',	'FLAG_DOCUMENT_17',	'FLAG_DOCUMENT_18',	'FLAG_DOCUMENT_19',	'FLAG_DOCUMENT_20',	'FLAG_DOCUMENT_21']

    data_df = data_df.drop(pop_cols3, axis=1)
    print(data_df["FLAG_DOCUMENT"].value_counts())

    data_df['AMT_REQ_CREDIT_BUREAU'] = data_df['AMT_REQ_CREDIT_BUREAU_HOUR'] + data_df['AMT_REQ_CREDIT_BUREAU_DAY'] + data_df['AMT_REQ_CREDIT_BUREAU_WEEK'] + data_df['AMT_REQ_CREDIT_BUREAU_MON'] + data_df['AMT_REQ_CREDIT_BUREAU_QRT'] + data_df['AMT_REQ_CREDIT_BUREAU_YEAR']

    pop_cols4 = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT' , 'AMT_REQ_CREDIT_BUREAU_YEAR']

    data_df = data_df.drop(pop_cols4, axis=1)

    print(data_df['AMT_REQ_CREDIT_BUREAU'].isna().sum())
    #print(data_df['NAME_TYPE_SUITE'].value_counts())
    #print(data_df['OCCUPATION_TYPE'].value_counts())
    #print(data_df['HOUSETYPE_MODE'].value_counts())
    #print(data_df['WALLSMATERIAL_MODE'].value_counts())
    #print(data_df['EMERGENCYSTATE_MODE'].value_counts())

    data_df['CNT_FAM_MEMBERS']  = data_df['CNT_FAM_MEMBERS'].fillna(2)
    data_df['AMT_REQ_CREDIT_BUREAU'] = data_df['AMT_REQ_CREDIT_BUREAU'].fillna(0)
    #data_df['AMT_REQ_CREDIT_BUREAU_HOUR'] = data_df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0)
    #data_df['AMT_REQ_CREDIT_BUREAU_DAY'] = data_df['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0)
    #data_df['AMT_REQ_CREDIT_BUREAU_WEEK'] = data_df['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(0)
    #data_df['AMT_REQ_CREDIT_BUREAU_MON'] = data_df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0)
    #data_df['AMT_REQ_CREDIT_BUREAU_QRT'] = data_df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0)
    #data_df['AMT_REQ_CREDIT_BUREAU_YEAR'] = data_df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0)


    data_df['OBS Ratio1'] = np.where(data_df['OBS_30_CNT_SOCIAL_CIRCLE'] == 0 , 0, data_df['DEF_30_CNT_SOCIAL_CIRCLE'] / data_df['OBS_30_CNT_SOCIAL_CIRCLE'])
    data_df['OBS Ratio2'] = np.where(data_df['OBS_60_CNT_SOCIAL_CIRCLE'] == 0, 0,
                                     data_df['DEF_60_CNT_SOCIAL_CIRCLE'] / data_df['OBS_60_CNT_SOCIAL_CIRCLE'])
    data_df['OBS DEF'] = (data_df['OBS Ratio1'] + data_df['OBS Ratio2']) / 2

    #print(data_df['OBS DEF'].head())
    data_df['OBS DEF'] =    data_df['OBS DEF'].fillna(0)
    #print(train_df.isna().sum())
    pop_cols5 =  ['OBS Ratio1', 'OBS Ratio2','OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']
    data_df = data_df.drop(pop_cols5, axis = 1)

    #print(data_df.dtypes.value_counts())
    #print("Numeric columns: ", num_float_cols + num_int_cols, "Object columns: ", num_object_cols)
    #test_df = data_df
    #test_df= data_df
    return data_df

def clean_NA(data_df, data_df_test):
    #data_df = train_df
    #data_df_test = test_df
    #print(data_df.shape)
    #print(data_df_test.shape)
    total_rows = data_df.shape[0]
    i = 0
    j = 0
    #print(data_df.select_dtypes(include=[np.float64]).columns.values[-2:])

    for col in data_df.select_dtypes(include=[np.float64]).columns.values:
        #print(col)
        if ( (data_df[col].isna().sum() != 0)):
            if(data_df[col].isna().sum() / total_rows <= 0.65):
                data_df[col] = data_df[col].fillna(data_df[col].mean())
                data_df_test[col] = data_df_test[col].fillna(data_df_test[col].mean())
                print(col, " ", data_df[col].isna().sum() / total_rows)
                #input()
                i += 1
            else:
                data_df.pop(col)
                data_df_test.pop(col)
                print("deleting col ", col)
                #input()
                j += 1
            # print(col, " column dropped")

    print("total values changed ", i)
    print(" total columns dropped", j)

    # data_df.pop("WEEKDAY_APPR_PROCESS_START")

    # data_df.pop("SK_ID_CURR")

    print("Total columns after cleaning ", data_df.shape[1])
    return data_df, data_df_test

def repl_catg_columns(tr_df, te_df):
    #tr_df = train_df
    #te_df = test_df
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
    for counter, col in enumerate(tr_data_types,0):
        if col == "object":
            #if tr_x != tr_y:
            print("Object data type identified ", tr_concat_df.columns[counter],  " at column ", counter)

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

            # concat_df = concat_df.po  p(df.iloc[:,counter])  # , inplace = True)
            # concat_df.drop(cols[counter])
            tr_concat_df = pd.concat((tr_concat_df, tr_dummy_df), axis=1)
            te_concat_df = pd.concat((te_concat_df, te_dummy_df), axis=1)

            # print(concat_df.shape)
    tr_concat_df = tr_concat_df.drop(tr_concat_df.columns[counter_list], axis=1)
    te_concat_df = te_concat_df.drop(te_concat_df.columns[counter_list], axis=1)
    print(tr_concat_df.shape)
    print(te_concat_df.shape)
    #train_df = tr_concat_df
    #test_df = te_concat_df
    return tr_concat_df, te_concat_df

def get_cleaned_bureau_data():
    print("Reading Bureau Files")
    bureau_df = pd.read_csv(BUREAU_FILENAME)
    print("Bureau File Records Shape ", bureau_df.shape)

    bureau_balance_df = pd.read_csv(BUREAU_BALANCE_FILENAME)
    print("Bureau Balance File Records Shape ", bureau_balance_df.shape)

    print("Cleaning the data now ")
    bureau_balance_df.loc[bureau_balance_df['STATUS'] == 'C', 'STATUS'] = '0'
    bureau_balance_df.loc[bureau_balance_df['STATUS'] == 'X', 'STATUS'] = '0'

    #bureau_balance_df['MONTHS_BALANCE'].value_counts()
    #bureau_balance_df['STATUS'].value_counts()
    #print(bureau_balance_df.isna().sum())
    bureau_balance_df['STATUS'] = bureau_balance_df['STATUS'].astype(int)
    #print(bureau_balance_df.dtypes)
    bureau_balance_df.pop("MONTHS_BALANCE")
    agg_bb_df = bureau_balance_df.groupby("SK_ID_BUREAU").agg(np.max)

    agg_bb_df = agg_bb_df.reset_index()
    print(agg_bb_df.shape)

    #agg_bb_df.to_csv(AGG_BUREAU_BALANCE_FILENAME)

    bb_df = bureau_df.merge(agg_bb_df, how='left', left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU')
    #print(bb_df.isna().sum())
    #print(bb_df.shape)
    #bb_df.to_csv(JOINED_BUREAU_FILENAME)
    #print(bb_df.dtypes)

    bb_df.pop("CREDIT_CURRENCY")
    bb_df.pop("SK_ID_BUREAU")
    bb_df.pop("CREDIT_ACTIVE")
    bb_df.pop("CREDIT_TYPE")
    bb_df.pop("AMT_ANNUITY")

    #x = bb_df.groupby("SK_ID_CURR").agg({'DAYS_CREDIT': 'mean', 'CREDIT_DAY_OVERDUE': 'max', 'DAYS_CREDIT_ENDDATE':'mean', 'DAYS_ENDDATE_FACT':'mean',
    #                                    'AMT_CREDIT_MAX_OVERDUE':'sum', 'CNT_CREDIT_PROLONG':'mean','AMT_CREDIT_SUM':'sum', 'AMT_CREDIT_SUM_LIMIT':'sum',
    #                                    'DAYS_CREDIT_UPDATE':'mean', 'STATUS': 'max'})

    x = bb_df.groupby("SK_ID_CURR").agg(
        {'DAYS_CREDIT': 'mean', 'CREDIT_DAY_OVERDUE': 'max', 'DAYS_CREDIT_ENDDATE': 'mean', 'DAYS_ENDDATE_FACT': 'mean',
         'AMT_CREDIT_MAX_OVERDUE': 'sum', 'CNT_CREDIT_PROLONG': 'mean', 'AMT_CREDIT_SUM': 'sum',
         'AMT_CREDIT_SUM_LIMIT': 'sum',
         'DAYS_CREDIT_UPDATE': 'mean', 'STATUS': ['max', 'count']})

    #x = bb_df.groupby("SK_ID_CURR").agg(np.mean)
    #print(bb_df.isna().sum())
    print(x.shape)
    x = x.reset_index()
    #print(x.columns)
    cols = ['SK_ID_CURR', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE',
     'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_LIMIT', 'DAYS_CREDIT_UPDATE', 'STATUS', 'COUNT']
    x.columns = cols
    print(x.head())

    print(bb_df.shape)
    #print(x.isna().sum())
    x['STATUS'] = x['STATUS'].fillna(0)
    x = x.fillna(0)
    print("Cleaned up beureau data set ", x.shape)
    return x