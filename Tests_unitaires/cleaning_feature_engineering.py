import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def lire_et_traiter_donnees():
    # Importation des features importances du script de base
    feature_importance_df = pd.read_csv('C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Feature_importance_for_model_optimi_time/feature_importance_df.csv')
    feature_importance_0 = feature_importance_df[feature_importance_df['importance']==0]
    feature_importance_0_list = feature_importance_0['feature'].unique()
    return feature_importance_0_list

# Fonction qui supprime les colonnes qui ont une valeur unique dans leurs rangs
def supprimer_colonnes_valeurs_uniques(df):
    unique_counts = df.nunique()
    single_value_cols = unique_counts[unique_counts == 1].index.tolist()
    df = df.drop(single_value_cols, axis=1)
    return df

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('C:/Users/smart asus/Desktop/sauvgarde mohamed/ordinateur/Formation data scientist/7ème_projet/livrable/données_étude/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('C:/Users/smart asus/Desktop/sauvgarde mohamed/ordinateur/Formation data scientist/7ème_projet/livrable/données_étude/application_test.csv', nrows= num_rows)
    df = supprimer_colonnes_valeurs_uniques(df)
    test_df = supprimer_colonnes_valeurs_uniques(test_df)
    
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df._append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('C:/Users/smart asus/Desktop/sauvgarde mohamed/ordinateur/Formation data scientist/7ème_projet/livrable/données_étude/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('C:/Users/smart asus/Desktop/sauvgarde mohamed/ordinateur/Formation data scientist/7ème_projet/livrable/données_étude/bureau_balance.csv', nrows = num_rows)
    bureau = supprimer_colonnes_valeurs_uniques(bureau)
    bb = supprimer_colonnes_valeurs_uniques(bb)
        
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'], 
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('C:/Users/smart asus/Desktop/sauvgarde mohamed/ordinateur/Formation data scientist/7ème_projet/livrable/données_étude/previous_application.csv', nrows = num_rows)
    prev = supprimer_colonnes_valeurs_uniques(prev)
    
    # Optional: Remove 4 applications with XNA NAME_CONTRACT_TYPE (train set)
    prev = prev[prev['NAME_CONTRACT_TYPE'] != 'XNA'] 
    prev = prev[prev['NAME_CLIENT_TYPE'] != 'XNA'] 
        
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
               
    # Days 365.243 values -> nan
    prev['SELLERPLACE_AREA'].replace(4000000, np.nan, inplace= True)
    
    prev['DAYS_FIRST_DRAWING_ANOM'] = prev["DAYS_FIRST_DRAWING"] == 365243
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    
    prev['DAYS_FIRST_DUE_ANOM'] = prev["DAYS_FIRST_DUE"] == 365243
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    
    prev['DAYS_LAST_DUE_1ST_VERSION_ANOM'] = prev["DAYS_LAST_DUE_1ST_VERSION"] == 365243
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    
    prev['DAYS_LAST_DUE_ANOM'] = prev["DAYS_LAST_DUE"] == 365243
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    
    prev['DAYS_TERMINATION_ANOM'] = prev["DAYS_TERMINATION"] == 365243
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('C:/Users/smart asus/Desktop/sauvgarde mohamed/ordinateur/Formation data scientist/7ème_projet/livrable/données_étude/POS_CASH_balance.csv', nrows = num_rows)
    pos = supprimer_colonnes_valeurs_uniques(pos)
    # Optional: Remove 4 applications with XNA NAME_CONTRACT_STATUS (train set)
    pos = pos[pos['NAME_CONTRACT_STATUS'] != 'XNA']    
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('C:/Users/smart asus/Desktop/sauvgarde mohamed/ordinateur/Formation data scientist/7ème_projet/livrable/données_étude/installments_payments.csv', nrows = num_rows)
    ins = supprimer_colonnes_valeurs_uniques(ins)
        
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('C:/Users/smart asus/Desktop/sauvgarde mohamed/ordinateur/Formation data scientist/7ème_projet/livrable/données_étude/credit_card_balance.csv', nrows = num_rows)
    cc = supprimer_colonnes_valeurs_uniques(cc)
        
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# Fonction qui supprime les colonnes dont les valeurs manquantes dépassent un thresold
def supprimer_colonnes_manquantes(df, seuil=0.75):
    colonnes_a_supprimer = []
    for col in df.columns:
        if (df[col].isna().sum() / len(df)) > seuil:
            colonnes_a_supprimer.append(col)
    df_sans_manquant = df.drop(columns=colonnes_a_supprimer)
    gc.collect()
    return df_sans_manquant

# Fonction qui remplit les valeurs manquantes numérique par la médiane et catégorielles par la plus courante
def remplir_valeurs_manquantes(df):
    for col in df.columns:
        if df[col].dtype.kind == 'f' and col != 'TARGET':
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].value_counts().index[0])
    gc.collect()
    return df

# Fonction qui remplit supprime les variables corrélées entre elles
def supprimer_var_correl(df, threshold=0.9):
    df_num = df.select_dtypes(include=['number'])
    # Matrice de corrélation des valeurs absolues
    corr_matrix = df_num.corr().abs()
    # Sélection de la partie supérieure de la matrice de corrélation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Colonnes à supprimer
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Suppression des colonnes du dataframe
    df_sans_colonnes_correlees = df.drop(to_drop, axis=1)
    gc.collect()
    return df_sans_colonnes_correlees
    

def feature_importances(df, debug= False):
    feature_importance_0_list = supprimer_colonnes_valeurs_uniques()
    for col in df.columns:
        if col in feature_importance_0_list:
            df = df.drop(columns=[col])
    df = df.rename(columns=lambda x: x.replace(',', '_'))
    df = df.rename(columns=lambda x: x.replace(':', '_'))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    gc.collect()
    return df
    

def mafonction(debug = False):
    num_rows = 200 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
        
    with timer("Process supprimer_colonnes_manquantes"):
        df = supprimer_colonnes_manquantes(df)
        gc.collect()
    with timer("Process remplir_valeurs_manquantes"):
        df = remplir_valeurs_manquantes(df)
        gc.collect()
    with timer("Process supprimer_var_correl"):
        df = supprimer_var_correl(df)
        gc.collect()
    with timer("Process feature_importances"):
        df = feature_importances(df, debug= debug)
        df.to_csv("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Data_cleaning_for_model/data_clean_2.csv", index=False)
        gc.collect()
    
                
if __name__ == "__main__":
    with timer("Full model run"):
        mafonction()