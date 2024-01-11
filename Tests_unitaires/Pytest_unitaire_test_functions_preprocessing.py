########################## Test function remplir_valeurs_manquantes ################################################################################################################################
import pandas as pd
import numpy as np
<<<<<<< HEAD
from cleaning_feature_engineering import remplir_valeurs_manquantes, supprimer_colonnes_manquantes, supprimer_var_correl  
import pytest
=======
import pytest 

from cleaning_feature_engineering import remplir_valeurs_manquantes, supprimer_colonnes_manquantes, supprimer_var_correl  
>>>>>>> da85c28567654ac37b573e527a96b75af22eeec8

def test_remplir_valeurs_manquantes():
    # Créez un petit jeu de données pour le test
    data = {
        'CODE_GENDER': ['F', np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, 'M', 'Y', np.nan],
        'FLAG_OWN_CAR': ['Y', 'N', 'Y', np.nan, np.nan, np.nan, np.nan, np.nan,'Y', 'Y'],
        'DAYS_EMPLOYED': [np.nan, np.nan, 400, 250, np.nan, np.nan, np.nan, np.nan, 350, 240],
        'DAYS_BIRTH': [10000, 12000, 15000, 11000, 14000, 15000, 20000, 18000, 16500, 12500],
        'AMT_INCOME_TOTAL': [np.nan, np.nan, 70000, 80000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'AMT_CREDIT': [100000, 120000, 150000, 110000, 110000, 120000, 180000, 170000, 160000, 150000],
        'CNT_FAM_MEMBERS': [2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        'TARGET': [0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    }

    df_test = pd.DataFrame(data)
    filled_df = remplir_valeurs_manquantes(df_test)

    # Vérifiez si les valeurs manquantes ont été correctement remplies
    assert pd.isnull(filled_df['CODE_GENDER']).sum() == 0  # Vérifiez si les valeurs manquantes pour 'CODE_GENDER' ont été remplies
    assert pd.isnull(filled_df['FLAG_OWN_CAR']).sum() == 0  # Vérifiez si les valeurs manquantes pour 'FLAG_OWN_CAR' ont été remplies
    assert pd.isnull(filled_df['DAYS_EMPLOYED']).sum() == 0  # Vérifiez si les valeurs manquantes pour 'FLAG_OWN_CAR' ont été remplies
    assert pd.isnull(filled_df['AMT_INCOME_TOTAL']).sum() == 0  # Vérifiez si les valeurs manquantes pour 'FLAG_OWN_CAR' ont été remplies
    
########################## Test function supprimer colonnes manquantes #############################################################################################################################

def test_supprimer_colonnes_manquantes():
    # Créez un petit jeu de données pour le test
    data = {
        'CODE_GENDER': [np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, 'M', np.nan, np.nan],
        'FLAG_OWN_CAR': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,'Y', 'Y'],
        'DAYS_EMPLOYED': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 350, 240],
        'DAYS_BIRTH': [10000, 12000, 15000, 11000, 14000, 15000, 20000, 18000, 16500, 12500],
        'AMT_INCOME_TOTAL': [np.nan, np.nan, 70000, 80000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'AMT_CREDIT': [100000, 120000, 150000, 110000, 110000, 120000, 180000, 170000, 160000, 150000],
        'CNT_FAM_MEMBERS': [2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        'TARGET': [0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    }

    df_test = pd.DataFrame(data)
    del_col = supprimer_colonnes_manquantes(df_test, 0.75)

    # Vérifiez si les colonnes ont été supprimées correctement
    assert 'CODE_GENDER' not in del_col
    assert 'FLAG_OWN_CAR' not in del_col
    assert 'DAYS_EMPLOYED' not in del_col
    
########################## Test function upprimer variables corrélées ##############################################################################################################################
    
def test_supprimer_var_correl():
    # Créez un petit jeu de données pour le test
    data = {
        'CODE_GENDER': [200000, 240000, 300000, 220000, 220000, 240000, 360000, 340000, 320000, 300000],
        'FLAG_OWN_CAR': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,'Y', 'Y'],
        'DAYS_EMPLOYED': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 350, 240],
        'DAYS_BIRTH': [200000, 240000, 300000, 220000, 220000, 240000, 360000, 340000, 320000, 300000],
        'AMT_INCOME_TOTAL': [np.nan, np.nan, 70000, 80000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'AMT_CREDIT': [100000, 120000, 150000, 110000, 110000, 120000, 180000, 170000, 160000, 150000],
        'CNT_FAM_MEMBERS': [2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        'TARGET': [0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    }
    
    df_test = pd.DataFrame(data)
    # Utilisez la fonction supprimer_var_correl pour supprimer les colonnes corrélées
    df_sans_colonnes_correlees = supprimer_var_correl(df_test, threshold=0.5)

    # Vérifiez si les colonnes ont été supprimées correctement
    assert 'DAYS_BIRTH' not in df_sans_colonnes_correlees.columns
    assert 'CODE_GENDER' in df_sans_colonnes_correlees.columns
    
