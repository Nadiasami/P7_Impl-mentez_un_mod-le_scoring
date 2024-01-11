import pandas as pd
import numpy as np
import os
import sys
import pytest 

# Obtenez le chemin absolu du fichier CSV
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Feature_importance_for_model_optimi_time', 'feature_importance_df.csv'))

# Ajouter le chemin du répertoire parent au chemin de recherche du module (si nécessaire)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Script import cleaning_feature_engineering
from cleaning_feature_engineering import remplir_valeurs_manquantes, supprimer_colonnes_manquantes, supprimer_var_correl, supprimer_colonnes_valeurs_uniques, one_hot_encoder

########################## Test function lire_et_traiter_donnees #############################################################################################################################

def test_csv_loading():
    # Détermine le chemin du fichier CSV
    df = pd.read_csv(csv_path)
    # Appliquer les lignes de code à tester
    feature_importance_0 = df[df['importance'] == 0]
    unique_features = feature_importance_0['feature'].unique()
    # Vérifie que le DataFrame n'est pas vide
    assert not df.empty, "Erreur dans le chargement du CSV."
    assert len(unique_features) == len(set(unique_features)), "La liste unique_features contient des valeurs en double."
    
########################## Test function one hot encoding #############################################################################################################################    
    
def test_one_hot_encoder():
        # Créer un DataFrame pour le test
        data = {'Category': ['A', 'B', 'A', 'C', 'B'],
                'Value': [10, 20, 15, 25, 30]}
        df = pd.DataFrame(data)

        # Appeler la fonction one_hot_encoder sur le DataFrame de test
        result  = one_hot_encoder(df)
        # Extraire le premier élément du tuple (le DataFrame encodé)
        df_encoded = result[0]

        # Vérifier que les colonnes encodées ne sont plus de type 'object'
        assert all(df_encoded[col].dtype != 'object' for col in df_encoded.columns)
    
########################### Test function remplir_valeurs_manquantes #############################################################################################################################

def test_supprimer_colonnes_valeurs_uniques():
    # Créez un petit jeu de données pour le test
    data = {
        'CODE_GENDER': ['F', np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, 'M', 'Y', np.nan],
        'FLAG_OWN_CAR': ['Y', 'N', 'Y', np.nan, np.nan, np.nan, np.nan, np.nan,'Y', 'Y'],
        'DAYS_EMPLOYED': [np.nan, np.nan, 400, 250, np.nan, np.nan, np.nan, np.nan, 350, 240],
        'DAYS_BIRTH': [12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000, 12000],
        'AMT_INCOME_TOTAL': [np.nan, np.nan, 70000, 80000, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'AMT_CREDIT': [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000],
        'CNT_FAM_MEMBERS': [2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        'TARGET': [0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    }

    df_test = pd.DataFrame(data)
    filled_df = supprimer_colonnes_valeurs_uniques(df_test)

    # Vérifiez si les valeurs manquantes ont été correctement remplies
    assert 'DAYS_BIRTH' not in filled_df
    assert 'AMT_CREDIT' not in filled_df

########################## Test function remplir_valeurs_manquantes #############################################################################################################################

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
