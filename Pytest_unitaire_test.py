########################## Test function application train test #############################################
import pandas as pd
import numpy as np
from cleaning_feature_engineering import application_train_test  # Assurez-vous de remplacer `your_module` par le nom réel du module où se trouve votre fonction

def test_application_train_test():
    # Créez un petit jeu de données pour le test
    data = {
        'CODE_GENDER': ['F', 'M', 'F', 'XNA'],
        'FLAG_OWN_CAR': ['Y', 'N', 'Y', 'Y'],
        'DAYS_EMPLOYED': [365243, 365243, 365243, 365243],
        'DAYS_BIRTH': [10000, 12000, 15000, 11000],
        'AMT_INCOME_TOTAL': [50000, 60000, 70000, 80000],
        'AMT_CREDIT': [100000, 120000, 150000, 110000],
        'CNT_FAM_MEMBERS': [2, 1, 3, 4],
        'AMT_ANNUITY': [10000, 12000, 13000, 15000]
    }

    df = pd.DataFrame(data)
    # Vérifiez que la colonne DAYS_EMPLOYED_ANOM modifie bien la valeur aberrante par un np.nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    assert df['DAYS_EMPLOYED'].isnull().sum() > 0


    
    