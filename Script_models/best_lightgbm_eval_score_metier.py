import numpy as np
import pandas as pd
import csv
import matplotlib
matplotlib.use('Agg')
import shap
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mlflow
from pathlib import Path
from imblearn.over_sampling import SMOTE
import scipy
import sklearn
import mlflow.sklearn
from xgboost import XGBClassifier
import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.pyll import scope
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)

#####################################################################################################################################################################################
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

#####################################################################################################################################################################################
# Importation du fichier data_clean
def importation_data():
    df = pd.read_csv('C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Data_cleaning_for_model/data_clean_1.csv')

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    
    # Exporter le fichier data_test 
    test_data=test_df.iloc[:, 3:]
    test_data.to_csv('C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Data_for_test/test_data.csv')
    del df
    return train_df, test_df
    gc.collect()
#####################################################################################################################################################################################
# Fonction qui calcule le score métier
def score_metier_obj(y_true, y_pred, optimal_threshold):
    
    predictions_binaires = (y_pred > optimal_threshold).astype(int)

    FP = np.sum((predictions_binaires == 1) & (y_true == 0))
    FN = np.sum((predictions_binaires == 0) & (y_true == 1))

    # Coûts associés
    cost_FP = 1  # Coût d'un faux positif
    cost_FN = 10  # Coût d'un faux négatif (10 fois le coût d'un faux positif)

    # Calcul du score métier
    score = (cost_FP * FP + cost_FN * FN) / (FP + FN + 1e-7)

    return score
#####################################################################################################################################################################################    
# LightGBM with KFold or Stratified KFold
def objective_lightgbm(params, train_df, num_folds, stratified=False, debug=False):
     
    with mlflow.start_run(run_name="modèle_lightgbm"):
        mlflow.set_tag("model", "LGBMClassifier")
        mlflow.log_params(params)
        
        # Initialisation de best_score, best_model, best_threshold et best_fold_data 
        best_score = float('inf')
        best_model = None
        best_threshold = None
        best_fold_data = None
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        
        # Creation de dataframe pour stocker les résultats
        oof_preds = np.zeros(train_df.shape[0])
        feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        
        # Initialisation de smote en dehors de la boucle
        smote = SMOTE(random_state=42)
    
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
           
            # Application de smote uniquement sur les données d'entrainement
            train_x_resampled, train_y_resampled = smote.fit_resample(train_x, train_y)

            # Obtenez le seuil optimal 
            optimal_threshold = params['threshold']
            
            # Créez une fonction make_scorer pour notre score métier
            score_metier_scorer = make_scorer(score_metier_obj,  greater_is_better=False, threshold=optimal_threshold)

            # Initialisation du modèle LGBMClassifier
            model = LGBMClassifier(**params)
            
            # Training the model
            model.fit(
                train_x_resampled, train_y_resampled,
                eval_set=[(valid_x, valid_y)],
                eval_metric=[('score_metier', score_metier_scorer)])
            
            # Réaliser les prédictions sur les données validations
            oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]

            # Calculez votre métrique avec le seuil optimal
            score_metier = score_metier_obj(valid_y, oof_preds[valid_idx], optimal_threshold)
            
            # Si le score actuel est meilleur que le meilleur score précédent, mettez à jour les variables
            if score_metier < best_score:
                best_score = score_metier
                best_model = model
                best_threshold = optimal_threshold
                best_fold_data = {
                    'train_x': train_x_resampled,
                    'train_y': train_y_resampled,
                    'valid_x': valid_x,
                    'valid_y': valid_y
                     }
            # Enregistrez le meilleur modèle avec mlflow
        mlflow.sklearn.log_model(model, "best_model")
                                
    return {"loss": best_score, "status": STATUS_OK, "best_model": best_model, "best_threshold": best_threshold, "best_fold_data": best_fold_data} 
######################################################################################################################################################################################
# Définition de l'intervalle de nos paramètres
def found_best_model_lightgbm(train_df):  
      
    search_space = {
    "threshold": hp.uniform("threshold", 0.4, 0.6),
    "nthread": 4,
    "n_estimators": 10000,
    "learning_rate": 1,  
    "num_leaves": hp.randint("num_leaves", 20, 31),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "max_depth": hp.randint("max_depth", 5, 15),
    "reg_alpha": hp.loguniform("reg_alpha", -4, 0),  
    "reg_lambda": hp.loguniform("reg_lambda", -4, 0),  
    "min_split_gain": hp.uniform("min_split_gain", 0.01, 0.1),
    "min_child_weight": hp.uniform("min_child_weight", 20, 60), 
    "early_stopping_rounds": 200,
    "verbosity": 1,}
    
    # Sauvegarde de nos meilleures paramètres dans best_params
    best_params = fmin(
        fn=lambda params: objective_lightgbm(params, train_df, 10, stratified=False, debug=False),
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )
          
    return best_params
######################################################################################################################################################################################
# Fonction pour calculer les meilleures métriques avec les meilleures paramètres
def best_model_lightgbm(
    train_df, 
    best_params: dict
       
) -> None:

    # Récupération du meilleur modèle et ses paramètres
    resultat_objective = objective_lightgbm(best_params, train_df, 10, stratified=False, debug=False)
    
    best_model = resultat_objective['best_model']
    best_threshold = resultat_objective['best_threshold']
    best_fold_data = resultat_objective['best_fold_data']
    
    # Importation des meilleures données csv
    train_x  = best_fold_data['train_x']
    train_y  = best_fold_data['train_y']
    valid_x  = best_fold_data['valid_x']
    valid_y  = best_fold_data['valid_y']
    
    train_x.to_csv('train_x.csv')
    train_y.to_csv('train_y.csv')
    valid_x.to_csv('valid_x.csv')
    valid_y.to_csv('valid_y.csv')
 
        
    # Calcul des prédictions des probabilités
    valid_predictions = best_model.predict_proba(valid_x, num_iteration=best_model.best_iteration_)[:, 1]                 
    
    # Calcul des prédictions binaires à l'aide des probabilités, le threshold optimal est utilisé pour la comparaison
    y_pred = np.where(valid_predictions > best_threshold, 0, 1)  
    
    # Calcul de la métrique AUC
    auc = roc_auc_score(best_fold_data['valid_y'], valid_predictions)      
    mlflow.log_metric("AUC", auc)
    
    fpr, tpr, thresholds = roc_curve(valid_y, valid_predictions)

    # Tracer la courbe ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Enregistrer la figure localement
    plt.savefig("roc_curve.png")

    # Enregistrer la figure en tant qu'artefact dans MLflow
    mlflow.log_artifact("roc_curve.png")


    # Calucl de la métrique précision
    accuracy = accuracy_score(best_fold_data['valid_y'], y_pred)
    mlflow.log_metric("accuracy", accuracy)  
    
    # Calcul de la métrique score métier
    custom_metier = score_metier_obj(best_fold_data['valid_y'], valid_predictions, best_threshold) 
    mlflow.log_metric("custom_metier", custom_metier)
    
    # Enregistrement du meilleur seuil optimal
    mlflow.log_metric("Optimal Threshold", best_threshold)           
    
    # Enregistrement du nom du modèle
    mlflow.sklearn.log_model(best_model, "LGBM_model")
             
    # Feature local importance graphique
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(best_fold_data['valid_x'])
    exp = shap.Explanation(shap_values.values[:,:,0], shap_values.base_values[:,1], data=best_fold_data['valid_x'].values, feature_names=best_fold_data['valid_x'].columns)
    shap.plots.waterfall(exp[0], max_display=30, show=True)
    plt.title('Lightgbm Local Features') 
    plt.tight_layout()  
    plt.savefig("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_local_importance.png")
    
    # Enregistrement du graphique SHAP en tant qu'artefact dans MLflow
    mlflow.log_artifact("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_local_importance.png","local_feature")
    plt.close()
        
    # Feature global importance csv
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = best_fold_data['valid_x'].columns
    feature_importance_df["importance"] = best_model.feature_importances_
    feature_importance_df.to_csv('C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_importances_global_lightgbm.csv', index=False)
    mlflow.log_artifact('C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_importances_global_lightgbm.csv', 'feature_importances')
        
    # Feature global importance graphique
    feature_importance_df_tri = feature_importance_df.sort_values("importance", ascending=False)
    top_features_globales = feature_importance_df_tri.head(40)
    plt.figure(figsize=(12, 10))
    sns.barplot(x="importance", y="feature", data=top_features_globales)
    plt.title('Lightgbm Features')
    plt.tight_layout()
    plt.savefig("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_global_importance.png")
    
    # Enregistrement du graphique SHAP en tant qu'artefact dans MLflow
    mlflow.log_artifact("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_global_importance.png","global_feature")
    plt.close()

    return {"AUC": auc, "accuracy": accuracy, "custom_metier": custom_metier , "status": STATUS_OK}    
##############################################################################################################################################################################################
# Lancement de notre script
def main_flow() -> None:
    """The main training pipeline"""
    
    mlflow.set_experiment("best_modele_lightgbm_test_courbe_roc")
    
    # Importation data
    train_df, test_df = importation_data()
    
    # Found best params lightgbm
    best_params_lightgbm = found_best_model_lightgbm(train_df)
    
    # train model lightgbm
    best_model_lightgbm(train_df, best_params_lightgbm)
    
    
if __name__ == "__main__":
    main_flow()