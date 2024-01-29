import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import shap
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
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
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Importation du fichier data_clean
df = pd.read_csv('C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Data_cleaning_for_model/data_clean.csv')

Path("models").mkdir(exist_ok=True)
# Divide in training/validation and test data
    
train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
del df
gc.collect()
    
################################ LightGBM GBDT with KFold or Stratified KFold ###############################################################################################################################
      

def objective_lightgbm(params, train_df, test_df, num_folds, stratified=False, debug=False):

     
    with mlflow.start_run(run_name="modèle_lightgbm"):
        mlflow.set_tag("model", "LGBMClassifier")
        mlflow.log_params(params)
        
        best_score = 0
        best_fold = 0
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        
        # Create arrays and dataframes to store results
    
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        
        # Initialize SMOTE outside the loop
        smote = SMOTE(random_state=42)
    
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
           
            # Apply SMOTE only to the training fold
            train_x_resampled, train_y_resampled = smote.fit_resample(train_x, train_y)
    

            # Initialize LGBMClassifier
            model = LGBMClassifier(**params)

            # Training the model
            model.fit(
                train_x_resampled, train_y_resampled,
                eval_set=[(valid_x, valid_y)],
                eval_metric='auc')
                    
            
            oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
            sub_preds += model.predict_proba(test_df[feats], num_iteration=model.best_iteration_)[:, 1] / folds.n_splits
        
        
            auc = roc_auc_score(train_df['TARGET'], oof_preds) 
            fpr, tpr, thresholds = roc_curve(train_df['TARGET'], oof_preds)
        
            # Choix du seuil en fonction des FPR et TPR
            seuil_optimal = thresholds[np.argmax(tpr - fpr)] # 
        
            # Transformation des probabilités prédites en classes binaires en utilisant le seuil sélectionné
            y_pred = np.where(oof_preds > seuil_optimal, 1, 0) # supervisé probalilité sup à seuil 1    
            f1 = f1_score(train_df['TARGET'], y_pred)
            
            if auc > best_score:
                best_score = auc
                best_fold = n_fold
                best_train_x = train_x_resampled
                best_train_y = train_y_resampled
                best_valid_x = valid_x
                best_valid_y = valid_y
                                  
    return {"loss": auc, "f1_score": f1, 'best_fold': best_fold, "best_train_x": best_train_x, "best_train_y": best_train_y, "best_valid_x": best_valid_x, "best_valid_y": best_valid_y, "status": STATUS_OK} 

#################### Function_to_find_best_parameters_and_best_fold ######################################################################################################################################### 

def found_best_model_lightgbm(train_df, test_df):
   
    search_space = {
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
    
    # Saving best results
    best_params = fmin(
        fn=lambda params: objective_lightgbm(params, train_df, test_df, 10, stratified=False, debug=False),
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )
    
    return best_params

#################### Function_to_train_best_model ##########################################################################################################################################################

def train_best_model_lightgbm(
    train_df, 
    test_df, 
    best_params: dict, 
 
    
) -> None:

    resultat_objective_lightgbm = objective_lightgbm(best_params, train_df, test_df,10, stratified=False, debug=False)

    with mlflow.start_run():
        mlflow.log_params(best_params)
        
        # Create arrays and dataframes to store results
        feature_importance_df = pd.DataFrame()
        feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
        # Use best_train_x_resampled and best_train_y_resampled
        train_x = resultat_objective_lightgbm['best_train_x']
        train_y = resultat_objective_lightgbm['best_train_y']
        valid_x = resultat_objective_lightgbm['best_valid_x']
        valid_y = resultat_objective_lightgbm['best_valid_y']

        # Initialize LGBMClassifier
        model_lgbm = LGBMClassifier(**best_params)

        # Training the model
        model_lgbm.fit(
            train_x, train_y,
            eval_set=[(valid_x, valid_y)],
            eval_metric='auc')
        
        # Feature local importance graphique
        explainer = shap.TreeExplainer(model_lgbm)
        shap_values = explainer(valid_x)
        exp = shap.Explanation(shap_values.values[:,:,0], shap_values.base_values[:,1], data=valid_x.values, feature_names=valid_x.columns)
        shap.plots.waterfall(exp[0], max_display=30, show=True)
        plt.title('Lightgbm Local Features') 
        plt.tight_layout()  
        plt.savefig("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_local_importance.png")
        # Enregistrement du graphique SHAP en tant qu'artefact dans MLflow
        mlflow.log_artifact("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_local_importance.png","local_feature")
        plt.close()
        
        # Feature global importance csv
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = model_lgbm.feature_importances_
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        feature_importance_df.to_csv('feature_importances_global_lightgbm.csv', index=False)
        mlflow.log_artifact('feature_importances_global_lightgbm.csv', 'feature_importances')
        
        # Feature global importance graphique
        cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
        plt.figure(figsize=(12, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('Lightgbm Features avg')
        plt.tight_layout()
        plt.savefig("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_global_importance.png")
        # Enregistrement du graphique SHAP en tant qu'artefact dans MLflow
        mlflow.log_artifact("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_lightgbm/feature_global_importance.png","global_feature")
        plt.close()

    # Call metrique function to calculate metrique
    metriques_function(valid_x, valid_y, model_lgbm)
    mlflow.sklearn.log_model(model_lgbm, "lightgbm")
    
    return model_lgbm, train_x, train_y, valid_x, valid_y

mlflow.end_run()

################################ XGBOOST with KFold or Stratified KFold #####################################################################################################################################
      
def objective_xgboost(params, train_df, test_df, num_folds, stratified=False, debug=False):
     
    with mlflow.start_run(run_name="modèle_xgboost", nested=True):
        mlflow.set_tag("model", "XGBoost")
        mlflow.log_params(params)
        
        best_score = 0
        best_fold = 0
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        
        # Create arrays and dataframes to store results
    
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        
        # Initialize SMOTE outside the loop
        smote = SMOTE(random_state=42)
    
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
           
            # Apply SMOTE only to the training fold
            train_x_resampled, train_y_resampled = smote.fit_resample(train_x, train_y)

            # Initialize LGBMClassifier
            model = XGBClassifier(**params)

            # Training the model
            model.fit(
                train_x_resampled, train_y_resampled,
                eval_set=[(valid_x, valid_y)],
                eval_metric='auc')
                    
            oof_preds[valid_idx] = model.predict_proba(valid_x)[:, 1]
            sub_preds += model.predict_proba(test_df[feats])[:, 1] / folds.n_splits
        
            auc = roc_auc_score(train_df['TARGET'], oof_preds) 
            fpr, tpr, thresholds = roc_curve(train_df['TARGET'], oof_preds)
        
            # Choix du seuil en fonction des FPR et TPR
            seuil_optimal = thresholds[np.argmax(tpr - fpr)] # 
        
            # Transformation des probabilités prédites en classes binaires en utilisant le seuil sélectionné
            y_pred = np.where(oof_preds > seuil_optimal, 1, 0) # supervisé probalilité sup à seuil 1    
            f1 = f1_score(train_df['TARGET'], y_pred)
            
            if auc > best_score:
                best_score = auc
                best_fold = n_fold
                best_train_x = train_x_resampled
                best_train_y = train_y_resampled
                best_valid_x = valid_x
                best_valid_y = valid_y
                                  
    return {"loss": auc, "f1_score": f1, 'best_fold': best_fold, "best_train_x": best_train_x, "best_train_y": best_train_y, "best_valid_x": best_valid_x, "best_valid_y": best_valid_y, "status": STATUS_OK} 

#################### Function_to_find_best_parameters_and_best_fold #########################################################################################################################################  

def found_best_model_xgboost(train_df, test_df):
   
    search_space = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 200, 500, 1000]),
    "missing" : np.nan,
    "max_depth": hp.randint("max_depth", 5, 15),
    "learning_rate": hp.choice("learning_rate", [0.5, 1.0]),
    "scale_pos_weight": hp.uniform("scale_pos_weight", 1, 20),
    "objective": "binary:logistic",
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "n_jobs": hp.choice("n_jobs", [1, 2, 4])
  }
    
    # Saving best results
    best_params = fmin(
        fn=lambda params: objective_xgboost(params, train_df, test_df, 10, stratified=False, debug=False),
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )
    
    return best_params

#################### Function_to_train_best_model ###########################################################################################################################################################

def train_best_model_xgboost(
    train_df, 
    test_df, 
    best_params: dict, 
 
) -> None:

    resultat_objective_xgboost = objective_xgboost(best_params, train_df, test_df,10, stratified=False, debug=False)

    with mlflow.start_run(nested=True):
        mlflow.log_params(best_params)
        
        # Create arrays and dataframes to store results
    
        feature_importance_df = pd.DataFrame()
        feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
        # Use best_train_x_resampled and best_train_y_resampled
        train_x = resultat_objective_xgboost['best_train_x']
        train_y = resultat_objective_xgboost['best_train_y']
        valid_x = resultat_objective_xgboost['best_valid_x']
        valid_y = resultat_objective_xgboost['best_valid_y']

        # Initialize LXGBClassifier
        model_xgboost = XGBClassifier(**best_params)

        # Training the model
        model_xgboost.fit(
            train_x, train_y,
            eval_set=[(valid_x, valid_y)],
            eval_metric='auc')
              
        # Feature local importance graphique
        explainer = shap.Explainer(model_xgboost)
        shap_values = explainer(valid_x)
        shap.plots.waterfall(shap_values[0], max_display=30,show=False)
        plt.title('XGBoost Local Features') 
        plt.tight_layout()    
        plt.savefig("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_xgboost/feature_local_importance.png")
        # Enregistrement du graphique SHAP en tant qu'artefact dans MLflow
        mlflow.log_artifact("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_xgboost/feature_local_importance.png","local_feature")
        plt.close()
        
        # Feature global importance csv
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = model_xgboost.feature_importances_
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        feature_importance_df.to_csv('feature_importances_global_xgboost.csv', index=False)
        mlflow.log_artifact('feature_importances_global_xgboost.csv', 'feature_importances')
        
        # Feature global importance graphique
        cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
        plt.figure(figsize=(12, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('XGBoost Features avg')
        plt.tight_layout()
        plt.savefig("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_xgboost/feature_global_importance.png")
        # Enregistrement du graphique SHAP en tant qu'artefact dans MLflow
        mlflow.log_artifact("C:/Users/smart asus/P7_données/code_vs_code/analyse_nettoyage_experiences/Shap_xgboost/feature_global_importance.png","global_feature")
        plt.close()

    # Call metrique function to calculate metrique
    metriques_function(valid_x, valid_y, model_xgboost)
    mlflow.sklearn.log_model(model_xgboost, "xgboost")
    
    return model_xgboost, train_x, train_y, valid_x, valid_y

#################### Function_calcul_métriques ##############################################################################################################################################################

def metriques_function(
    valid_x,
    valid_y, 
    model):
    
    preds = model.predict_proba(valid_x)[:,1]     
    auc = roc_auc_score(valid_y, preds)      
    fpr, tpr, thresholds = roc_curve(valid_y, preds)
        
    # Choix du seuil en fonction des FPR et TPR
    seuil_optimal = thresholds[np.argmax(tpr - fpr)] # to do imbalanced data avec smote dans analyse exporatoire
        
    # Transformation des probabilités prédites en classes binaires en utilisant le seuil sélectionné
    y_pred = np.where(preds > seuil_optimal, 1, 0) # supervisé probalilité sup à seuil 1 
    f1 = f1_score(valid_y, y_pred) 
    accuracy = accuracy_score(valid_y, y_pred)

    # Calcul des faux positifs (FP) et des faux négatifs (FN) avec le seuil sélectionné
    FP = np.sum((valid_y == 0) & (y_pred == 1))  # Faux positifs
    FN = np.sum((valid_y == 1) & (y_pred == 0))  # Faux négatifs

    # Coûts associés
    cost_FP = 1  # Coût d'un faux positif
    cost_FN = 10  # Coût d'un faux négatif (10 fois le coût d'un faux positif)

    # Calcul du score métier
    score_metier = (cost_FP * FP + cost_FN * FN) / (FP + FN + 1e-7) # divise par le total (FP + FN + 1e-7 pour ne pas afficher un score de 0 pour le score)
    
    # Enregistrement de la métrique AUC et le seuil optimal
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("Optimal Threshold", seuil_optimal) 
    mlflow.log_metric("Score métier", score_metier)
    
mlflow.end_run()
   
   
################## RUN ALL MODEL ############################################################################################################################################################################ 
    

def main_flow() -> None:
    """The main training pipeline"""
    
    mlflow.set_experiment("best_model")
    
    # Found best params lightgbm
    best_params_lightgbm = found_best_model_lightgbm(train_df, test_df)
    
    # train model lightgbm
    train_best_model_lightgbm(train_df, test_df, best_params_lightgbm)
    
    # Found best params lightgbm
    best_params_xgboost = found_best_model_xgboost(train_df, test_df)
    
    # train model lightgbm
    train_best_model_xgboost(train_df, test_df, best_params_xgboost)
    
if __name__ == "__main__":
    main_flow()