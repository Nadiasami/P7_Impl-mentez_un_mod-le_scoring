import joblib
import pandas as pd
import shap
from flask import Flask, jsonify, request
import os

# charger le modèle
current_directory = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_directory,"best_model_lightgbm.pkl")
model=joblib.load(model_path)

def prepare(ride):
    data = pd.DataFrame(ride)
    df_pred =data.drop(columns=data.columns[0])
    return df_pred

def predicts(df_pred):
    prediction = model.predict(df_pred)[0]          
    return prediction

def shap_values(pred_data):
    print(pred_data)

def shap_values(df_pred):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_pred)
    exp = shap.Explanation(shap_values.values[:,:,0], shap_values.base_values[:,1], data=df_pred.values, feature_names=df_pred.columns)   
    return exp

def feature_global(df_pred):
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = df_pred.columns
    feature_importance_df["importance"] = model.feature_importances_
    feature_importance_df=feature_importance_df.sort_values('importance', ascending=False)
    top_features_globales = feature_importance_df.head(40)
    return top_features_globales

def probability(df_pred):
    probability=model.predict_proba(df_pred)
    probability_class1=round(probability.tolist()[0][0],2)
    probability_class2=round(probability.tolist()[0][1],2)
    return max(probability_class1, probability_class2)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict_endpoints():
    ride = request.get_json()
    
    pred_data = prepare(ride)
    prediction = predicts(pred_data)
    exp = shap_values(pred_data)
    top_features_globales = feature_global(pred_data)
    probability_value = probability(pred_data)

    result = {
    'prediction': prediction,
    'feature_global': top_features_globales.to_dict(orient='records'),
    'probability':probability_value,
    'shap_values': {
        'values': exp.values.tolist(),
        'base_values': exp.base_values.tolist(),
        'feature_names': exp.feature_names,
    }}
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port = 9696)
    