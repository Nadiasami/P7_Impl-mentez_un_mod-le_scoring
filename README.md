# P7_Implementez_un_modele_scoring:

* Data:

-Le jeu de données provient de la compétition Kaggle. Le prétraitement est basé sur le prétraitement de cette compétition.
Quelques étapes de prétraitement supplémentaires ont été effectuées, en se basant sur une analyse exploratoire qu'on peut retrouver dans le notebook analyse_exploratoire.ipynb enregistré dans le dossier Notebook_help_for_preprocessing, Le processus est détaillé autant que possible à l'aide du script retrouvé dans le dossier Script_preprocessong nommé cleaning_feature_engineering.py

* Classes déséquilibrées :

Les clients ayant remboursés leurs prêts sont présents à presque 90% dans nos données, alors que seulement 10% de clients qui n’ont pas remboursé leurs prêts sont présents, voici le graphique suivant, la valeur de défaut de paiment est 1, alors que 0 représente un client sovable.
 
Pour résoudre cette problématique on a opté à l’utilisation de la méthode smote (Synthetic Minority Over-sampling Technique) qui est une technique utilisée pour traiter le problème de déséquilibre de classes dans un ensemble de données. Elle vise spécifiquement à résoudre le déséquilibre entre les classes minoritaires et majoritaires en générant synthétiquement des exemples de la classe minoritaire.

* Modélisations :

Deux modèles ont été testés pour cette étude, il s’agit du modèle xgboostclassifier et lightgbmclassifier, en appliquant une méthode semblable à la validation croisée qui est l’instauration d’une boucle itératif pour retrouver le meilleur fold qui nous donne le meilleur modèle, ainsi que l’application de la méthode hyperopt pour la recherche des meilleures hyper-paramètres de chaque modèle.
La modélisation a été réalisée en deux parties, dans un premier temps la comparaison de nos deux modèles tout en évaluant ces derniers sur la métrique auc, en second lieu on récupère le meilleur des deux modèles et on l’entraine sur un score métier demandé par l’enseigne afin de minimiser ce dernier.
Le script best_model_lightgbm_xgboost_eval_auc.py dans le dossier Scripts_models est celui de la première partie alors que best_lightgbm_eval_score_metier.py est celui mis en production pour réaliser notre application.

* Data Drift des données :

"Évidently" est utilisé pour calculer la dérive des données en supposant que l'ensemble d'entraînement est constitué de données connues application_train, tandis que l'application_test représente les données actuelles. La vérification croisée entre les caractéristiques d'importance élevée et les variables fortement décalées montre qu’on a 9 variables sur 120 qui présnent une dérive à plus de 50 % - on pourrait alors dire qu’il n’y a pas de dérive de données actuellement, mais une surveillance constante est recommandée cependant. Le rapport au format HTML est affiché dans un dossier nommé Data drift.

