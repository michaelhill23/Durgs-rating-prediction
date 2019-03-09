import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import time
import warnings; warnings.simplefilter('ignore')


df1 = pd.read_csv('train_output.csv', sep=';')
df2 = pd.read_csv('test_output.csv', sep=';')

X_train = df1[['effectiveness', 'sideEffects', 'urlDrugName_group', 'condition_group',
               'benefitsReview_len', 'sideEffectsReview_len', 'commentsReview_len',
               'benefitsReview_subjectivity', 'sideEffectsReview_polarity']]
y_train = df1[['rating']]

X_test = df2[['effectiveness', 'sideEffects', 'urlDrugName_group', 'condition_group',
              'benefitsReview_len', 'sideEffectsReview_len', 'commentsReview_len',
              'benefitsReview_subjectivity', 'sideEffectsReview_polarity']]
y_test = df2[['rating']]

def select_model(X, Y):
    best_models = {}
    models = [
        {
            'name': 'LogisticRegression',
            'estimator': LogisticRegression(),
            'hyperparameters': {
                'solver': ['newton-cg', 'sag', 'lbfgs'],
                'C': [0.001, 0.01, 0.1, 1, 10],
                'max_iter': [10, 100, 150]
            },
        },
        {
            'name': 'RandomForest',
            'estimator': RandomForestClassifier(),
            'hyperparameters': {
                'bootstrap': ['True'],
                'criterion': ['entropy'],
                'max_features': [0.4, 0.5, 0.55, 0.6],
                'min_samples_leaf': [5, 10, 20],
                'min_samples_split': [2, 6, 10],
                'n_estimators': [20, 50, 100]
            }
        },

        {
            'name': 'GaussianNB',
            'estimator': GaussianNB(),
            'hyperparameters': {}
        },
        {
            'name': 'XGBoost',
            'estimator': XGBClassifier(),
            'hyperparameters': {
                'learning_rate': [0.01, 0.05],
                'colsample_bytree': [0.5, 0.8],
                'subsample': [0.8],
                'n_estimators': [150, 1000],
                'reg_alpha': [0.3],
                'max_depth': [4, 5, 6],
                'gamma': [1, 5],
                'eval_metric': ['mae']
            }

        }

    ]

    for model in models:
        print('\n', '-'*20, '\n', model['name'])
        start = time.perf_counter()
        grid = GridSearchCV(model['estimator'], param_grid=model['hyperparameters'], cv=5, scoring="accuracy",
                            verbose=False, n_jobs=1)
        grid.fit(X, Y.values.ravel())
        best_models[model['name']] = {'score': grid.best_score_, 'params': grid.best_params_}
        run = time.perf_counter() - start
        print('accuracy: {}\n{} --{:.2f} seconds.'.format(str(grid.best_score_), str(grid.best_params_), run))

    return best_models


X, y = X_train, y_train
best = select_model(X, y)
