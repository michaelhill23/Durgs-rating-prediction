import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.simplefilter('ignore')
import numpy as np


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


def prediction(model):
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    return y_pred


clf_rf = RandomForestClassifier(bootstrap=True,  criterion="entropy", max_features=0.55, min_samples_leaf=20,
                                min_samples_split=10, n_estimators=100)

rating_pred = prediction(clf_rf)
predictions = pd.DataFrame(rating_pred, columns=['rating_pred'])
df2 = pd.concat([df2, predictions], axis=1)
df2['rating_pred_diff'] = abs(df2['rating'] - df2['rating_pred'])
df2 = df2[['urlDrugName', 'urlDrugName_group','condition', 'condition_group', 'effectiveness', 'sideEffects',
           'benefitsReview_polarity', 'sideEffectsReview_polarity', 'commentsReview_polarity',
           'rating_pred', 'rating_pred_diff']]


cols_for_groupby = ['urlDrugName']

drugs = df2.groupby(['urlDrugName']).agg({'effectiveness': [np.mean],
                                        'sideEffects': [np.mean],
                                        'benefitsReview_polarity': [np.mean],
                                        'sideEffectsReview_polarity': [np.mean],
                                        'commentsReview_polarity': [np.mean],
                                        'rating_pred': [np.mean],
                                        'rating_pred_diff': [np.mean, np.size]}).reset_index()

conditions = df2.groupby(['condition', 'urlDrugName']).agg({'effectiveness': [np.mean],
                                        'sideEffects': [np.mean],
                                        'benefitsReview_polarity': [np.mean],
                                        'sideEffectsReview_polarity': [np.mean],
                                        'commentsReview_polarity': [np.mean],
                                        'rating_pred': [np.mean],
                                        'rating_pred_diff': [np.mean, np.size]}).reset_index()



drugs.to_csv('drugs.csv', sep=';', index=False)
conditions.to_csv('conditions.csv', sep=';', index=False)





