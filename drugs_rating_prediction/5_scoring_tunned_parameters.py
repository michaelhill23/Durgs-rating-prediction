import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
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

# X_train = df1[['effectiveness', 'sideEffects', 'benefitsReview_polarity', 'benefitsReview_subjectivity',
#                 'sideEffectsReview_polarity', 'sideEffectsReview_subjectivity', 'commentsReview_polarity',
#                 'commentsReview_subjectivity', 'urlDrugName_group', 'condition_group', 'benefitsReview_len',
#                'sideEffectsReview_len', 'commentsReview_len']]
# y_train = df1[['rating']]
#
# X_test = df2[['effectiveness', 'sideEffects', 'benefitsReview_polarity', 'benefitsReview_subjectivity',
#                 'sideEffectsReview_polarity', 'sideEffectsReview_subjectivity', 'commentsReview_polarity',
#                 'commentsReview_subjectivity', 'urlDrugName_group', 'condition_group', 'benefitsReview_len',
#                'sideEffectsReview_len', 'commentsReview_len']]
#
# y_test = df2[['rating']]


def prediction(model):

    # mean accuracy with cross validation
    cv_accuracy = cross_val_score(model, X_train, y_train.values.ravel(), cv=10, scoring='accuracy')
    print("Mean cross validation accuracy is {}".format(cv_accuracy.mean()))

    # training selected model
    model.fit(X_train, y_train.values.ravel())

    # making prediction on test data set
    y_pred = model.predict(X_test)

    # measuring accuracy score for selected model
    acc_score = accuracy_score(y_test, y_pred)
    print('Predicted accuracy is :{}\n'.format(acc_score))


clf_logreg = LogisticRegression(C=1, max_iter=100, solver='newton-cg')
print('Logistic Regression')
prediction(clf_logreg)

clf_rf = RandomForestClassifier(bootstrap=True,  criterion="entropy", max_features=0.55, min_samples_leaf=20,
                                min_samples_split=10, n_estimators=100)
print('Random Forest')
prediction(clf_rf)

clf_gnb = GaussianNB()
print('Gaussian Naive Bayes')
prediction(clf_gnb)

clf_xgb = XGBClassifier(colsample_bytree=0.8, eval_metric='mae', gamma=1, learning_rate=0.01, max_depth=6,
                        n_estimators=150, reg_alpha=0.3, subsample=0.8)
print('XGBoost')
prediction(clf_xgb)












