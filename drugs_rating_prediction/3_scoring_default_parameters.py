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

clf_logreg = LogisticRegression()
print('Logistic Regression')
prediction(clf_logreg)

clf_rf = RandomForestClassifier()
print('Random Forest')
prediction(clf_rf)

clf_gnb = GaussianNB()
print('Gaussian Naive Bayes')
prediction(clf_gnb)

clf_xgb = XGBClassifier()
print('XGBoost')
prediction(clf_xgb)












