import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('train_output.csv', sep=';')
df2 = pd.read_csv('test_output.csv', sep=';')

X_train = df1[['effectiveness', 'sideEffects', 'benefitsReview_polarity', 'benefitsReview_subjectivity',
                'sideEffectsReview_polarity', 'sideEffectsReview_subjectivity', 'commentsReview_polarity',
                'commentsReview_subjectivity', 'urlDrugName_group', 'condition_group', 'benefitsReview_len',
               'sideEffectsReview_len', 'commentsReview_len']]
y_train = df1[['rating']]

X_test = df2[['effectiveness', 'sideEffects', 'benefitsReview_polarity', 'benefitsReview_subjectivity',
                'sideEffectsReview_polarity', 'sideEffectsReview_subjectivity', 'commentsReview_polarity',
                'commentsReview_subjectivity', 'urlDrugName_group', 'condition_group', 'benefitsReview_len',
               'sideEffectsReview_len', 'commentsReview_len']]
y_test = df2[['rating']]
# -------------------------------------------------
# CORRELATION - DONE
X = X_train
y = y_train

corr_matrix = X.corr()
corr_features = corr_matrix.index
plt.figure(figsize=(13, 6))
g=sns.heatmap(X[corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# ----------------------------------------------------------
# Univariate Selection - DONE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = X_train
y = y_train


bestfeatures = SelectKBest(score_func=chi2)
fit = bestfeatures.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

featureScores = pd.concat([df_columns, df_scores], axis=1)
featureScores.columns = ['Feature', 'Score']
print(featureScores.sort_values(by='Score', ascending=False))

# ----------------------------------
# FEATURE IMPORTANCE - DONE
from sklearn.ensemble import ExtraTreesClassifier

X = X_train
y = y_train

model = ExtraTreesClassifier()
model.fit(X, y.values.ravel())
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
