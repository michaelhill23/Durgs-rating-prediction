import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# libs for further researches
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def delete_null_vals(train_ds):
    print(train_ds.isna().sum())
    train_ds = train_ds.dropna(axis=0, how='any')
    train_ds = train_ds.reset_index(drop=True)
    return train_ds


def text_to_numbers(train_ds, test_ds):
    train_ds['effectiveness'].replace({'Ineffective': 0,
                                      'Marginally Effective': 1,
                                      'Moderately Effective': 2,
                                      'Considerably Effective': 3,
                                      'Highly Effective': 4}, inplace=True)

    test_ds['effectiveness'].replace({'Ineffective': 0,
                                       'Marginally Effective': 1,
                                       'Moderately Effective': 2,
                                       'Considerably Effective': 3,
                                       'Highly Effective': 4}, inplace=True)

    train_ds['sideEffects'].replace({'Extremely Severe Side Effects': 0,
                                    'Severe Side Effects': 1,
                                    'Moderate Side Effects': 2,
                                    'Mild Side Effects': 3,
                                    'No Side Effects': 4}, inplace=True)

    test_ds['sideEffects'].replace({'Extremely Severe Side Effects': 0,
                                     'Severe Side Effects': 1,
                                     'Moderate Side Effects': 2,
                                     'Mild Side Effects': 3,
                                     'No Side Effects': 4}, inplace=True)
    return train_ds, test_ds

def decontract_reviews(ds):

    columns =['benefitsReview', 'sideEffectsReview', 'commentsReview']
    for column in columns:
        reviews = ds[column]
        for i in range(0, len(reviews)):
            review = reviews[i]
            if not review:
                i += 1
            else:
                # specific
                review = re.sub(r"won\'t", "will not", review)
                review = re.sub(r"can\'t", "can not", review)
                # general
                review = re.sub(r"n\'t", " not", review)
                review = re.sub(r"\'re", " are", review)
                review = re.sub(r"\'s", " is", review)
                review = re.sub(r"\'d", " would", review)
                review = re.sub(r"\'ll", " will", review)
                review = re.sub(r"\'t", " not", review)
                review = re.sub(r"\'ve", " have", review)
                review = re.sub(r"\'m", " am", review)
                ds.at[i, column] = review
    return ds


def clear_reviews(ds):
    columns = ['benefitsReview', 'sideEffectsReview', 'commentsReview']
    for column in columns:
        reviews = ds[column]
        for i in range(0, len(reviews)):
            review = reviews[i]
            if not review:
                i += 1
            else:
                review = re.sub(r'\W', ' ', review)

                # Remove all single characters not at the begining of sentence
                review = re.sub(r'\s+[a-zA-Z]\s+', ' ', review)

                # Remove single characters from the start
                review = re.sub(r'\^[a-zA-Z]\s+', ' ', review)

                # Remove digits from text
                review = re.sub(r'\d+', ' ', review)

                # Substituting multiple spaces with single space
                review = re.sub(r'\s+', ' ', review, flags=re.I)

                # Lowercase all characters
                review = review.lower()

                ds.at[i, column] = review

    return ds


def stemm_reviews(ds):
    columns = ['benefitsReview', 'sideEffectsReview', 'commentsReview']
    snowball = SnowballStemmer(language='english')
    for column in columns:
        # corpus = []
        reviews = ds[column]

        for i in range(0, len(reviews)):
            review = reviews[i]
            review = review.split()
            review = [snowball.stem(word) for word in review]
            review = ' '.join(review)
            ds.at[i, column] = review
            # corpus.append(review)
    return ds


def reviews_sentiment_measure(ds):
    columns = ['benefitsReview', 'sideEffectsReview', 'commentsReview']

    for column in columns:
        reviews = ds[column]
        rev_polarity = []
        rev_subjectivity = []

        for i in range(0, len(reviews)):
            review = reviews[i]
            senti = TextBlob(review)
            polarity = senti.sentiment.polarity
            subjectivity = senti.sentiment.subjectivity
            rev_polarity.append(polarity)
            rev_subjectivity.append(subjectivity)

        column_sent_df = pd.DataFrame({column + '_polarity': rev_polarity,
                                     column + '_subjectivity': rev_subjectivity})

        scaler = MinMaxScaler()
        column_sent_df[[column + '_polarity']] = scaler.fit_transform(column_sent_df[[column + '_polarity']])

        ds = pd.concat([ds, column_sent_df], axis=1)

    return ds


def reviews_lenghts(train_ds, test_ds):
    columns = ['benefitsReview', 'sideEffectsReview', 'commentsReview']
    for col in columns:
        new_col = col + '_len'
        train_ds[new_col] = train_ds[col].apply(lambda x: 0 if x is np.nan else len(x.split()))
        test_ds[new_col] = test_ds[col].apply(lambda x: 0 if x is np.nan else len(x.split()))

    # train_ds.drop(columns, axis=1, inplace=True)
    # test_ds.drop(columns, axis=1, inplace=True)
    return train_ds, test_ds


def group_drugs(train_ds, test_ds):
    df = pd.concat([train_ds['urlDrugName'], test_ds['urlDrugName']], ignore_index=True)
    df = df.unique()
    df = pd.DataFrame(df, columns=['urlDrugName'])
    df.insert(0, 'urlDrugName_group', range(1, 1 + len(df)))

    train_ds['urlDrugName_group'] = train_ds['urlDrugName'].replace(df['urlDrugName'].values, df['urlDrugName_group'].values)
    test_ds['urlDrugName_group'] = test_ds['urlDrugName'].replace(df['urlDrugName'].values, df['urlDrugName_group'].values)

    return train_ds, test_ds


def group_conditions(train_ds, test_ds):
    df = pd.concat([train_ds['condition'], test_ds['condition']], ignore_index=True)
    df = df.unique()
    df = pd.DataFrame(df, columns=['condition'])
    df.insert(0, 'condition_group', range(1, 1 + len(df)))

    train_ds['condition_group'] = train_ds['condition'].replace(df['condition'].values, df['condition_group'].values)
    test_ds['condition_group'] = test_ds['condition'].replace(df['condition'].values, df['condition_group'].values)

    return train_ds, test_ds

train_dataset = pd.read_csv('training_data.tsv', sep='\t', header=0)
test_dataset = pd.read_csv('test_data.tsv', sep='\t', header=0)


train_dataset = delete_null_vals(train_dataset)
train_dataset, test_dataset = text_to_numbers(train_dataset, test_dataset)
train_dataset = decontract_reviews(train_dataset)
test_dataset = decontract_reviews(test_dataset)
train_dataset = clear_reviews(train_dataset)
test_dataset = clear_reviews(test_dataset)
# train_dataset = stemm_reviews(train_dataset)
# test_dataset = stemm_reviews(test_dataset)
train_dataset = reviews_sentiment_measure(train_dataset)
test_dataset = reviews_sentiment_measure(test_dataset)
train_dataset, test_dataset = reviews_lenghts(train_dataset, test_dataset)
train_dataset, test_dataset = group_drugs(train_dataset, test_dataset)
train_dataset, test_dataset = group_conditions(train_dataset, test_dataset)

train_dataset.to_csv('train_output.csv', sep=';')
test_dataset.to_csv('test_output.csv', sep=';')
