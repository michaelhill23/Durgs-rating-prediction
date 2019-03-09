import pandas as pd
import warnings;warnings.simplefilter('ignore')
import numpy as np
import matplotlib.pyplot as plt

conditions = pd.read_csv('conditions.csv', sep=';')

condition_value = 'acne'

dfx = conditions[conditions['Condition'] == condition_value]
dfx = dfx.reset_index(drop=True)

cols = ['Effectiveness (0-4)', 'Side effects (0-4)', 'Benefits review polarity (0-1)',
        'Side effects review polarity (0-1)', 'Comments review polarity (0-1)', 'Number of reviews',
        'Prediction mae', 'Predicted rating (1-10)']
for col in cols:
    dfx[col] = pd.to_numeric(dfx[col])

effectiveness = dfx[['Drug name', 'Effectiveness (0-4)']]
effectiveness.sort_values('Effectiveness (0-4)', ascending=True).plot(x="Drug name", kind="barh")
plt.title('Mean effectiveness of drugs for Acne condition')
text_ticks = ['Ineffective', 'Marginally', 'Moderately', 'Considerably', 'Highly']
numerical_ticks = np.array([0, 1, 2, 3, 4])
plt.xticks(numerical_ticks, text_ticks)

sideEffects = dfx[['Drug name', 'Side effects (0-4)']]
sideEffects.sort_values('Side effects (0-4)', ascending=True).plot(x="Drug name", kind="barh")
plt.title('Mean side effects of drugs for Acne condition')
text_ticks = ['Extremely Severe', 'Severe', 'Moderate', 'Mild', 'No']
numerical_ticks = np.array([0, 1, 2, 3, 4])
plt.xticks(numerical_ticks, text_ticks)

polarity = dfx[['Drug name', 'Benefits review polarity (0-1)', 'Side effects review polarity (0-1)',
                     'Comments review polarity (0-1)']]
polarity.plot(x="Drug name", kind="barh")
plt.title('Mean polarity of drugs\' reviews for Acne condition')
text_ticks = ['Negative', 'Neutral', 'Positive']
numerical_ticks = np.array([0, 0.5, 1])
plt.xticks(numerical_ticks, text_ticks)

pred_rating = dfx[['Drug name', 'Predicted rating (1-10)']]
err = dfx['Prediction mae'].values

pred_rating.plot(x="Drug name", kind="bar", yerr=err)
plt.title('Mean predicted ratings of drugs for Acne condition with errors')
numerical_ticks = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.yticks(numerical_ticks)

plt.show()

