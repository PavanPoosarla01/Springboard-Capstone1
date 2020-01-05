'''
Springboard Data Science Career Track
Capstone Project 1 : TED talk analysis
Author : Pavan Poosarla, pavanpoosarla01@gmail.com
Start Date : 9/ 10/ 2019
Description :
As a part of the first capstone project, I will be analysing TED talk transcripts and analyse the sentiment

Date Source
https://www.kaggle.com/rounakbanik/ted-talks/downloads/ted-talks.zip/3
'''

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


nltk.download('punkt')
def CheckMissing_df(df):
    '''Function to remove missing values from the dataframe. It does the following
    Print the number of missing values in each columns
    '''
    missing_dict = {'ColName' : [], 'MissingNumber':[]}
    df_cols = df.columns
    for item in df_cols:
        missing_dict['ColName'].append(item)
        missing_dict['MissingNumber'].append(df[item].isna().sum())
    missing_df = pd.DataFrame(missing_dict)
    return missing_df

def text_normalize(text):
    normalized_text = text
    # Strip leading and lagging whitespace
    normalized_text = normalized_text.strip()
    # Convert all text to lower case
    normalized_text.lower()
    # Remove punctuation
    normalized_text.translate( str.maketrans('','', string.punctuation))
    # Word tokenization
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(normalized_text)
    result = [i for i in tokens if not i in stop_words]


    stemmer = PorterStemmer()
    stemmed_result = []
    for word in result:
        stemmed_result.append(stemmer.stem(word))
    return stemmed_result

# Read data from the Kaggle TED talk database
df_main = pd.read_csv(r'ted_main.csv', parse_dates = ['film_date', 'published_date'])
df_transcripts = pd.read_csv(r'transcripts.csv')

# lets work with those combined dataframe from both files
df_joined = pd.merge (df_main, df_transcripts, how = 'inner', on = 'url')
print ('Combined Dataframe is read')
print ('Raw dataset has %d columns and %d talks'%(df_joined.shape[1], df_joined.shape[0]))

# Look for any missing data
missing_df = CheckMissing_df(df_joined)
print ('Columns with missing values are /n', missing_df[missing_df.MissingNumber != 0])
# Fill Missing values
df_joined['speaker_occupation'].fillna('Unknown', inplace = True)
print ('After filling, # missing in speaker occupation is', df_joined['speaker_occupation'].isna().sum())

# Build a clean dataset from the raw dataset, df_clean
# Drop talks with more than 1 speaker
df_clean = df_joined[df_joined.num_speaker == 1] # To avoid convoluting the analysis

# Drop the columns that are not meaningful to the analysis
df_clean = df_clean.drop(columns = ['related_talks', 'languages', 'url','event','name','speaker_occupation','views', 'num_speaker'])

print ('After dropping extra columns, dataset has', df_clean.columns)

# We need to clean up the ext data : description, transcript
test_talk = df_clean.loc[2,'transcript']
# print (test_talk)
# Count number of sentences
sentence_list = test_talk.split('.')
print ('No of sentences are ',len(sentence_list))
word_count = 0
for sentence in sentence_list:
    word_list = sentence.strip( ). split(' ')
    word_count = word_count + len(word_list)
#     print (word_list)
# print (sentence_list)
print ('No of words in talk are', word_count)
'''# lets also plot a histogram of the number of views
plt.figure()
hist_views =df_clean['views'].hist(bins = 50)
# plt.yscale('log')
plt.xlabel ('Number of views')
plt.ylabel('Number of talks')
# plt.show()
plt.xscale('log')
plt.draw()
plt.pause(5)
print ('Max number of views is ', max(df_joined.views))
print ('Min number of views is ', min(df_joined.views))

'''
# Counts of events
print('Number of different TED events is ', df_joined.groupby('event').count().shape[0])
print (len(word_tokenize(test_talk)))

result = text_normalize(test_talk)
print (len(result))