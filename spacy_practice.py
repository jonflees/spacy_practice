import spacy
import glob
import os
import json
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

# Get current working directory
cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))

# Change the current working directory
os.chdir('/Users/jonflees/spaCy/2018_01_112b52537b67659ad3609a234388c50a')

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Store all files from folder in a list
all_files = []
for each_file in glob.glob('*.json'):
    all_files.append(each_file)


# Initialize empty DataFrame
df = pd.DataFrame()

for i in range(10):         #len(all_files)):

    # Initialize lists
    article_sent_score = []
    article_sent_label = []
    title_sent_score = []
    title_sent_label = []
    positive_words = []
    negative_words = []
    total_pos = []
    total_neg = []
    ent_dict = {}

    # Open file
    xx = open(all_files[i])

    # Save json file as dictionary
    data = json.load(xx)

    # Store and clean text from article
    text = data['text']
    text = text.strip().replace("  ","")
    text = "".join([s for s in text.splitlines(True) if s.strip("\r\n")])

    # Store text of article as document for spaCy
    doc = nlp(text)

    # Sentiment analysis of article text
    sentiment = doc._.blob.polarity
    sentiment = round(sentiment,2)
    if sentiment > 0:
        sent_label = "Positive"
    elif sentiment < 0:
        sent_label = "Negative"
    else:
        sent_label = "Neutral"

    # Add sent_label and sentiment to respective lists
    article_sent_label.append(sent_label)
    article_sent_score.append(sentiment)

    # Store title as doc for spaCy
    doc_title = nlp(data['title'])

    # Sentiment analysis of title
    title_sentiment = doc_title._.blob.polarity
    title_sentiment = round(title_sentiment,2)
    if title_sentiment > 0:
        t_sent_label = "Positive"
    elif title_sentiment < 0:
        t_sent_label = "Negative"
    else:
        t_sent_label = "Neutral"

    # Add sent_label and sentiment to respective lists
    title_sent_label.append(t_sent_label)
    title_sent_score.append(title_sentiment)

    # Classify words in article as positive or negative
    for x in doc._.blob.sentiment_assessments.assessments:
        if x[1] > 0:
            positive_words.append(x[0][0])
        elif x[1] < 0:
            negative_words.append(x[0][0])
        else:
            pass

    total_pos.append(', '.join(set(positive_words)))
    total_neg.append(', '.join(set(negative_words)))

    # Store a dictionary resulting from NER of title
    for ent in doc_title.ents:
        #print(ent.text, ent.start_char, ent.end_char, ent.label_)
        ent_dict[ent.text] = ent.label_

    # Close file
    xx.close

    # Append to DataFrame
    df = df.append({'file': all_files[i], 'title': data['title'], 'title_sent_label': title_sent_label, 
                    'title_sent_score': title_sent_score, 'article_sent_label': article_sent_label, 
                    'article_sent_score': article_sent_score, 'pos_words': total_pos, 'neg_words': total_neg,
                    'entities': ent_dict}, 
                        ignore_index = True)


print(df)

# Export to CSV
df.to_csv('sentiment.csv')
