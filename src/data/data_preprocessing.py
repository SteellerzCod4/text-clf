import pandas as pd
import json
from sklearn.model_selection import train_test_split
from src.preprocessors.data_cleaner import MyDataCleaner
from src.preprocessors.embeddings import MyEmbeddings
from src.preprocessors.truncator import MyTruncator

PATH = r"C:\Users\User\Desktop\datasets\bbs_text\bbc-text.csv"


with open('model_config.json', 'r') as fp:
    data = json.load(fp)

# import data
df = pd.read_csv(data['data'])
print(df.head())

label_ids = {label: i for i, label in enumerate(df['category'].unique())}


# id2labels = {i: label for label, i in label_ids.items()}

def replace_labels(label):
    return label_ids[label]


df['category'] = df['category'].apply(replace_labels)

# clean data
cleaner = MyDataCleaner("english", True, True)
df['text'] = cleaner.clean_data(df['text'])

# truncate data
max_doc_length = 200
truncator = MyTruncator(max_doc_length)
df['text'] = truncator.truncate_data(df['text'])

# separate data
X, y = df.drop('category', axis=1), df['category']

# create embaddings
embedder = MyEmbeddings("one-hot", 150)
embedding_X = embedder.fit_transform(X)

# train and test separation
X_train, X_test, y_train, y_test = train_test_split(embedding_X, y.values, test_size=0.2, stratify=y, random_state=42)

print("Training is finished")
