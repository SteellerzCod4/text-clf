import telebot
import pandas as pd
import torch
import json


from src.preprocessors.data_cleaner import MyDataCleaner
from src.preprocessors.embeddings import MyEmbeddings
from src.preprocessors.truncator import MyTruncator

TOKEN = '6484128996:AAGO1ggOxUxOVG5gy5uZRnzf5pHEKfCGZ9k'

if __name__ == "__main__":
    bot = telebot.TeleBot(TOKEN, parse_mode=None)
    print("bot is working")


    def load_json_data(json_data):
        with open(json_data, 'r') as fp:
            data = json.load(fp)
        return data


    def preprocess_text(text):
        df = pd.DataFrame({"text": [text]})

        cleaner = MyDataCleaner("english", True, True)
        df['text'] = cleaner.clean_data(df['text'])

        max_doc_length = 200
        truncator = MyTruncator(max_doc_length)
        df['text'] = truncator.truncate_data(df['text'])

        embedder = MyEmbeddings("one-hot", 150)
        embedder.voc = set(load_json_data("scripts/voc.json"))
        embedder.token2id = load_json_data("scripts/token2id_dict.json")
        embedding = embedder.transform(df)

        return embedding


    def convert_index_to_label(prediction):
        id2label = {0: 'tech', 1: 'business', 2: 'sport', 3: 'entertainment', 4: 'politics'}
        return id2label[prediction.item()]


    def prediction(message):
        text_to_clf = message.text
        texts_embedding = preprocess_text(text_to_clf)
        model = torch.load('scripts/my_model.pt')
        y_pred = model.predict(texts_embedding)
        return convert_index_to_label(y_pred)


    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        bot.reply_to(message, "Howdy, how are you doing?")


    @bot.message_handler()
    def send_welcome(message):
        predicted_label = prediction(message)
        bot.reply_to(message, predicted_label)


    bot.infinity_polling()
