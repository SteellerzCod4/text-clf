import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class MyDataCleaner:
    def __init__(self, language="english", delete_stopwords=True, remove_wrong_sym=True):
        self.language = language
        self.delete_stopwords = delete_stopwords
        self.remove_wrong_sym = remove_wrong_sym

    @staticmethod
    def delete_links(doc):
        start_ind = doc.find("&lt")
        fin_ind = doc.rfind("&gt") + 3
        return doc[:start_ind] + doc[fin_ind:]

    @staticmethod
    def remove_wrong_symbols(doc):
        # doc = self.delete_links(doc)
        new_doc = []
        for symbol in doc:
            if symbol == "-":
                new_doc.append("")
            elif symbol.isalpha():
                new_doc.append(symbol.lower())
            elif symbol == " " or symbol == "\\":
                new_doc.append(" ")
        new_doc = "".join(new_doc)
        return new_doc

    def remove_stopwords(self, doc):
        return " ".join([word for word in doc.split() if not word in self.stop_words])

    def clean_data(self, data):
        if self.remove_wrong_sym:
            data = data.apply(self.remove_wrong_symbols)

        if self.delete_stopwords:
            self.stop_words = set(stopwords.words(self.language))
            self.stop_words.add("ltbgtltbgt")
            data = data.apply(self.remove_stopwords)

        return data

# Example of usage:
# cleaner = MyDataCleaner("english", True, True)
# df['text'] = cleaner.clean_data(df['text'])
