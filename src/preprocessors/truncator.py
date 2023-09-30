import numpy as np


class MyTruncator:
    def __init__(self, doc_length):
        self.doc_length = doc_length

    def truncate_doc(self, doc):
        if len(doc.split()) < self.doc_length:
            return doc
        return " ".join(np.random.choice(doc.split(), size=self.doc_length, replace=False).tolist())

    def truncate_data(self, data):
        data = data.apply(self.truncate_doc)
        return data


# max_doc_length = 200
# truncator = MyTruncator(max_doc_length)
# df['text'] = truncator.truncate_data(df['text'])
