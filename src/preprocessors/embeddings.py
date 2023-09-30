import numpy as np
from tqdm import tqdm
import json


class MyEmbeddings:
    def __init__(self, trans_type="one-hot", embedding_length=20):
        self.trans_type = trans_type
        self.embedding_length = embedding_length

    def make_vocabulary(self, data):
        voc = set()
        for column in data.columns:
            for doc in tqdm(data[column], desc=f"Making vocabulary from column '{column}'"):
                for word in doc.split():
                    # if len(word) > 2 and len(word) < 13:
                    voc.add(word)
        return voc

    @staticmethod
    def make_token2id(voc):
        return dict([("<PAD>", 0)]
                    + [("<UNKNOWN>", 1)]
                    + [(token, id + 2) for id, token in enumerate(voc)])

    def replace_token2id(self, doc):
        new_doc = [self.token2id[token] if token in self.voc else self.token2id["<UNKNOWN>"] for token in doc.split()]
        while len(new_doc) < self.embedding_length:
            new_doc.append(self.token2id["<PAD>"])
        return new_doc[:self.embedding_length]

    def remove_token_to_id(self, data):
        for column in tqdm(data.columns, desc=f"Transform column words to tokens"):
            data[column] = data[column].apply(self.replace_token2id)
        return data

    def vectorize(self, data_with_ids):
        matrix = np.zeros((data_with_ids.shape[0], len(self.voc) + 2))
        for column in data_with_ids.columns:
            for i, indices in tqdm(enumerate(data_with_ids[column]), desc=f"Vectorizing column '{column}'"):
                if self.trans_type == 'one-hot':
                    matrix[i, indices] = 1
        return matrix

    @staticmethod
    def save_info(info, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(info, file)

    @staticmethod
    def save_set_to_file(my_set, file_name):
        # Convert the set to a list
        list_from_set = list(my_set)

        # Convert the list to a JSON string
        json_string = json.dumps(list_from_set)

        # Write the JSON string to a file
        with open(file_name, 'w') as outfile:
            outfile.write(json_string)

    def transform(self, data):
        data_with_ids = self.remove_token_to_id(data)
        vectorized_data = self.vectorize(data_with_ids)
        return vectorized_data

    def fit_transform(self, data):
        self.voc = self.make_vocabulary(data)
        self.token2id = self.make_token2id(self.voc)
        data_with_ids = self.remove_token_to_id(data)
        vectorized_data = self.vectorize(data_with_ids)
        self.save_info(self.token2id, "token2id_dict.json")
        self.save_set_to_file(self.voc, 'voc.json')
        return vectorized_data

# embedder = MyEmbeddings("one-hot", 150)
# embedding_X = embedder.fit_transform(X)
