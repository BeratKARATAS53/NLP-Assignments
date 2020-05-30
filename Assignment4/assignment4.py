import random
import math
import numpy as np
# import dynet as dy
import json

from scipy import spatial
from sklearn.manifold import TSNE

class FNN:
    def __init__(self):
        print("Init")
        self.embeddings_dict = {}
        self.unim_poem = {}
        
    def glove_file(self, folder_path):
        print("Read Glove")
        with open(folder_path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        
        return self.embeddings_dict
    
    def poem_file(self, folder_path):
        print("Read Poem")
        with open(folder_path, 'r') as f:
            for line in f:
                print(line)
                poem = line.get("poem")
                print("-------------------------")
                print(poem)
                # values = poem.splitlines()
                # line["poem"] = values
                # print("-------------------------")
                # print(line)
            self.unim_poem = json.load(f)
        
        return self.unim_poem
    
    def process_text(self, text):
        text = text.lower()
        tokens = text.split()
        table = str.maketrans('', '', str.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        return tokens
    
    def find_closest_embeddings(self, embedding):
        print("Find Closest")
        return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))
    
    total_corpus = [] # Total Words Number in the Dataset

    # def biGram(self): # Generate NGram Models
    #     print("Creating Bi-gram Model...")
        
    #     bigram_list = []
    #     corpus = 0
        
    #     for data in dataset_list: # 
    #         data = data.split()
    #         corpus += len(data)
    #         sentence_ngram = []
    #         for number in range(0, len(data)): 
    #             ngram = ' '.join(data[number:number + n])
    #             sentence_ngram.append(ngram)
    #         ngrams_list.append(sentence_ngram)
            
    #     total_corpus.append(corpus)
        
    #     return ngrams_list

neuralNetwok = FNN()

embeddings_dict = neuralNetwok.glove_file("./Assignment4/glove.6B/glove.6B.50d.txt")
poem_dict = neuralNetwok.poem_file("./Assignment4/unim_poem.json")

closest_embedding = neuralNetwok.find_closest_embeddings(embeddings_dict["king"])[1:6]
print(closest_embedding)

print(poem_dict)