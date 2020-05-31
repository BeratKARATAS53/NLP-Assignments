import re
import random
import math
import numpy as np
import string
import json

import dynet as dy
import dynet_config
dynet_config.set_gpu()

import nltk
from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer, word_tokenize
from collections import Counter
# nltk.download('punkt')


class FNN:
    def __init__(self):
        print("Init")
        self.embeddings = []
        self.word_to_index = {}
        self.index_to_vocab = {}
        self.vocab_to_index = {}
        self.word_vectors = {}
        
        self.unim_poem = []
        self.poems = []
        self.mini_poems = []
        
        self.word_list = []
        self.words_count = []
        self.bigram_poem = []
        self.bigram_count = {}
        self.bigram_prob = {}

    def glove_file(self, folder_path):
        print("Read Glove")
        with open(folder_path, 'r', encoding='UTF-8') as file:
            for line in file:
                line = line.split()
                self.word_to_index[line[0]] = len(self.embeddings)
                self.embeddings.append(np.array(line[1:], dtype=np.float32))


    def poem_file(self, folder_path):
        print("Read Poem")
        with open(folder_path, 'r') as f:
            self.unim_poem = json.load(f)
            mini_poems = random.sample(self.unim_poem, 100)
            self.preprocessing(mini_poems)

    def preprocessing(self, unim_poem):
        print("Preproccesing")
        for each in unim_poem:
            for k, v in each.items():
                if(k == "poem"):
                    each["poem"] = each["poem"].replace("\n", " eol ")
                    each["poem"] = "bos " + each["poem"] + " eos"
                    self.poems.append(each["poem"])
        

    def vocabulary(self):
        print("Vocabulary")
        for each in self.poems:
            nltk_tokens = nltk.word_tokenize(each)
            for token in nltk_tokens:
                if(token not in self.vocab_to_index):
                    if(token in self.word_to_index):
                        self.word_vectors[token] = self.embeddings[self.word_to_index[token]]
                    else:
                        self.word_vectors[token] = np.random.rand(50)
                    self.vocab_to_index[token] = len(self.word_vectors)
                    self.index_to_vocab[len(self.word_vectors)] = token
            

    def bigram(self):
        print("Bigram")
        for each_poem in self.poems:
            nltk_tokens = nltk.word_tokenize(each_poem)
            self.bigram_poem += list(nltk.bigrams(nltk_tokens))
        print(self.bigram_poem)
    # def bigram_counts(self):
    #     print("Bigram Count")
    #     for poem in self.bigram_poem:
    #         fdist = nltk.FreqDist(poem)
    #         for k, v in fdist.items():
    #             self.bigram_count[k] = v

    # def calculate_bigramProbs(self):
    #     print("Bigram Prob")
    #     for k, v in self.bigram_count.items():
    #         first_word = k[0]
    #         bigram_prob = (
    #             self.bigram_count[k] + 1) / (self.words_count[first_word] + len(self.words_count))
    #         self.bigram_prob[k] = bigram_prob

    # def model(self):
    #     hidden_size = 64
    #     vocabulary_size = len(self.words_count)
    #     input_size = output_size = vocabulary_size

    #     m = dy.Model()
        
    #     W = m.add_parameters((hidden_size, input_size))
    #     b = m.add_parameters(hidden_size)
    #     V = m.add_parameters((output_size, hidden_size))
    #     a = m.add_parameters(output_size)
        
    #     x = dy.vecInput(input_size)
    #     y = dy.vecInput(output_size)
    #     h = dy.tanh((W * x) + b)
    #     output = dy.logistic(V*h)
        
    #     y_pred = h

    #     loss = dy.squared_distance(output, y)
    #     trainer = dy.SimpleSGDTrainer(m)
        
    #     for iter in range(1):
    #         mloss = 0.0
    #         seen_instances = 0
    #         for poem in self.bigram_poem:
    #             for word_pair in poem:
    #                 print(self.embeddings_dict.get(word_pair[0]))
    #                 x.set(self.embeddings_dict.get(word_pair[0]))
    #                 y.set(self.embeddings_dict.get(word_pair[1]))
    #                 seen_instances += 1

    #                 mloss += loss.scalar_value()
    #                 loss.forward()
    #                 loss.backward()
    #                 trainer.update()

    #                 if (seen_instances > 1 and seen_instances % 100 == 0):
    #                     print(seen_instances, "/", len(self.bigram_poem), "***average loss is:", mloss / seen_instances)

    #         print( mloss / seen_instances)


neuralNetwork = FNN()
neuralNetwork.glove_file("./Assignment4/glove.6B/glove.6B.50d.txt")
neuralNetwork.poem_file("./Assignment4/unim_poem.json")

# neuralNetwork.preprocessing()
neuralNetwork.bigram()
# neuralNetwork.vocabulary()
# neuralNetwork.bigram_counts()
# neuralNetwork.calculate_bigramProbs()

# rand = random.choice(neuralNetwork.word_list)
# print(rand)
# print(neuralNetwork.embeddings_dict[rand])

# print(neuralNetwork.model())
