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
# nltk.download('punkt')

from collections import Counter
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.util import ngrams
from nltk import FreqDist

file = open('poems.txt', 'w')
class FNN:
    def __init__(self):
        print("Init")
        self.embeddings = []
        self.word_to_index = {}
        self.index_to_vocab = {}
        self.vocab_to_index = {}
        self.unique_words = []
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
                    self.unique_words.append(token)

    def bigram(self):
        print("Bigram")
        for each_poem in self.poems:
            nltk_tokens = nltk.word_tokenize(each_poem)
            self.bigram_poem += list(nltk.bigrams(nltk_tokens))

    def model(self):
        hidden_size = 64
        vocabulary_size = len(self.vocab_to_index)
        input_size = output_size = vocabulary_size

        m = dy.Model()

        W = m.add_parameters((hidden_size, input_size))
        b = m.add_parameters(hidden_size)
        V = m.add_parameters((output_size, hidden_size))
        a = m.add_parameters(output_size)

        x = dy.vecInput(input_size)
        y = dy.vecInput(output_size)
        h = dy.tanh((W * x) + b)
        output = dy.softmax(V*h)

        y_pred = h

        loss = dy.squared_distance(output, y)
        trainer = dy.SimpleSGDTrainer(m)

        for iter in range(1):
            mloss = 0.0
            seen_instances = 0
            for word_pair in self.bigram_poem:
                x.set(self.word_vectors[word_pair[0]])
                y.set(self.word_vectors[word_pair[1]])
                seen_instances += 1

                mloss += loss.scalar_value()
                loss.forward()
                loss.backward()
                trainer.update()

                if (seen_instances > 1 and seen_instances % 100 == 0):
                    print(seen_instances, "/", len(self.bigram_poem),
                          "***average loss is:", mloss / seen_instances)

            print(mloss / seen_instances)

    def predict(self, word):
        hidden_size = 64
        vocabulary_size = len(self.vocab_to_index)
        input_size = output_size = vocabulary_size

        m = dy.Model()

        W = m.add_parameters((hidden_size, input_size))
        b = m.add_parameters(hidden_size)
        V = m.add_parameters((output_size, hidden_size))
        a = m.add_parameters(output_size)

        x = dy.vecInput(input_size)
        y = dy.vecInput(output_size)
        h = dy.tanh((W * x) + b)
        output = dy.softmax(V*h)

        x.set(self.word_vectors[word])
        probs = output.npvalue()

        predicted_idx = np.random.choice(
            self.unique_words, p=(probs+0.001)/sum(probs+0.001))

        return predicted_idx

    
    def generation(self, number_of_line):
        start_token = "bos"
        while number_of_line > 0:
            each_line = []
            word = 20
            while word > 0:
                if(word == 20):
                    next_word = self.predict(start_token)
                    each_line.append(next_word)
                    if(next_word == "eol"):
                        break
                    if(next_word == "eos"):
                        return
                else:
                    next_word = self.predict(each_line[-1])
                    each_line.append(next_word)
                    if(next_word == "eol"):
                        break
                    if(next_word == "eos"):
                        return
                word -= 1
            number_of_line -= 1

            each_line = ' '.join([str(word) for word in each_line])
            file.write(each_line+"\n")
        file.write("\n")

    
    # def perplexity(self):
    #     sprob_dict = sprob(sentence)
    #     log_probs_bi = sprob_dict["bigram"]
    
    #     len_bi = len(bigramSProbs)
    #     HW_bi = (-1/len_bi) * log_probs_bi
    #     perplexity_bi = math.exp(HW_bi)
        
    #     perplexity_dict["bigram"] = perplexity_bi

neuralNetwork = FNN()
neuralNetwork.glove_file("./Assignment4/glove.6B/glove.6B.50d.txt")
neuralNetwork.poem_file("./Assignment4/unim_poem.json")

neuralNetwork.bigram()
neuralNetwork.vocabulary()

neuralNetwork.model()

number_of_poems = 5
number_of_lines = 4
while number_of_poems > 0:
    neuralNetwork.generation(number_of_lines)
    number_of_poems -= 1