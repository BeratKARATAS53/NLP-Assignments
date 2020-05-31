from nltk import FreqDist
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import re
import random
import math
import numpy as np
import string
import json

import dynet as dy
import dynet_config
dynet_config.set_gpu()

nltk.download('punkt')  # For Generate Bigram


file = open('poems.txt', 'w')


class FNN:
    def __init__(self):
        print("Init")
        self.embeddings = []  # Glove Vectors
        self.word_to_index = {}
        self.index_to_vocab = {}
        self.vocab_to_index = {}
        self.unique_words = []
        self.word_vectors = {}

        self.unim_poem = []
        self.poems = []

        self.bigram_poem = []

    """
    **Arguments**:

        :param folder_path: Folder Path of Glove Vectors
        :type folder_path: A string
        
    **Explanation**:
        I read the Glove vector and write it in 2 different variables:
        >>> 'word_to_index': It keeps the index of each word as a dictionary. For example; {"the": 5}
        >>> 'embeddings': Keeps vector information of each word in 'word_to_index', as a list.
    
    """
    def glove_file(self, folder_path):
        print("Reading Glove...")
        with open(folder_path, 'r', encoding='UTF-8') as file:
            for line in file:
                line = line.split()
                self.word_to_index[line[0]] = len(self.embeddings)
                self.embeddings.append(np.array(line[1:], dtype=np.float32))

    """
    **Arguments**:

        :param folder_path: Folder Path of Poem File
        :type folder_path: A string
        
    **Explanation**:
        I read the poetry data and write it to the 'unim_poem' list:
        Then I send each sentence in the list to the preprocessing() function.
    
    """
    def poem_file(self, folder_path):
        print("Reading Poems...")
        with open(folder_path, 'r') as f:
            self.unim_poem = json.load(f)
            mini_poems = random.sample(self.unim_poem, 100)
            self.preprocessing(mini_poems)

    """
    **Arguments**:

        :param unim_poem: A Poems List
        :type unim_poem: A string
        
    **Explanation**:
        I replace the '\n' characters in each sentence in the poem with 'eol' and 
        put 'bos' as the start token at the beginning of the sentence and 'eos' as the end token at the end.
        Then I add these processed sentences to the 'poems' list.

    """
    def preprocessing(self, unim_poem):
        print("Preproccesing...")
        for each in unim_poem:
            for k, v in each.items():
                if(k == "poem"):
                    each["poem"] = each["poem"].replace("\n", " eol ")
                    each["poem"] = "bos " + each["poem"] + " eos"
                    self.poems.append(each["poem"])

    """        
    **Explanation**:
        This function works with unique words in poetry:
        >>> First of all, it adds vectors of unique words to the 'word_vectors' list.
        >>> Then I need to have index information to access these vectors. 
            In this, I add 'vocab_to_index' to access the index of the word and 
            'index_to_vocab' dictionaries to access the word from the index.
        >>> Finally, I add unique words to the list of 'unique_words'.

    """
    def vocabulary(self):
        print("Unique Words Calculating...")
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

    """ I create a bigram of each poem with the Nltk library. """
    def bigram(self):
        print("Bigram Creating...")
        for each_poem in self.poems:
            nltk_tokens = nltk.word_tokenize(each_poem)
            self.bigram_poem += list(nltk.bigrams(nltk_tokens))

    """ Creating Model """
    def model(self):
        print("Model Creating...")
        hidden_size = 64
        vocabulary_size = len(self.vocab_to_index)
        input_size = output_size = vocabulary_size # Input and Output size are equal to V.

        m = dy.Model()

        W = m.add_parameters((hidden_size, input_size))
        b = m.add_parameters(hidden_size)
        V = m.add_parameters((output_size, hidden_size)) # Softmax weights
        a = m.add_parameters(output_size) # Softmax bias

        x = dy.vecInput(input_size)
        y = dy.vecInput(output_size)
        h = dy.tanh((W * x) + b) 
        output = dy.softmax(V * h) # Softmax

        loss = dy.squared_distance(output, y)
        trainer = dy.SimpleSGDTrainer(m)

        epoch = 1
        for iter in range(epoch):
            my_loss = 0.0
            seen_instances = 0
            for binary_word in self.bigram_poem:
                x.set(self.word_vectors[binary_word[0]])
                y.set(self.word_vectors[binary_word[1]])
                seen_instances += 1

                my_loss += loss.scalar_value()
                loss.forward()
                loss.backward()
                trainer.update()

                if (seen_instances > 1 and seen_instances % 100 == 0):
                    print(seen_instances, "/", len(self.bigram_poem),
                          "***average loss is:", my_loss / seen_instances)

            print(my_loss / seen_instances)

    """
    **Arguments**:

        :param word: The word used to predict the next word
        :type word: A string
        
        :return predicted_word: Predicted word
        :type predicted_word: A string
        
    **Explanation**:
        Due to the bigram model, each word depends on the word before it.
        So I guess the next word according to the word I read from the parameter in the predict function.
        Finally, I return the predicted word.

    """
    def predict(self, word):
        hidden_size = 64
        vocabulary_size = len(self.vocab_to_index)
        input_size = output_size = vocabulary_size

        m = dy.Model()

        W = m.add_parameters((hidden_size, input_size))
        b = m.add_parameters(hidden_size)
        V = m.add_parameters((output_size, hidden_size)) # Softmax weights
        a = m.add_parameters(output_size) # Softmax bias

        x = dy.vecInput(input_size)
        y = dy.vecInput(output_size)
        h = dy.tanh((W * x) + b)
        output = dy.softmax(V * h)

        x.set(self.word_vectors[word])
        probabilities = output.npvalue()

        predicted_word = np.random.choice(
            self.unique_words, p=(probabilities+0.0002)/sum(probabilities+0.0002))

        return predicted_word

    """
    **Arguments**:

        :param number_of_line: Number of lines in each poem
        :type number_of_line: An int
        
    **Explanation**:
        I created a while loop according to the number of lines in the poem, which is the parameter.
        Then I created another while loop with a maximum of 50 words in each line.
        Because in some cases, eol, which is the end of line token, does not come. In such cases, poetry does not end.
        To prevent this, I limited each line to 50 words.
        Then I use the predict function to guess the next word.
        If the predicted word is eol, I finish that line, or if eos, I finish that poem.
        And I write all these poems in the poems.txt file.
    """
    def generation(self, number_of_line):
        start_token = "bos"
        file.write(start_token+" ")
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

    def perplexity(self):
        pass


neuralNetwork = FNN()
neuralNetwork.glove_file("./Assignment4/glove.6B/glove.6B.50d.txt")
neuralNetwork.poem_file("./Assignment4/unim_poem.json")

neuralNetwork.bigram()
neuralNetwork.vocabulary()

neuralNetwork.model()

number_of_poems = 5
number_of_lines = 4
while number_of_poems > 0:
    print("Generating Poem...")
    neuralNetwork.generation(number_of_lines)
    number_of_poems -= 1
