import sys
import re

import math
import random
from collections import defaultdict

import numpy as np
from collections import Counter

class CYK():
    
    def __init__(self):
        self.rules_dict = defaultdict(list) # rules_dict is list that use for generate a random sentences. It's mixed of cfg_rules and vocabulary lists.
    
    
    def rules(self, folder_path):
        cfg_rules = open(folder_path, 'r')
        lines = []
        for line in cfg_rules:
            word = line.strip().split('\t')
            each_char = word[0].split(' ')
            if each_char[0] != '#':
                if each_char[0] != 'ROOT':
                    if line.strip():
                        lines.append(word)
        
        rules = {}
        vocabulary = {}
        for line in lines:
            lhs = line[0].strip()
            rhs = line[1]
            for word in rhs.split():
                word = word.strip()
                if word.islower(): # vocabulary
                    if lhs not in vocabulary:
                        vocabulary[lhs] = [word]
                    else:
                        vocabulary[lhs].append(word)
                else:
                    if lhs not in rules:
                        rules[lhs] = [rhs]
                    else:
                        if rhs not in rules[lhs]:
                            rules[lhs].append(rhs)
                    
        
        self.cfg_rules = rules
        self.cfg_vocabs = vocabulary
        
        for key, value in rules.items():
            for each in value:
                self.rules_dict[key].append(tuple(each.split()))
            
        for key, value in vocabulary.items():
            for each in value:
                self.rules_dict[key].append(tuple(each.split()))
        
        print(self.cfg_rules,"\n",self.cfg_vocabs,"\n",self.rules_dict)
    
    
    def randsentence(self, symbol, output_file):
        sentence = ''
            
        # Step-1: First rule is 'S' key 
        rand_rule = random.choice(self.rules_dict[symbol])

        for each_rule in rand_rule:
            if (len(sentence.split()) > 3) and (len(sentence.split()) < 6):
                break
            # for non-terminals, recurse
            if each_rule in self.rules_dict:
                sentence += self.randsentence(each_rule, output_file)
            else:
                sentence += each_rule + ' '
        
        return sentence
        
        
    def CYKParser(self, generated_sentence):
        sentence_type = []
        for word in generated_sentence.split():
            sentence_type.append([key for key, value in self.cfg_vocabs.items() if word in value])
        
        for i in range(len(sentence_type)):
            sentence_type[i] = sentence_type[i][0]
            
        generated_sentence = generated_sentence.split()
        length = len(generated_sentence)
        cyk_matrix = np.empty((length, length), dtype=object)
        
        sentence_type_dict = {}
        
        for row in range(length):
            index = length - row
            if row == 0:
                for column in range(index):
                    word = generated_sentence[column:row+column+1]
                    t = ' '.join([tag for tag in word if len(tag) > 0])
                    
                    cyk_matrix[row][column] = sentence_type[column]
                    sentence_type_dict[t] = sentence_type[column]
            else:
                for column in range(index):
                    word = generated_sentence[column:row+column+1]
                    t = ' '.join([tag for tag in word if len(tag) > 0])
                    
                    result = []
                    # print(m,",",j,"+",i-(m+1),",",j+m+1)
                    for m in range(row):
                        x1 = cyk_matrix[m][column]
                        x2 = cyk_matrix[row-(m+1)][column+m+1]
                        result_cell = x1 + " " + x2
                        result.append(result_cell)
                    
                    self.result_cell = 'X'
                    for k,v in self.cfg_rules.items():
                        for res in result:
                            if res in v:
                                self.result_cell = k
                        
                    cyk_matrix[row][column] = self.result_cell
        
        return cyk_matrix


classCYK = CYK()

classCYK.rules("./Assignment3/cfg.gr")

file_output = open("output.txt","w")

random_sentences = []
for i in range(10):
    sentence = classCYK.randsentence('S',file_output)
    file_output.write(sentence+"\n")
    random_sentences.append(sentence)

file_output.close()

cyk_parser = []
for rand_sentence in random_sentences:
    cyk_parser.append(classCYK.CYKParser(rand_sentence))

i = 0
for parse in cyk_parser:
    print(random_sentences[i])
    i += 1
    print(parse)
    if 'S' in parse[len(parse)-1][0]:
        print("It's in this language")
    else:
        print("It's not in this language!")