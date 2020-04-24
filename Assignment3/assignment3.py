import sys
import re

import math
import random
from collections import defaultdict

import numpy as np
from collections import Counter

class CYK():
    
    def __init__(self):
        self.rules_dict = defaultdict(list)
        
    mix_rules_and_vocab = {     'S': ['NP VP'], 
                                'VP': ['Verb NP'], 
                                'NP': ['Det Noun | Pronoun | NP PP'],
                                'PP': ['Prep NP'], 
                                'Noun': ['Adj Noun'],
                                'Verb': ['ate | wanted | kissed | washed | pickled | is | prefer | like | need | want',],
                                'Det': ['the | a | every | this | that'],
                                'Noun': ['president | sandwich | pickle | mouse | floor'],
                                'Adj': ['fine | delicious | beautiful | old'],
                                'Prep': ['with | on | under | in | to | from'],
                                'Pronoun': ['me | I | you | it'] 
                            }
    
    cfg_rules = {   'S': ['NP VP'], 
                    'VP': ['Verb NP'], 
                    'NP': ['Det Noun', 'Pronoun', 'NP PP'],
                    'PP': ['Prep NP'], 
                    'Noun': ['Adj Noun'] }
    
    vocabulary = {  'Verb': ['ate', 'wanted', 'kissed', 'washed', 'pickled', 'is', 'prefer', 'like', 'need', 'want',],
                    'Det': ['the', 'a', 'every', 'this', 'that'],
                    'Noun': ['president', 'sandwich', 'pickle', 'mouse', 'floor'],
                    'Adj': ['fine', 'delicious', 'beautiful', 'old'],
                    'Prep': ['with', 'on', 'under', 'in', 'to', 'from'],
                    'Pronoun': ['me', 'I', 'you', 'it'] }
    
    def cfg_rules_change(self, cfg_rules):
        for k,v in cfg_rules.items():
            rhs = v[0].split("|")
            for each in rhs:
                self.rules_dict[k].append(tuple(each.split()))
    
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
        
    """
    Example:
    >>> row=1 | 'sandwich', 'like', 'fine', 'beautiful', 'mouse'
    >>> row=2 | 'sandwich like', 'like fine', 'fine beautiful', 'beautiful mouse'
    >>> row=3 | 'sandwich like fine', 'like fine beautiful', 'fine beautiful mouse'
    >>> row=4 | 'sandwich like fine beautiful', 'like fine beautiful mouse'
    >>> row=5 | 'sandwich like fine beautiful mouse'
    
    
    CYK PARSER VISUALIZATION
    'sandwich like fine beautiful mouse'
    'Noun     Verb Adj  Adj       Noun'
    
    Step-1                                                  Step-2
       _________                                                _________
    5 |_________|______                                      5 |_________|______
    4 |_________|______|______                               4 |_________|______|______
    3 |_________|______|______|____________                  3 |_________|______|______|____________ 
    2 |____X____|__X___|__X___|____Noun____|_________        2 |____X____|__X___|__X___|____Noun____|_________
    1 |___Noun__|_Verb_|_Adj__|____Adj_____|__Noun___|       1 |___Noun__|_Verb_|_Adj__|____Adj_____|__Noun___|
       sandwich   like   fine   beautiful     mouse             sandwich   like   fine   beautiful     mouse
         
     Step-3                                                  Step-4
       _________                                                _________
    5 |_________|______                                      5 |_________|______
    4 |_________|______|______                               4 |____X____|__X___|______
    3 |____X____|__X___|__X___|____________                  3 |____X____|__X___|__X___|____________
    2 |____X____|__X___|__X___|____Noun____|_________        2 |____X____|__X___|__X___|____Noun____|_________
    1 |___Noun__|_Verb_|_Adj__|____Adj_____|__Noun___|       1 |___Noun__|_Verb_|_Adj__|____Adj_____|__Noun___|
       sandwich   like   fine   beautiful     mouse             sandwich   like   fine   beautiful     mouse
         
     Step-5
       _________
    5 |____X____|______
    4 |____X____|__X___|______
    3 |____X____|__X___|__X___|____________
    2 |____X____|__X___|__X___|____Noun____|_________
    1 |___Noun__|_Verb_|_Adj__|____Adj_____|__Noun___|
       sandwich   like   fine   beautiful     mouse       
         
         
    This is my cyk_matrix:
    
    0   Noun     Verb   Adj     Adj     Noun
    1     x        x     x     Noun
    2     x        x     x   
    3     x        x  
    4     x 
          0        1     2       3       4   
    """
    def CYKParser(self, generated_sentence):
        sentence_type = []
        for word in generated_sentence.split():
            sentence_type.append([key for key, value in self.vocabulary.items() if word in value])
        
        for i in range(len(sentence_type)):
            sentence_type[i] = sentence_type[i][0]
            
        generated_sentence = generated_sentence.split()
        length = len(generated_sentence)
        cyk_matrix = np.empty((length, length), dtype=object)
        
        print(generated_sentence)
        
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
                """ Formula: Xrow,column = (row-1,column),(row-1,column+1) 
                    >>> X1,0 = (X0,0),(X0,1)
                    Formula: Xrow,column = (Xrow-2,column),(row-1,column+1) U (Xrow-1,column),(Xrow-2,column+1) 
                    >>> X2,0 = (X1,0),(X0,1) U (X0,0),(X1,1)
                    Formula: Xrow,column = (Xrow-3,column),(row-1,column+1) U (Xrow-2,column),(Xrow-2,column+2) U (Xrow-1,column),(Xrow-3,column+3) 
                    >>> X3,0 = (X0,0),(X2,1) U (X1,0),(X1,2) U (X2,0),(X0,3)
                    Formula: Xrow,column = (Xrow-4,column),(row-1,column+1) U (Xrow-3,column),(Xrow-2,column+2) 
                                            U 
                                           (Xrow-2,column),(Xrow-3,column+3) U (Xrow-1,column),(Xrow-4,column+3) 
                    >>> X4,0 = (X0,0),(X3,1) U (X1,0),(X2,2) U (X2,0),(X1,3) U (X3,0),(X0,4)
                """
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

cfg_rules = classCYK.cfg_rules_change(classCYK.mix_rules_and_vocab)

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

for parse in cyk_parser:
    print(parse)
    if 'S' in parse[len(parse)-1][0]:
        print("It's in this language")
    else:
        print("It's not in this language!")


    