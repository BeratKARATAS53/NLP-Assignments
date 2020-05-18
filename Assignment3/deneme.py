import sys
import re

import math
import random
from collections import defaultdict

import numpy as np
from collections import Counter

class CYK():
    
    def __init__(self):
        # rules_dict is a list that uses to generating random sentences. It's mixed with cfg_rules and vocabulary lists.
        self.rules_dict = defaultdict(list)
    
    """
    **Arguments**:
    
        :param folder_path: File path to read cfg_rules
        :type folder_path: A string
        
    **Arg. Example**:
    
        >>> folder_path = ./Assignment3/cfg.gr
        
    **Explanation**:
    
        This function have 3 steps.
        
        >>> Step-1:
            I read the first word from the file the lines 
            that are not '#' and 'ROOT' and write them in the 'lines' array.
        
        >>> Step-2:
            I have both of rules and vocabulary in my 'lines' array. To use it in the CYKParser() function,
            I need to separate them.
            So I perform the separation of each line in a loop by looking at the right-hand-side (RHS) part of the rules. 
            If the RHS part is lowercase, that line represents the vocabulary, so I add that line to the 'vocabulary' dict. 
            In the other case, I add it to the 'rules' dict.
            
        >>> Step-3:
            I combine rules and vocabulary dictionaries into the defaultdict type named rules_dict 
            for easier use in the randsentence() function. I will talk about why it is easier in the randsentence() function.
            
        As a result, I have 3 dict:
            ... rules = {       'VP': ['Verb NP'], 
                                'Noun': ['Adj Noun'], 
                                'S': ['NP VP'], 
                                'PP': ['Prep NP'], 
                                'NP': ['Det Noun', 'Pronoun', 'NP PP']  }
            
            ... vocabulary =  { 'Prep': ['with', 'on', 'under', 'in', 'to', 'from'], 
                                'Verb': ['ate', 'wanted', 'kissed', 'washed', 'pickled', 'is', 'prefer', 'like', 'need', 'want'], 
                                'Noun': ['president', 'sandwich', 'pickle', 'mouse', 'floor'], 
                                'Adj': ['fine', 'delicious', 'beautiful', 'old'], 
                                'Pronoun': ['me', 'i', 'you', 'it'], 
                                'Det': ['the', 'a', 'every', 'this', 'that']    } 
                                
            ... rules_dict =  { 'S': [('NP', 'VP')], 
                                'PP': [('Prep', 'NP')], 
                                'Adj': [('fine',), ('delicious',), ('beautiful',), ('old',)], 
                                'VP': [('Verb', 'NP')], 
                                'NP': [('Det', 'Noun'), ('Pronoun',), ('NP', 'PP')],
                                'Noun': [('Adj', 'Noun'), ('president',), ('sandwich',), ('pickle',), ('mouse',), ('floor',)], 
                                'Verb': [('ate',), ('wanted',), ('kissed',), ('washed',), ('pickled',), ('is',), ('prefer',), ('like',), ('need',), ('want',)], 
                                'Det': [('the',), ('a',), ('every',), ('this',), ('that',)], 
                                'Pronoun': [('me',), ('i',), ('you',), ('it',)],
                                'Prep': [('with',), ('on',), ('under',), ('in',), ('to',), ('from',)]   }
        
    """
    def rules(self, folder_path):
        """ Step-1: Read a cfg rules file. """
        cfg_rules = open(folder_path, 'r')
        lines = []
        for line in cfg_rules:
            word = line.strip().split('\t')
            each_char = word[0].split(' ')
            if each_char[0] != '#':
                if each_char[0] != 'ROOT':
                    if line.strip():
                        lines.append(word)
        
        """ Step-2: Lines split to rules and vocabulary dicts. """
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
                else: # rules
                    if lhs not in rules:
                        rules[lhs] = [rhs]
                    else:
                        if rhs not in rules[lhs]:
                            rules[lhs].append(rhs)
                    
        self.cfg_rules = rules
        self.cfg_vocabs = vocabulary
        
        """ Step-3: I combine rules and vocabulary dict and write to rules_dict. """
        for key, value in rules.items():
            for each in value:
                self.rules_dict[key].append(tuple(each.split()))
            
        for key, value in vocabulary.items():
            for each in value:
                self.rules_dict[key].append(tuple(each.split()))
    
    
    """
    **Arguments**:

        :param symbol: The non-terminals variable such as 'S', 'NP', 'Verb', ...
        :type symbol: A string
        
        :return sentence: A randomly generate sentence
        :type sentence: A string
        
    **Explanation**:
    
        I have defined the randsentence() function is recursive.
        First of all, I take the initial symbol 'S' and select a random terminal value from rules_dict.
        Then I check each word in this terminal value;
        ... If this word is a non-terminal value, I send this value back to the function.
        ... I understand that this value is a word if it is terminal, and I add it to the string 'sentence'.
        
        !   Note: Normally the if block in line 154th should ensure that the number of words in the sentence does not exceed 10. 
            But sometimes it doesn't work.
            The reason is that every time the function is called recursive, the sentence value is reset.
            Despite this, I did not remove it, because 1-2 of the 10 sentences 
            I have created have the number of words of 10 and above.
            If I remove it, only 1-2 out of 10 sentences have the number of 10 and six words I want.
            So even though that part doesn't work at 100% efficiency, I get results close to what I want.
        
        !!  The answer to the question about why I am combining the 'rules' and 'vocabulary' dict 
            I have mentioned while reading the file is that I define the structure of the randsentence () function is recursive.
            So, instead of checking whether the symbol value that comes in every step 
            I call the function recursive is rules or vocabulary, combining these 2 dict has made it easier for me to do the operation.
        
        !!! 
    """
    def randsentence(self, symbol, output_file):
        sentence = ''
            
        # Step-1: The first rule was selected according to the 'S' key
        rand_rule = random.choice(self.rules_dict[symbol])

        for each_rule in rand_rule:
            # The number of words in the sentence is less than 10, but it does work with 90% efficiency.
            if (len(sentence.split()) > 10):
                break
            elif each_rule in self.rules_dict: # If the word I selected is a non-terminal value, send it back to the function.
                sentence += self.randsentence(each_rule, output_file)
            else: # If the word I selected is a terminal value, add to the sentence.
                sentence += each_rule + ' '
                output_file.write(each_rule + " ")
        
        return sentence
        
    """
    **Arguments**:

        :param generate_sentence: A Random Generated Sentence
        :type generate_sentence: str
        
        :return cyk_matrix: The cyk matrix of that sentence
        :return cyk_matrix: 2d np array
        
    **Arg. Example**:
    
        >>> generate_sentence: the mouse on every pickle 
        >>> cyk_matrix:
                [['Det' 'Noun' 'Prep' 'Det' 'Noun']
                ['NP' 'X' 'X' 'NP' None]
                ['X' 'X' 'PP' None None]
                ['X' 'X' None None None]
                ['NP' None None None None]]
                
    **Explanation**:

        >>> Step-1:  ------ [line 279]
            Before starting the CYK parser algorithm, 
            I have to find out which type of each word my sentence belongs to. In this part, 
            I use vocabulary dict.
            And I write the result to the sentence_type array.
        
        >>> Step-2:   ------ [line 294-300]
            I fill the first line of my CYK matrix with the types of my words in the sentence. 
            The reason I have separated the first step is that I will apply a formula in other steps.
            
        >>> Other Steps:
            Now in other steps, I apply the following formula:
            ... Formula: Xrow,column = (Xm,column)(Xrow-(m+1),column+m+1) ---- m: row count spacing [0-row], row: row count [0-count(sentence)], column: colum count [0-count(sentence)]

            ... Formula Example:
                >>> First Row: X1,0 = (X0,0)(X0,1) ---------------------    m: 0, row: 1, column: 0
                >>> Second Row X2,0 = (X0,0)(X0,1) U (X1,0)(X0,2) ----    m: [0-1], row: 2, column: 0
                .
                .
                .
        
    **CYK Parser Visualization**:

    ... sentence: the mouse kissed a mouse 
        >>> row=0 | 'the', 'mouse', 'kissed', 'a', 'mouse'
        >>> row=1 | 'the mouse', 'mouse kissed', 'kissed a', 'a mouse'
        >>> row=2 | 'the mouse kissed', 'mouse kissed a', 'kissed a mouse'
        >>> row=3 | 'the mouse kissed a', 'mouse kissed a mouse'
        >>> row=4 | 'the mouse kissed a mouse'
    
    ------------------------------------------
    ... Step-1: --- row=0                                 
        'the  mouse   kissed  a    mouse'
        'Det   Noun    Verb  Det   Noun'
                     
    ... Step-2: --- row=1
        'the mouse'     - X1,0 = (X0,0)(X0,1) -> (Det)(Noun)  => NP
        'mouse kissed'  - X1,1 = (X0,1)(X0,2) -> (Noun)(Verb) => Empty (X)
        
    Step-1                                            Step-2
       _______                                           _______
    4 |_______|________                               4 |_______|________
    3 |_______|________|________                      3 |_______|________|________
    2 |_______|________|________|_____                2 |_______|________|________|_____ 
    1 |_______|________|________|_____|_______        1 |___NP__|____X___|____X___|_NP__|_______
    0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__|       0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__|
         the    mouse    kissed    a    mouse              the    mouse    kissed    a    mouse
    ------------------------------------------
    *********
    ------------------------------------------
    ... Step-3: --- row=2
        'the mouse kissed'  - X2,0 = (X0,0)(X1,1) U (X1,0)(X0,2) -> (Det)(X) U (NP)(Verb) => Empty (X)
        'mouse kissed a'    - X2,1 = (X0,1)(X1,2) U (X1,1)(X0,3) -> (Noun)(X) U (X)(Det)  => Empty (X)
        
    ... Step-4: --- row=3
        'the mouse kissed a'    - X3,0 = (X0,0)(X2,1) U (X1,0)(X1,2) U (X2,0)(X0,3) -> (Det)(X) U (NP)(X) U (X)(Verb)   => Empty (X)
        'mouse kissed a mouse'  - X3,1 = (X0,1)(X2,2) U (X1,1)(X1,3) U (X2,1)(X0,4) -> (Noun)(VP) U (X)(NP) U (X)(Det)  => Empty (X)
        
     Step-3                                                  Step-4
       _______                                           _______
    4 |_______|________                               4 |_______|________
    3 |_______|________|________                      3 |___X___|___X____|________
    2 |___X___|___X____|___VP___|_____                2 |___X___|___X____|___VP___|_____
    1 |___NP__|___X____|___X____|_NP__|_______        1 |___NP__|____X___|____X___|_NP__|_______
    0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__|       0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__|
         the    mouse    kissed    a    mouse              the    mouse    kissed    a    mouse
    ------------------------------------------
    *********
    ------------------------------------------
    ... Step-5: --- row=4
        'the mouse kissed a mouse'  - X4,0 = (X0,0)(X3,1) U (X1,0)(X2,2) U (X2,0)(X1,3) U (X3,0)(X0,4) -> (Det)(X) U (NP)(VP) U (X)(NP) U (X)(Noun) => S
        
     Step-5
       _______ 
    4 |___S___|________
    3 |___X___|___X____|________ 
    2 |___X___|___X____|___VP___|_____ 
    1 |___NP__|___X____|___X____|_NP__|_______ 
    0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__| 
         the    mouse    kissed    a    mouse      

    ------------------------------------------
         
    This is my cyk_matrix:
        [       0      1      2       3        4
        0    ['Det' 'Noun' 'Verb'   'Det'   'Noun']
        1    ['NP'  'X'     'X'     'NP'     None]
        2    ['X'   'X'     'VP'    None     None]
        3    ['X'   'X'     None    None     None]
        4    ['S'    None   None    None     None]
        ]
    """
    def CYKParser(self, generated_sentence):
        sentence_type = []
        for word in generated_sentence.split(): # Step-1
            sentence_type.append([key for key, value in self.cfg_vocabs.items() if word in value])
        
        for i in range(len(sentence_type)):
            sentence_type[i] = sentence_type[i][0]
            
        # cyk_matrix filling started
        length = len(sentence_type)
        cyk_matrix = np.empty((length, length), dtype=object)
        
        for row in range(length):
            index = length - row
            if row == 0: # Step-2: Filling the first row
                for column in range(index):
                    word = generated_sentence[column:row+column+1]
                    t = ' '.join([tag for tag in word if len(tag) > 0])
                    
                    cyk_matrix[row][column] = sentence_type[column]
            else: # Other Steps
                for column in range(index):
                    word = generated_sentence[column:row+column+1]
                    t = ' '.join([tag for tag in word if len(tag) > 0])
                    
                    result = []
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
    file_output.write("\n")
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