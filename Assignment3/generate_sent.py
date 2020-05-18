import math
import sys

from random import randint

cfg_rules = {   'S': ['NP VP'], 
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


# Random generate sentences        
random_sentence_word_size = 5
        
generated_sentences = []
generated_sentences_type = []

# step-1: S
s_value = cfg_rules['S']
rand_num1 = randint(0,len(s_value)-1)
start_rule = s_value[rand_num1]

# step-other:
random_rules = []

next_rule = []
next_rule.append(start_rule)
for i in range(random_sentence_word_size-1):
    each_rule = next_rule[i]
    print("Rule:", each_rule)
    split_rule = each_rule.split()
    for non_terminal in split_rule:
        if non_terminal in cfg_rules:
            rand_num1 = randint(0,len(cfg_rules[non_terminal])-1)
            rule = cfg_rules[non_terminal][rand_num1]
            each = rule.split()
            for w in each:
                if not w.isupper():
                    random_rules.append(w)
            next_rule.append(rule)
            print(non_terminal, "->", rule)
    print(next_rule)

print(random_rules)
    


#     rand_num1 = randint(0,len(cfg_rules)-1)
#     rand_key = list(cfg_rules.keys())[rand_num1] # According to first random number, row select a word type such as Verb, Noun, etc.

#     rand_num2 = randint(0,len(cfg_rules[rand_key])-1)
#     generated_sentences.append(cfg_rules[rand_key][rand_num2]) # According to second random number, row select a word, which is in rand_key.
#     generated_sentences_type.append(rand_key)
        
# # generated_sentences_with_vocab, first index generated sentences, second index type of word
# generated_sentences_with_vocab = [generated_sentences, generated_sentences_type]
