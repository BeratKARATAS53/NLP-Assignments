import math
import random
from collections import defaultdict

# arr = ["sentence","like","THAT","IN","This"]
# arr2 = ["11","22","33","44","55"]

# for i in range(len(arr)):
#     index = len(arr) - i
#     for j in range(index):
#         word = arr[j:i+j+1]
#         t = ' '.join([tag for tag in arr[j:i+j+1] if len(tag) > 0])
#         print(t)
#         if i > 0:
#             print(i,",",j,":")
#             l = 0
#             for m in range(i):
#                 print(m,",",j,"+",i-(m+1),",",j+m+1)


# cfg_rules = {   'S': ['NP VP'], 
#                 'VP': ['Verb NP'], 
#                 'NP': ['Det Noun', 'Pronoun', 'NP PP'],
#                 'PP': ['Prep NP'], 
#                 'Noun': ['Adj Noun'] }

# key_list = list(cfg_rules.keys())
# for i in range(5):
#     if not key_list[i].isupper():
#         print(key_list[i])
# next_rule = ['NP VP']
# print([[word for word in rule if not word.isupper()] for rule in next_rule])
# print([sum([1 for word in rule if not word.isupper()]) for rule in next_rule])
# next_rule = ['NP VP', 'NP PP']
# print([[word for word in rule if not word.isupper()] for rule in next_rule])
# print([sum([word.isupper() for word in rule]) for rule in next_rule])
# next_rule = ['NP VP', 'NP PP', 'Verb NP']
# print([[word for word in rule if not word.isupper()] for rule in next_rule])
# print([sum([word.isupper() for word in rule]) for rule in next_rule])
# next_rule = ['NP VP', 'NP PP', 'Verb NP', 'Det Noun']
# print([[word for word in rule if not word.isupper()] for rule in next_rule])
# print([sum([word.isupper() for word in rule]) for rule in next_rule])

cfg_rules = {      'S': ['NP VP'], 
                    'VP': ['Verb NP'], 
                    'NP': ['Det Noun | Pronoun | NP PP'],
                    'PP': ['Prep NP'], 
                    'Noun': ['Adj Noun'],
                    'Verb': ['ate | wanted | kissed | washed | pickled | is | prefer | like | need | want',],
                    'Det': ['the | a | every | this | that'],
                    'Noun': ['president | sandwich | pickle | mouse | floor'],
                    'Adj': ['fine | delicious | beautiful | old'],
                    'Prep': ['with | on | under | in | to | from'],
                    'Pronoun': ['me | I | you | it'] }


class CFG(object):
    def __init__(self):
        self.prod = defaultdict(list)

    def add_prod(self, lhs, rhs):
        """ Add production to the grammar. 'rhs' can
            be several productions separated by '|'.
            Each production is a sequence of symbols
            separated by whitespace.

            Usage:
                grammar.add_prod('NT', 'VP PP')
                grammar.add_prod('Digit', '1|2|3|4')
        """
        prods = rhs[0].split("|")
        for prod in prods:
            self.prod[lhs].append(tuple(prod.split()))
        print(self.prod)
    def gen_random(self, symbol):
        """ Generate a random sentence from the
            grammar, starting with the given
            symbol.
        """
        sentence = ''

        # select one production of this symbol randomly
        rand_prod = random.choice(self.prod[symbol])

        for sym in rand_prod:
            # for non-terminals, recurse
            if sym in self.prod:
                sentence += self.gen_random(sym)
            else:
                sentence += sym + ' '

        return sentence

cfg1 = CFG()
for k,v in cfg_rules.items():
    cfg1.add_prod(k,v)
    
# cfg1.add_prod(cfg_rules)
# # cfg1.add_prod('S', 'NP VP')
# # cfg1.add_prod('NP', 'Det N | Det N')
# # cfg1.add_prod('NP', 'I | he | she | Joe')
# # cfg1.add_prod('VP', 'V NP | VP')
# # cfg1.add_prod('Det', 'a | the | my | his')
# # cfg1.add_prod('N', 'elephant | cat | jeans | suit')
# # cfg1.add_prod('V', 'kicked | followed | shot')

for i in range(10):
    print(cfg1.gen_random('S'))