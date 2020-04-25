import sys

class cyk():
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
                    
        print(rules,"\n",vocabulary)
        
        cfg_rules_vocab_set = {}
        
        for key, value in rules.items():
            rules[key] = [' | '.join(value[0:len(value)])]
            cfg_rules_vocab_set[key] = rules[key]
            
        for key, value in vocabulary.items():
            vocabulary[key] = [' | '.join(value[0:len(value)])]
            cfg_rules_vocab_set[key] = vocabulary[key]
            
        print(rules,"\n",vocabulary,"\n",cfg_rules_vocab_set)
        return lines

clas = cyk()
lines = clas.rules("./Assignment3/cfg.gr")
