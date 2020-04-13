import math
import sys
import re
import numpy as np
from collections import Counter

class HMM():
    
    def dataset(self, input_file):
        """Read a BIO data!"""
        rf = open(input_file, 'r')
        lines = []; words = []; labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if (len(line.strip()) == 0 and words[-1] == '.') or not line.strip():
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l, w))
                words = []
                labels = []
            words.append(word)
            labels.append(label)

        return lines


    def HMM(self, dataset):
            
        transition_tags = []
        transition_tag_counts = {}
        transition_prob = {}
        
        for sentence in dataset:
            tags = sentence[0].split()
            
            tags.insert(0, '<s>')
            tags.append('</s>')
            
            tags_bigram = []        
            for number in range(0, len(tags)): 
                bigram = ' '.join(tags[number:number + 2])
                if bigram != '</s>':
                    tags_bigram.append(bigram)
            transition_tags.append(tags_bigram)
        # print(transition_tags)
        
        out_arr = np.asarray(transition_tags)

        for tags in out_arr:
            each_counts = dict(Counter(tags))
        
            for k,v in each_counts.items():
                if k not in transition_tag_counts:
                    transition_tag_counts[k] = v
                else:
                    transition_tag_counts[k] += 1
        print(transition_tag_counts)
        
        
        
        
        emission_prob = {}
        
        

    def viterbi(self):
        return 1


    def accuracy(self):
        return 1

classHMM = HMM()

dataset = classHMM.dataset("./Assignment2/dataset/train.txt")

classHMM.HMM(dataset)