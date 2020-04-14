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
                lines.append((l.lower(), w.lower()))
                words = []
                labels = []
            words.append(word)
            labels.append(label)

        return lines


    def HMM(self, dataset):
            
        each_tag_counts = {}
        
        transition_tags = []
        transition_tag_counts = {}
        transition_prob = {}
        
        for sentence in dataset:
            tags = sentence[0].split()
            
            tags.insert(0, '<s>')
            
            tags_bigram = []        
            for number in range(0, len(tags)): 
                bigram = ' '.join(tags[number:number + 2])
                if bigram != '</s>':
                    tags_bigram.append(bigram)
            transition_tags.append(tags_bigram)
        
        array_to_npArr = np.asarray(transition_tags)

        for tags in array_to_npArr:
            each_counts = dict(Counter(tags))
        
            for k,v in each_counts.items():
                if k not in transition_tag_counts:
                    transition_tag_counts[k] = v
                else:
                    transition_tag_counts[k] += 1
        # print("transition_tag_counts\n",transition_tag_counts)
        
        for tags in array_to_npArr:
            for tag in tags:
                tag = tag.split()
                each_counts = dict(Counter(tag))
            
                for k,v in each_counts.items():
                    if k not in each_tag_counts:
                        each_tag_counts[k] = v
                    else:
                        each_tag_counts[k] += 1
        # print("each_tag_counts\n",each_tag_counts)
        
        for k,v in transition_tag_counts.items():
            first_tag = k.split()
            first_tag = first_tag[0]
            first_tag_count = each_tag_counts[first_tag]
            trans_prob = transition_tag_counts[k] / each_tag_counts[first_tag]
            transition_prob[k] = math.log2(trans_prob)
        
        # print("transition_prob\n",transition_prob)
        
        emissionTag_Word_dict = {}
        emission_word_counts = {}
        emission_prob = {}
        
        for sentence in dataset:
            tags = sentence[0].split()
            words = sentence[1].split()
            
            for i in range(len(tags)):
                if tags[i] not in emissionTag_Word_dict:
                    arr = [words[i]]
                    emissionTag_Word_dict[tags[i]] = arr
            else:
                emissionTag_Word_dict[tags[i]].append(words[i])
                    
        # print(emissionTag_Word_dict)
        
        for k1,v1 in emissionTag_Word_dict.items():
            array_to_npArr = np.asarray(v1)
            each_counts = dict(Counter(array_to_npArr))
            
            for k2,v2 in each_counts.items():
                if k1 not in emission_word_counts:
                    word_dict = {k2: v2}
                    emission_word_counts[k1] = word_dict
                else:
                    if k2 not in emission_word_counts[k1]:
                        emission_word_counts[k1].update({k2: v2})
                    else:
                        emission_word_counts[k1][k2] += 1
                        
        # print(emission_word_counts)
        
        for k1,v1 in emission_word_counts.items():
            total_corpus = sum(emission_word_counts[k1].values())
            
            em_prob = {}
            for k2,v2 in v1.items():
                emis_prob = v2 / total_corpus
                em_prob[k2] = math.log2(emis_prob)
                
            emission_prob[k1] = em_prob 
        print(emission_prob)
        

    def viterbi(self):
        return 1


    def accuracy(self):
        return 1

classHMM = HMM()

dataset = classHMM.dataset("./Assignment2/dataset/train.txt")

classHMM.HMM(dataset)