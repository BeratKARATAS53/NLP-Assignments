import math
import sys
import re

import numpy as np
from collections import Counter

class HMM():
    
    datasets = []
    def dataset(self, input_folder):
        """Read a Train data!"""
        train_data = open(input_folder+"train.txt", 'r')
        lines = []; words = []; tags = []
        for line in train_data:
            word = line.strip().split(' ')[0]
            tag = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if not line.strip():
                t = ' '.join([tag for tag in tags if len(tag) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((t, w.lower()))
                words = []
                tags = []
            words.append(word)
            tags.append(tag)

        self.datasets.append(lines)
        """Read a Train data!"""
        test_data = open(input_folder+"test.txt", 'r')
        lines = []; words = []; tags = []
        for line in test_data:
            word = line.strip().split(' ')[0]
            tag = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if not line.strip():
                t = ' '.join([tag for tag in tags if len(tag) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((t, w.lower()))
                words = []
                tags = []
            words.append(word)
            tags.append(tag)
            
        self.datasets.append(lines)


    def HMM(self, dataset):
            
        each_tag_counts = {}
        
        transition_tags = []
        transition_tag_counts = {}
        transition_prob = {}
        
        for sentence in dataset:
            tags = sentence[0].split()
            
            tags.insert(0, '<s>')
            tags.append('</s>')
            # print(tags)
            
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
            transition_prob[k] = trans_prob
        
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
                em_prob[k2] = emis_prob
                
            emission_prob[k1] = em_prob
            
        # print(emission_prob)
        
        return transition_prob, emission_prob

    
    def viterbi(self, transition_prob, emission_prob, test_sentences):
        transition_tags = list(emission_prob.keys())
        
        transition_tags.sort()
        transition_tags.insert(0,"<s>")
        print("Unique Transition Tags: ",transition_tags)

        transition_matrix_len = len(transition_tags)

        transition_matrix = np.ones((transition_matrix_len-1,transition_matrix_len))

        for i in range(transition_matrix_len):
            uniqe_tag = transition_tags[i]
            for j in range((transition_matrix_len)-1):
                tag = uniqe_tag + " " + transition_tags[j+1]
                if tag in transition_prob:
                    transition_matrix[j][i] = transition_prob[tag]
        
        # print("Transition:\n",transition_tags,"\n",transition_matrix)
        
        emission_tags = list(emission_prob.keys())
        emission_tags.sort()
        # print("Unique Emission Tags: ",emission_tags)

        row_count = len(emission_tags)
        
        # print("Emission:\n",emission_tags)
        for sentences in test_sentences:
            sentences = sentences.split()
            column_count = len(sentences)
            emission_matrix = np.zeros((row_count,column_count))
            for word in sentences:
                i = sentences.index(word)
                for tag in emission_tags:
                    word = word.lower()
                    if word in emission_prob[tag]:
                        j = emission_tags.index(tag)
                        emission_matrix[j][i] = emission_prob[tag][word]
                                    
            # print(sentences, "\n", emission_matrix)
            
            
            # Start State
            viterbi_matrix = np.zeros((row_count,column_count+2))
            for tag in emission_tags:
                i = emission_tags.index(tag)
                viterbi_matrix[i][0] = transition_matrix[i][0]
            
            # print("Viterbi:\n", viterbi_matrix)
            for word in sentences:
                i = sentences.index(word)
                i = i + 1
                for tag in emission_tags:
                    j = emission_tags.index(tag)
                    each_cell = []
                    if emission_matrix[j][i-1] != 0:
                        for k in range(len(emission_tags)):
                            if viterbi_matrix[k][i-1] != 0.0:
                                # print("Transition: ",transition_tags[k],"-",transition_tags[j+1])
                                # print("Emission: ",emission_tags[j],"-",word)
                                # print("viterbi: ",viterbi_matrix[k][i-1])
                                # print("transition: ",transition_matrix[k][j+1])
                                # print("emission: ",emission_matrix[j][i-1])
                                result = viterbi_matrix[k][i-1] * transition_matrix[k][j+1] * emission_matrix[j][i-1] # (?)
                                each_cell.append(result)
                                    
                            else:
                                each_cell.append(0.0)
                    else:
                        each_cell.append(0.0)
                        
                    viterbi_matrix[j][i] = max(each_cell)
            print(sentences, "\n",viterbi_matrix)

    def accuracy(self):
        return 1

classHMM = HMM()

classHMM.dataset("./Assignment2/dataset/")

train = classHMM.datasets[0]
test = classHMM.datasets[1]

model = classHMM.HMM(train)

test_sentences = [sentences[1] for sentences in test]

viterbi = classHMM.viterbi(model[0], model[1], test_sentences)