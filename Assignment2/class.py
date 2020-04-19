import math
import sys
import re

import numpy as np
from collections import Counter

class HMM():
    
    datasets = []
    V_size = 0
    
    def dataset(self, input_folder):
        print("Read a Train data!")
        train_data = open(input_folder+"train.txt", 'r')
        lines = []; words = []; tags = []
        for line in train_data:
            word = line.strip().split(' ')[0]
            tag = line.strip().split(' ')[-1]
            
            if word != "-DOCSTART-":
                if not line.strip():
                    t = ' '.join([tag for tag in tags if len(tag) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append((t, w))
                    words = []
                    tags = []
                words.append(word)
                tags.append(tag)

        self.datasets.append(lines)
        print("Read a Test data!")
        test_data = open(input_folder+"test.txt", 'r')
        lines = []; words = []; tags = []
        for line in test_data:
            word = line.strip().split(' ')[0]
            tag = line.strip().split(' ')[-1]
            
            if word != "-DOCSTART-":
                if not line.strip():
                    t = ' '.join([tag for tag in tags if len(tag) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append((t,w))
                    words = []
                    tags = []
                words.append(word)
                tags.append(tag)
            
        self.datasets.append(lines)


    def HMM(self, dataset):
        print("Transition Calculating!")
        each_tag_counts = {}
        
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
        
        print("Emission Calculating!")
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
            self.V_size += len(v1)
            
        # print(emission_prob)
        
        return transition_prob, emission_prob

    
    def viterbi(self, transition_prob, emission_prob, test_data):
        print("Viterbi Calculating!")
        test_sentences_tags = [sentences[0] for sentences in test_data]
        
        transition_tags = list(emission_prob.keys())
        
        transition_tags.sort()
        transition_tags.insert(0,"<s>")
        print("Unique Transition Tags: ",transition_tags)

        transition_matrix_len = len(transition_tags)

        transition_matrix = np.zeros((transition_matrix_len-1,transition_matrix_len))

        for i in range(transition_matrix_len):
            uniqe_tag = transition_tags[i]
            for j in range((transition_matrix_len)-1):
                tag = uniqe_tag + " " + transition_tags[j+1]
                if tag in transition_prob:
                    transition_matrix[j][i] = transition_prob[tag]
        
        # print("Transition:\n",transition_tags,"\n",transition_matrix)
        
        emission_tags = list(emission_prob.keys())
        emission_tags.sort()
        print("Unique Emission Tags: ",emission_tags)

        row_count = len(emission_tags)
        
        predict_tags = []
        # print("Emission:\n",emission_tags)
        for test_sentences in test_data:
            sentences = test_sentences[1]
            sentences = sentences.split()
            
            column_count = len(sentences)
            emission_matrix = np.zeros((row_count,column_count))
            for word in sentences:
                i = sentences.index(word)
                for tag in emission_tags:
                    j = emission_tags.index(tag)
                    if word in emission_prob[tag]:
                        emission_matrix[j][i] = emission_prob[tag][word]
                    else:
                        emission_prob[tag][word] = 1 / (len(sentences) + self.V_size)
                        emission_matrix[j][i] = emission_prob[tag][word]
                                    
            # print(sentences, "\n", emission_matrix)
            
            tag_path_array = []
            # Start State
            viterbi_matrix = np.zeros((row_count,column_count+2))
            for tag in emission_tags:
                i = emission_tags.index(tag)
                viterbi_matrix[i][0] = transition_matrix[i][0]
            
            tag_path_array.append(emission_tags[np.argmax(viterbi_matrix[:,0])])
            
            # print("Viterbi:\n", viterbi_matrix)
            for word in sentences:
                i = sentences.index(word)
                i = i + 1
                for tag in emission_tags:
                    j = emission_tags.index(tag)
                    each_cell = np.zeros(len(emission_tags))
                    if emission_matrix[j][i-1] != 0:
                        for k in range(len(emission_tags)):
                            if viterbi_matrix[k][i-1] != 0.0:
                                if transition_matrix[j][k+1] != 0.0:
                                        
                                    # print("Transition: ",transition_tags[k+1],"-",transition_tags[j+1],
                                        #     ", Emission: ",emission_tags[j],"-",word,
                                        #     ", viterbi: ",viterbi_matrix[k][i-1],
                                        #     ", transition: ",transition_matrix[j][k+1],
                                        #     ", emission: ",emission_matrix[j][i-1])
                                        
                                    result = viterbi_matrix[k][i-1] * transition_matrix[j][k+1] * emission_matrix[j][i-1]
                                    each_cell[k] = result
                        
                    viterbi_matrix[j][i] = max(each_cell)
                    
            end_result = np.zeros(len(emission_tags))
            for tag in emission_tags:
                i = emission_tags.index(tag)
                result = viterbi_matrix[i][len(viterbi_matrix[0])-2] * transition_matrix[len(transition_matrix)-1][i+1]
                
                # print("viterbi: ",viterbi_matrix[i][len(viterbi_matrix[0])-2])
                # print("transition: ",transition_matrix[i+1][len(transition_matrix[0])-1] )
                
                end_result[i]
                
            viterbi_matrix[viterbi_matrix == 0] = -1
            
            for i in range(len(viterbi_matrix[0])-2):
                argmax = np.argmax(viterbi_matrix[:,i+1])
                if argmax == 0:
                    argmax = 8
                tag_path_array.append(emission_tags[argmax])
            
            viterbi_matrix[len(viterbi_matrix)-1][len(viterbi_matrix[0])-1] = max(end_result)
            
            predict_tags.append(tag_path_array)
            
            # print(sentences," - ",tag_path_array, "\n",viterbi_matrix)
            
        self.accuracy(test_sentences_tags, predict_tags)
            
            

    def accuracy(self, test_sentences_tags, predict_tags):
        print("Accuracy Calculating")      
        total_match_tag = 0
        total_tags = 0
        
        file = open('submission.txt', 'w')
        file.write("Id,Category\n")
        
        index=1
        for i in range(len(test_sentences_tags)):
            test_sentences_tags[i] = test_sentences_tags[i].split()
            test_sentences_tags[i] = np.asarray(test_sentences_tags[i])
            
            predict_tags[i] = np.asarray(predict_tags[i])
            predict_tags[i] = predict_tags[i][1:]
            
            for x in range(len(predict_tags[i])):
                file.write(str(index)+","+predict_tags[i][x]+"\n")
                index += 1
                
            # testSent = ' '.join([str(elem) for elem in test_sentences_tags[i]])
            # predict = ' '.join([str(elem) for elem in predict_tags[i]])
            # file.write(testSent+"\n"+predict+"\n----------\n")
            
            total_match_tag += np.sum(test_sentences_tags[i] == predict_tags[i])
            total_tags += len(test_sentences_tags[i])
                    
        print("Match:",total_match_tag)
        print("Total:",total_tags)
        print("Accuracy:",total_match_tag/total_tags)

classHMM = HMM()

classHMM.dataset("./Assignment2/dataset/")

train = classHMM.datasets[0]
test = classHMM.datasets[1]

model = classHMM.HMM(train)

# test_sentences = [sentences[1] for sentences in test]

viterbi = classHMM.viterbi(model[0], model[1], test)