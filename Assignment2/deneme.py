import numpy as np
from collections import Counter

import math
import sys
import re

def dataset(input_file):
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


def HMM(dataset):
    
    # all_tags = [] # Tüm Tag'ler
    # all_tags_counts = {} # Tüm Tag'lerin Countları
    # all_tags_prob = {} # Tüm Tag'lerin Olasılıkları
    
    # for sentence in dataset:
    #     tags = sentence[0].split()
    #     for tag in tags:
    #         all_tags.append(tag)
    
    # for tag in all_tags:
    #     if tag not in all_tags_counts:
    #         all_tags_counts[tag] = all_tags.count(tag)
    
    # for tag_c in all_tags_counts:
    #     if tag_c not in all_tags_prob:
    #         all_tags_prob[tag_c] = all_tags_counts[tag_c] / len(all_tags)
            
    # all_words = [] # Tüm Kelimeler
    # all_words_counts = {} # Tüm Kelimelerin Countları
    # all_words_prob = {} # Tüm Kelimelerin Olasılıkları
    
    # for sentence in dataset:
    #     words = sentence[1].split()
    #     for word in words:
    #         word = word.lower()
    #         all_words.append(word)
    
    # for word in all_words:
    #     if word not in all_words_counts:
    #         all_words_counts[word] = all_words.count(word)
    
    # for word_c in all_words_counts:
    #     if word_c not in all_words_prob:
    #         all_words_prob[word_c] = all_words_counts[word_c] / len(all_words)
    # print(all_words_prob)
    
    # initial_tags = [] # Başlangıç Tag'leri
    # initial_tag_counts = {} # Başlangıç Tag'lerinin Countları
    # initial_prob = {} # Başlangıç Tag'lerinin Olasılıkları
    
    # for sentence in dataset:
    #     start_tag = sentence[0].split()[0]
    #     initial_tags.append(start_tag)
    
    # for i_tag in initial_tags:
    #     if i_tag not in initial_tag_counts:
    #         initial_tag_counts[i_tag] = initial_tags.count(i_tag)
    # print(initial_tag_counts)
    
    # for i_tag_c in initial_tag_counts:
    #     if i_tag_c not in initial_prob:
    #         initial_prob[i_tag_c] = initial_tag_counts[i_tag_c] / len(initial_tags)
    # print(initial_prob)
    
        
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
    print(transition_tags)
    
    # for tags in transition_tags:
    #     for i in range(len(sentence) - 1):
    #         temp = (sentence[i], sentence[i+1])
    #         listToString = ' '.join([str(elem) for elem in temp])
    #         if listToString in tags:
    #             if listToString not in transition_tag_counts:
    #                 transition_tag_counts[listToString] = 1
    #             else:
    #                 transition_tag_counts[listToString] += 1
    
    out_arr = np.asarray(transition_tags)

    for tags in out_arr:
        each_counts = dict(Counter(tags))
    
        for k,v in each_counts.items():
            if k not in transition_tag_counts:
                transition_tag_counts[k] = v
            else:
                transition_tag_counts[k] += 1
    print(transition_tag_counts)
    # # Calculating the probabilities of each word according to the bigram and write to the bigramProbs dictionary.
    # for k,v in transition_tag_counts.items():
    #     first_word = k.split()
    #     first_word = first_word[0]
    #     first_word_count = all_tags_counts[first_word]
    #     bi_probs = transition_tag_counts[k] / all_tags_counts[first_word]
    #     bigramProbs[k] = bi_probs
    
    
    
    emission_prob = {}
    
    
    return 1


def viterbi():
    return 1


def accuracy():
    return 1


dataset = dataset("./Assignment2/dataset/train.txt")
# print(dataset)
HMM(dataset)