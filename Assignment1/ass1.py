import math
import random
import operator
import bisect

import sys

import re
import json

file = open('perplexity.txt', 'w')

def dataset(folderPath): # Read Dataset from this folderpath
    print("Reading Dataset...")
    
    dataset_list = []
    for word in open(folderPath):
        token = word.strip()
        if token:
            
            token = processData(token)
            
            listToString = ' '.join([str(elem) for elem in token]) 
            dataset_list.append(listToString)
            
    return dataset_list

def processData(data): # Process every single line by 're' library
    data = data.lower()
    data = data.replace("|", " ")
    data = re.sub(r'[^A-Za-z. _-]', '', data)
    
    result = data.split()
    result.insert(0, '<s>')
    result.append('</s>')
    
    return result


total_corpus = []

def Ngram(n): # NGram Models
    print("Creating",n,"gram Model...")
    ngrams_list = []
    
    corpus = 0
    for data in dataset_list:
        data = data.split()
        corpus += len(data)
        sentence_ngram = []
        for number in range(0, len(data)):
            ngram = ' '.join(data[number:number + n])
            sentence_ngram.append(ngram)
        ngrams_list.append(sentence_ngram)
        
    total_corpus.append(corpus)
    
    return ngrams_list

unigramCounts = {}
bigramCounts = {}
trigramCounts = {}
    
unigramProbs = {}
bigramProbs = {}
trigramProbs = {}

log_dict = {}
    
def prob(sentence):
    print("Calculating Probabilities...")
    
    if type(sentence) == str:
        sentence = sentence.split()
    
    print("Calculate Unigram...")
    for word in sentence:
        unigramCounts[word] = sum([data.count(word) for data in unigram_list])
    
    for k,v in unigramCounts.items():
        first_word = k
        first_word_count = unigramCounts[first_word]
        uni_prob = unigramCounts[first_word] / total_corpus[0]
        unigramProbs[first_word] = uni_prob
    
    print("Calculate Bigram...")
    for data in bigram_list:
        for i in range(len(sentence) - 1):
            temp = (sentence[i], sentence[i+1])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in bigramCounts:
                    bigramCounts[listToString] = 1
                else:
                    bigramCounts[listToString] += 1
    
    for k,v in bigramCounts.items():
        first_word = k.split()
        first_word = first_word[0]
        first_word_count = unigramCounts[first_word]
        bi_probs = bigramCounts[k] / unigramCounts[first_word]
        bigramProbs[k] = bi_probs
    
    print("Calculate Trigram...")
    for data in trigram_list:
        for i in range(len(sentence) - 2):
            temp = (sentence[i], sentence[i+1], sentence[i+2])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in trigramCounts:
                    trigramCounts[listToString] = 1
                else:
                    trigramCounts[listToString] += 1
        
    for k,v in trigramCounts.items():
        first_word = k.split()
        first_two_word = (first_word[0], first_word[1])
        listToString = ' '.join([str(elem) for elem in first_two_word]) 
            
        first_two_word_count = bigramCounts[listToString]
        tri_probs = trigramCounts[k] / bigramCounts[listToString]
        trigramProbs[k] = tri_probs
    
    print("Calculating Logarithm Operations...")
    log_unigram = 0
    for k,v in unigramProbs.items():
        log_unigram += math.log2(v)
    
    log_bigram = 0
    for k,v in bigramProbs.items():
        log_bigram += math.log2(v)
    
    log_trigram = 0
    for k,v in trigramProbs.items():
        log_trigram += math.log2(v)
        
    log_dict["unigram"] = log_unigram
    log_dict["bigram"] = log_bigram
    log_dict["trigram"] = log_trigram
    
    return log_dict


unigramSProbs = {}
bigramSProbs = {}
trigramSProbs = {}

logS_dict = {}

def sprob(sentence):
    print("Calculating Smooth Probabilities...")
    
    if type(sentence) == str:
        sentence = sentence.split()
    
    unigramCounts.clear()
    bigramCounts.clear()
    trigramCounts.clear()
    
    unigram_unique_count = {}
                
    for text in unigram_list:
        for word in sentence:
            if word in text:
                if word not in unigramCounts:
                    unigramCounts[word] = 1
                    unigram_unique_count[word] = 1
                else:
                    unigramCounts[word] += 1
    
    unique_word_count = sum(unigram_unique_count.values())
    
    print("Calculate Unigram...")
    for k,v in unigramCounts.items():
        first_word = k
        first_word_count = unigramCounts[first_word]
        uni_prob = (unigramCounts[first_word] + 1) / (total_corpus[0] + unique_word_count) # Smooth: P(x) = C(x) + 1 / C(total) + V
        unigramSProbs[first_word] = uni_prob
    
    print("Calculate Bigram...")                    
    for data in bigram_list:
        for i in range(len(sentence) - 1):
            temp = (sentence[i], sentence[i+1])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in bigramCounts:
                    bigramCounts[listToString] = 1
                else:
                    bigramCounts[listToString] += 1
            else:
                bigramCounts[listToString] = 0
    
    for k,v in bigramCounts.items():
        first_word = k.split()
        first_word = first_word[0]
        first_word_count = unigramCounts[first_word]
        bi_probs = (bigramCounts[k] + 1) / (unigramCounts[first_word] + unique_word_count)
        bigramSProbs[k] = bi_probs
    
    print("Calculate Trigram...")
    for data in trigram_list:
        for i in range(len(sentence) - 2):
            temp = (sentence[i], sentence[i+1], sentence[i+2])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in trigramCounts:
                    trigramCounts[listToString] = 1
                else:
                    trigramCounts[listToString] += 1
            else:
                trigramCounts[listToString] = 0
        
    for k,v in trigramCounts.items():
        first_word = k.split()
        first_two_word = (first_word[0], first_word[1])
        listToString = ' '.join([str(elem) for elem in first_two_word]) 
            
        first_two_word_count = bigramCounts[listToString]
        tri_probs = (trigramCounts[k] + 1) / (bigramCounts[listToString] + unique_word_count)
        trigramSProbs[k] = tri_probs
    
    print("Calculating Logarithm Operations...")
    logS_unigram = 0
    for k,v in unigramSProbs.items():
        logS_unigram += math.log2(v)
    
    logS_bigram = 0
    for k,v in bigramSProbs.items():
        logS_bigram += math.log2(v)
    
    logS_trigram = 0
    for k,v in trigramSProbs.items():
        logS_trigram += math.log2(v)
    
    logS_dict["unigram"] = logS_unigram
    logS_dict["bigram"] = logS_bigram
    logS_dict["trigram"] = logS_trigram

    return logS_dict


def ppl(sentence): # Perplexity
    print("Calculating Perplexities...")
    
    perplexity_dict = {}
    
    prob_dict = prob(sentence)
    
    # Unigram Perplexity
    log_probs_uni = prob_dict["unigram"]
    
    len_uni = len(unigramProbs)
    HW_uni = (-1/len_uni) * log_probs_uni
    perplexity_uni = math.exp(HW_uni)
    
    perplexity_dict["unigram"] = perplexity_uni
        
    # Bigram Perplexity
    if prob_dict["bigram"] == 0:
        sprob_dict = sprob(sentence)
        log_probs_bi = sprob_dict["bigram"]
    
        len_bi = len(bigramSProbs)
        HW_bi = (-1/len_bi) * log_probs_bi
        perplexity_bi = math.exp(HW_bi)
        
        perplexity_dict["bigram"] = perplexity_bi
    else:
        log_probs_bi = prob_dict["bigram"]
        
        len_bi = len(bigramProbs)
        HW_bi = (-1/len_bi) * log_probs_bi
        perplexity_bi = math.exp(HW_bi)
        
        perplexity_dict["bigram"] = perplexity_bi
    
    # Trigram Perplexity
    if prob_dict["trigram"] == 0:
        sprob_dict = sprob(sentence)
        log_probs_tri = sprob_dict["trigram"]
        
        len_tri = len(trigramSProbs)
        HW_tri = (-1/len_tri) * log_probs_tri
        perplexity_tri = math.exp(HW_tri)
        
        perplexity_dict["trigram"] = perplexity_tri
    else:
        log_probs_tri = prob_dict["trigram"]
        
        len_tri = len(trigramProbs)
        HW_tri = (-1/len_tri) * log_probs_tri
        perplexity_tri = math.exp(HW_tri)
        
        perplexity_dict["trigram"] = perplexity_tri
    
    unigramCounts.clear()
    bigramCounts.clear()
    trigramCounts.clear()
        
    unigramProbs.clear()
    bigramProbs.clear()
    trigramProbs.clear()

    log_dict.clear()
    
    unigramSProbs.clear()
    bigramSProbs.clear()
    trigramSProbs.clear()

    logS_dict.clear()
    
    file.write(sentence + "\n")
    file.write(json.dumps(perplexity_dict) + "\n")
    return perplexity_dict


uni_count = {}
uni_prob = {}
uni_chart = {}

bi_count = {}
bi_prob = {}
bi_chart = {}
next_bi = []

tri_count = {}
tri_prob = {}
tri_chart = {}
next_tri = []

word_count = []
word_length = []

uni_sentences = []

recursive_count = [0]
tri_step = [0]

def next(word, n):    
    if n == 1: # Unigram
        for data in unigram_list:
            for text in data:
                if text not in uni_count:
                    uni_count[text] = 1
                else:
                    uni_count[text] += 1
        
        for k,v in uni_count.items():
            first_word = k
            first_word_count = uni_count[first_word]
            uniProb = uni_count[first_word] / total_corpus[0]
            uni_prob[first_word] = uniProb
        
        print("Calculate Unigram Charts...")
        first_word = list(uni_prob.keys())[0]
        for k,v in uni_prob.items():
            if k == first_word:
                uni_chart[k] = v
            else:
                k_index = list(uni_prob.keys()).index(k)
                previous_key = list(uni_prob.keys())[k_index-1]
                uni_chart[k] = uni_chart[previous_key] + v
                
        uni_chart_switched = {y:x for x,y in uni_chart.items()}
        
        for a in range(word_count[0]):
            next_uni = []
            for b in range(word_length[0]):
                rand = random.random()
                index = bisect.bisect_left(list(uni_chart_switched.keys()), rand)
                value = uni_chart_switched.get(list(uni_chart_switched)[index])
                next_uni.append(value)
                if value == "</s>":
                    break
            
            listToString = ' '.join([str(elem) for elem in next_uni])
            uni_sentences.append(listToString)
        
        print("Calculated Unigram Sentences...")
        return uni_sentences
    
    elif n == 2: # Bigram
        if (recursive_count[0] == word_length[0]) or (word == "</s>"):
            bi_sentences = ' '.join([str(elem) for elem in next_bi])
            print("Calculated Bigram Sentences...")
            
            # Clear all variables
            recursive_count[0] = 0
            next_bi.clear()
                        
            return bi_sentences
        else:
            bi_count.clear()
            bi_prob.clear()
            bi_chart.clear()
            
            for data in bigram_list:
                for i in range(len(data)):
                    text = data[i].split()
                    if text[0] == word:
                        if data[i] not in bi_count:
                            bi_count[data[i]] = 1
                        else:
                            bi_count[data[i]] += 1
                            
            for k,v in bi_count.items():
                first_word = k.split()
                first_word = first_word[0]
                first_word_count = uni_count[first_word]
                bi_probs = bi_count[k] / uni_count[first_word]
                bi_prob[k] = bi_probs
                
            
            print("Calculate Bigram Charts...:",recursive_count[0])
            first_word = list(bi_prob.keys())[0]
            for k,v in bi_prob.items():
                if k == first_word:
                    bi_chart[k] = v
                else:
                    k_index = list(bi_prob.keys()).index(k)
                    previous_key = list(bi_prob.keys())[k_index-1]
                    bi_chart[k] = bi_chart[previous_key] + v
            
            bi_chart_switched = {y:x for x,y in bi_chart.items()}
            
            rand = random.random()
            
            index = bisect.bisect_left(list(bi_chart_switched.keys()), rand)
            value = bi_chart_switched.get(list(bi_chart_switched)[index])
            next_bi.append(value.split()[1])
            
            recursive_count[0] = recursive_count[0] + 1
            return next(next_bi[-1], 2)
    
    else: # Trigram
        finish_token = []
        if len(word.split()) == 3:
            finish_token.append(word.split()[2])
        elif len(word.split()) == 2:
            finish_token.append(word.split()[1])
        else:
            finish_token.append(word)
            
        if (recursive_count[0] == word_length[0]) or (finish_token[0] == "</s>"):
            tri_sentences = ' '.join([str(elem.split()[0]) for elem in next_tri])
            print("Calculated Trigram Sentences...")
            
            # Clear all variables
            recursive_count[0] = 0
            tri_step[0] = 0
            next_tri.clear()
                        
            return tri_sentences
        else:            
            bi_count.clear()
            bi_prob.clear()
            tri_count.clear()
            tri_prob.clear()
            tri_chart.clear()
            
            if tri_step[0] == 0:
                for data in bigram_list:
                    for i in range(len(data)):
                        text = data[i].split()
                        if text[0] == word:
                            if data[i] not in bi_count:
                                bi_count[data[i]] = 1
                            else:
                                bi_count[data[i]] += 1
                                
                for k,v in bi_count.items():
                    first_word = k.split()
                    first_word = first_word[0]
                    first_word_count = uni_count[first_word]
                    bi_probs = bi_count[k] / uni_count[first_word]
                    bi_prob[k] = bi_probs
                
                print("Calculate Bigram Charts...:",recursive_count[0])
                first_word = list(bi_prob.keys())[0]
                for k,v in bi_prob.items():
                    if k == first_word:
                        bi_chart[k] = v
                    else:
                        k_index = list(bi_prob.keys()).index(k)
                        previous_key = list(bi_prob.keys())[k_index-1]
                        bi_chart[k] = bi_chart[previous_key] + v
                        
                bi_chart_switched = {y:x for x,y in bi_chart.items()}
                
                rand = random.random()
                
                index = bisect.bisect_left(list(bi_chart_switched.keys()), rand)
                value = bi_chart_switched.get(list(bi_chart_switched)[index])
                
                new_word = word + " " + value.split()[1]
                next_tri.append(new_word)
                
                tri_step[0] = 1
                recursive_count[0] = recursive_count[0] + 1
            
            for data in bigram_list:
                for i in range(len(data)):
                    text = data[i].split()
                    if text[0] == next_tri[-1].split()[0]:
                        if data[i] not in bi_count:
                            bi_count[data[i]] = 1
                        else:
                            bi_count[data[i]] += 1
                            
            for k,v in bi_count.items():
                first_word = k.split()
                first_word = first_word[0]
                first_word_count = uni_count[first_word]
                bi_probs = bi_count[k] / uni_count[first_word]
                bi_prob[k] = bi_probs
            
            for data in trigram_list:
                for i in range(len(data)):
                    text = data[i].split()
                    if len(text) > 1:
                        first_two_word = text[0] + " " + text[1]
                        if first_two_word == next_tri[-1]:
                            if data[i] not in tri_count:
                                tri_count[data[i]] = 1
                            else:
                                tri_count[data[i]] += 1
                                
            for k,v in tri_count.items():
                first_word = k.split()
                first_two_word = (first_word[0], first_word[1])
                listToString = ' '.join([str(elem) for elem in first_two_word]) 
                        
                first_two_word_count = bi_count[listToString]
                tri_probs = tri_count[k] / bi_count[listToString]
                tri_prob[k] = tri_probs
            
            first_word = list(tri_prob.keys())[0]
            for k,v in tri_prob.items():
                if k == first_word:
                    tri_chart[k] = v
                else:
                    k_index = list(tri_prob.keys()).index(k)
                    previous_key = list(tri_prob.keys())[k_index-1]
                    tri_chart[k] = tri_chart[previous_key] + v
                
            tri_chart_switched = {y:x for x,y in tri_chart.items()}
                
            rand = random.random()
            
            index = bisect.bisect_left(list(tri_chart_switched.keys()), rand)
            value = ""
            if index >= len(list(tri_chart_switched)):
                value += tri_chart_switched.get(list(tri_chart_switched)[index - 1])
            else:
                value += tri_chart_switched.get(list(tri_chart_switched)[index])
            
            new_word = next_tri[-1].split()[1]
            if len(value.split()) == 3:
                new_word += " " + value.split()[2]
            elif len(value.split()) == 2:
                new_word += " " + value.split()[1]
            else:
                new_word += " " + value
                
            next_tri.append(new_word)
                
            recursive_count[0] = recursive_count[0] + 1
            return next(next_tri[-1], 3)


generate_list = {}
def generate(length, count):
    word_count.append(count)
    word_length.append(length)
    
    # Unigram Sentences Generating
    n = 1
    UnigramSentences = next("<s>", n)
    generate_list["Unigram"] = UnigramSentences
    
    # Bigram Sentences Generating
    BigramSentences = []
    for i in range(count):
        n = 2
        BigramSentences.append(next("<s>", n))
    generate_list["Bigram"] = BigramSentences
    
    # Trigram Sentences Generating
    TrigramSentences = []
    for i in range(count):
        n = 3
        TrigramSentences.append(next("<s>", n))
    generate_list["Trigram"] = BigramSentences
    
    file.write("Generating Sentences")
    file.write(json.dumps(generate_list))

dataset_list = dataset("C:/Users/Berat/Documents/GitHub/NLP-Assignments/Assignment1/cbt_train.txt")

unigram_list = Ngram(1)
bigram_list = Ngram(2)
trigram_list = Ngram(3)

generate(20,5)

unigram_ppl = {}
bigram_ppl = {}
trigram_ppl = {}

for k,v in generate_list.items():
    if k == "Unigram":
        file.write("Unigram Sentences:\n")
        for i in range(word_count[0]):
            unigram_ppl[v[i]] = ppl(v[i])
    elif k == "Bigram":
        file.write("Bigram Sentences:\n")
        for i in range(word_count[0]):
            bigram_ppl[v[i]] = ppl(v[i])
    else:
        file.write("Trigram Sentences:\n")
        for i in range(word_count[0]):
            trigram_ppl[v[i]] = ppl(v[i])
