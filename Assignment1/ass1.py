import math
import sys
import re

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
    data = re.sub(r'[^A-Za-z. _-]', '', data)
    
    result = data.split()
    result.insert(0, '<s>')
    result.append('</s>')
    
    return result

def Ngram(n): # NGram Models
    print("Creating",n,"gram Model...")
    ngrams_list = []
 
    for data in dataset_list:
        data = data.split()
        sentence_ngram = []
        for number in range(0, len(data)):
            ngram = ' '.join(data[number:number + n])
            sentence_ngram.append(ngram)
        ngrams_list.append(sentence_ngram)
    
    return ngrams_list

total_corpus = 0

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
    for text in unigram_list:
        for word in sentence:
            if word in text:
                if word not in unigramCounts:
                    unigramCounts[word] = text.count(word)
                else:
                    unigramCounts[word] += 1
           
    total_corpus = sum(unigramCounts.values())
    
    for k,v in unigramCounts.items():
        first_word = k
        first_word_count = unigramCounts[first_word]
        uni_prob = unigramCounts[first_word] / total_corpus
        unigramProbs[first_word] = uni_prob
    
    print("Calculate Bigram...")
    for data in bigram_list:
        for i in range(len(sentence) - 1):
            temp = (sentence[i], sentence[i+1])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in bigramCounts:
                    bigramCounts[listToString] = data.count(listToString)
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
                    trigramCounts[listToString] = data.count(listToString)
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


unigramSProbs = {}
bigramSProbs = {}
trigramSProbs = {}

logS_dict = {}

def sprob(sentence):
    print("Calculating Smooth Probabilities...")
    
    if type(sentence) == str:
        sentence = sentence.split()
    
    unique_word_count = len(unigramCounts)
    
    total_corpus = sum(unigramCounts.values())
    
    print("Calculate Unigram...")
    for k,v in unigramCounts.items():
        first_word = k
        first_word_count = unigramCounts[first_word]
        uni_prob = unigramCounts[first_word] + 1 / total_corpus + unique_word_count # Smooth: P(x) = C(x) + 1 / C(total) + V
        unigramSProbs[first_word] = uni_prob
    
    print("Calculate Bigram...")
    for data in bigram_list:
        for i in range(len(sentence) - 1):
            temp = (sentence[i], sentence[i+1])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in bigramCounts:
                    bigramCounts[listToString] = data.count(listToString)
                else:
                    bigramCounts[listToString] += 1
    
    for k,v in bigramCounts.items():
        first_word = k.split()
        first_word = first_word[0]
        first_word_count = unigramCounts[first_word]
        bi_probs = bigramCounts[k] + 1 / unigramCounts[first_word] + unique_word_count
        bigramSProbs[k] = bi_probs
    
    print("Calculate Trigram...")
    for data in trigram_list:
        for i in range(len(sentence) - 2):
            temp = (sentence[i], sentence[i+1], sentence[i+2])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in trigramCounts:
                    trigramCounts[listToString] = data.count(listToString)
                else:
                    trigramCounts[listToString] += 1
        
    for k,v in trigramCounts.items():
        first_word = k.split()
        first_two_word = (first_word[0], first_word[1])
        listToString = ' '.join([str(elem) for elem in first_two_word]) 
            
        first_two_word_count = bigramCounts[listToString]
        tri_probs = trigramCounts[k] + 1 / bigramCounts[listToString] + unique_word_count
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


perplexity_dict = {}

def ppl(sentence): # Perplexity
    print("Calculating Perplexities...")
    
    # Unigram Perplexity
    log_probs_uni = []
    for k, v in unigramProbs.items():
        log_probs_uni.append(float(v))
    
    logP_uni = sum(log_probs_uni)
    len_uni = len(log_probs_uni)
    
    HW_uni = (-1/len_uni) * logP_uni
    perplexity_uni = math.exp(HW_uni)
    
    perplexity_dict["unigram"] = perplexity_uni
        
    # Bigram Perplexity
    log_probs_bi = []
    for k, v in bigramProbs.items():
        log_probs_bi.append(float(v))
    
    logP_bi = sum(log_probs_bi)
    len_bi = len(log_probs_bi)
    
    HW_bi = (-1/len_bi) * logP_bi
    perplexity_bi = math.exp(HW_bi)
    
    perplexity_dict["bigram"] = perplexity_bi
        
    # Trigram Perplexity
    log_probs_tri = []
    for k, v in trigramProbs.items():
        log_probs_tri.append(float(v))
    
    logP_tri = sum(log_probs_tri)
    len_tri = len(log_probs_tri)
    
    HW_tri = (-1/len_tri) * logP_tri
    perplexity_tri = math.exp(HW_tri)
    
    perplexity_dict["trigram"] = perplexity_tri


dataset_list = dataset("C:/Users/Berat/Documents/GitHub/NLP-Assignments/Assignment1/CBTest/data/cbt_train.txt")

unigram_list = Ngram(1)
bigram_list = Ngram(2)
trigram_list = Ngram(3)

prob(processData("I will look after you ."))
print("Logarithm Prob: ",log_dict)

sprob(processData("I will look after you ."))
print("Logarithm SProb: ",logS_dict)

ppl(processData("I will look after you ."))
print("Perplexity: ",perplexity_dict)