import math
import sys
import re

def dataset(folderPath): # Read Dataset from this folderpath
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

# def processData2(data): # Process every single line by string replace method
#     data = data.replace(',','')
#     data = data.replace('(','')
#     data = data.replace(')','')
#     data = data.replace('/','')
#     data = data.replace('``','')
#     data = data.replace('"','')
#     data = data.replace('?','')
#     data = data.replace('!','')
#     data = data.replace(':','')
#     data = data.replace(';','')
#     data = data.replace('*','')
    
#     result = data.split()
#     result.insert(0, '<s>')
#     result.append('</s>')
    
#     return result

def Ngram(n): # NGram Models
    ngrams_list = []
 
    for data in dataset_list:
        data = data.split()
        # sentence = processData(data)
        sentence_ngram = []
        for number in range(0, len(data)):
            ngram = ' '.join(data[number:number + n])
            sentence_ngram.append(ngram)
        ngrams_list.append(sentence_ngram)
    
    return ngrams_list

def prob(sentence): # MLE Probability Function
    count_dict = {}
    
    # print("Word: ",word)
    for text in dataset_list:
        # print("Sentence:\n",sentence)
        for word in sentence:
            if word in text:
                if word not in count_dict:
                    count_dict[word] = 1
                else:
                    count_dict[word] += 1
    
    print(count_dict)
    
    prob_dict = {}
    
    
    return 1

total_corpus = 0
def probAllModels(sentence):
    unigramCounts = {}
    bigramCounts = {}
    trigramCounts = {}
    
    unigramProbs = {}
    bigramProbs = {}
    trigramProbs = {}
    
    if type(sentence) == str:
        sentence = sentence.split()
    
    print("Probabilities:\n")
    for text in unigram_list:
        for word in sentence:
            if word in text:
                if word not in unigramCounts:
                    unigramCounts[word] = text.count(word)
                else:
                    unigramCounts[word] += 1
           
    total_corpus = sum(unigramCounts.values())
    
    # for k,v in unigramCounts.items():
    #     with open("unigram.txt", 'a') as out:
    #         out.write(k + '\t' + str(v)) # Delete whatever you don't want to print into a file
    #         out.write('\n')
    #         out.close()
    
    for k,v in unigramCounts.items():
        first_word = k
        first_word_count = unigramCounts[first_word]
        uni_prob = unigramCounts[first_word] / total_corpus
        unigramProbs[first_word] = uni_prob
    
    print(unigramProbs)
    
    for data in bigram_list:
        for i in range(len(sentence) - 1):
            temp = (sentence[i], sentence[i+1])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in bigramCounts:
                    bigramCounts[listToString] = data.count(listToString)
                else:
                    bigramCounts[listToString] += 1
    
    # for k,v in bigramCounts.items():
    #     with open("bigram.txt", 'a') as out:
    #         out.write(k + '\t' + str(v)) # Delete whatever you don't want to print into a file
    #         out.write('\n')
    #         out.close()
    
    for k,v in bigramCounts.items():
        first_word = k.split()
        first_word = first_word[0]
        first_word_count = unigramCounts[first_word]
        bi_probs = bigramCounts[k] / unigramCounts[first_word]
        bigramProbs[k] = bi_probs
    
    print(bigramProbs)
    
    for data in trigram_list:
        for i in range(len(sentence) - 2):
            temp = (sentence[i], sentence[i+1], sentence[i+2])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in trigramCounts:
                    trigramCounts[listToString] = data.count(listToString)
                else:
                    trigramCounts[listToString] += 1
    
    # for k,v in trigramCounts.items():
    #     with open("trigram.txt", 'a') as out:
    #         out.write(k + '\t' + str(v)) # Delete whatever you don't want to print into a file
    #         out.write('\n')
    #         out.close()
    
    for k,v in trigramCounts.items():
        first_word = k.split()
        first_two_word = (first_word[0], first_word[1])
        listToString = ' '.join([str(elem) for elem in first_two_word]) 
        
        first_two_word_count = bigramCounts[listToString]
        tri_probs = trigramCounts[k] / bigramCounts[listToString]
        trigramProbs[k] = tri_probs
    
    print(trigramProbs)
    
    print("\nLogarithms:\n")
    
    log_unigram = 0
    log_bigram = 0
    log_trigram = 0
    
    for k,v in unigramProbs.items():
        log_unigram += math.log2(v)
        
    print(log_unigram)
    
    for k,v in bigramProbs.items():
        log_bigram += math.log2(v)
        
    print(log_bigram)
    
    for k,v in trigramProbs.items():
        log_trigram += math.log2(v)
        
    print(log_trigram)
        
def ppl(sentence): # Perplexity
        logprobs = [float(sentence[1])]
        logP = sum(logprobs)
        N = len(logprobs)
        HW = (-1/N) * logP
        perplexity = math.exp(HW)
        print(str(perplexity))
        return perplexity

dataset_list = dataset("C:/Users/Berat/Documents/GitHub/NLP-Assignments/Assignment1/CBTest/data/cbt_train.txt")

unigram_list = Ngram(1)
bigram_list = Ngram(2)
trigram_list = Ngram(3)

probAllModels(processData("I will look after you ."))

# ppl(processData("I will look after you ."))