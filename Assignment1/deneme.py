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

def probAllModels(sentence):
    unigramCounts = {}
    bigramCounts = {}
    trigramCounts = {}
    
    if type(sentence) == str:
        sentence = sentence.split()
    
    for text in unigram_list:
        for word in sentence:
            if word in text:
                if word not in unigramCounts:
                    unigramCounts[word] = text.count(word)
                else:
                    unigramCounts[word] += 1
           
    for k,v in unigramCounts.items():
        with open("unigram.txt", 'a') as out:
            out.write(k + '\t' + str(v)) # Delete whatever you don't want to print into a file
            out.write('\n')
            out.close()
    
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
        with open("bigram.txt", 'a') as out:
            out.write(k + '\t' + str(v)) # Delete whatever you don't want to print into a file
            out.write('\n')
            out.close()
            
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
        with open("trigram.txt", 'a') as out:
            out.write(k + '\t' + str(v)) # Delete whatever you don't want to print into a file
            out.write('\n')
            out.close()
            
    # print(trigramCounts)
    
def ppl(sentence): # Perplexity
        logprobs = [float(sentence.split()[1])]
        logP = sum(logprobs)
        N = len(logprobs)
        HW = (-1/N) * logP
        perplexity = math.exp(HW)
        print(str(perplexity))
        return perplexity

dataset_list = dataset("C:/Users/Berat/Documents/GitHub/NLP-Assignments/Assignment1/CBTest/data/cbt_train.txt")

n = 3

unigram_list = Ngram(1)
bigram_list = Ngram(2)
trigram_list = Ngram(3)

# unigram_list = []
# bigram_list = []
# trigram_list = []

# if n == 1:
#     unigram_list = Ngram(1)
# elif n == 2:
#     bigram_list = Ngram(2)
# elif n == 3:
#     trigram_list = Ngram(3)

# for i in dataset_list:
#     probAllModels(i)

