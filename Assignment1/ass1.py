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
        
        prob(sentence_ngram)
        
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
    
    total_corpus = sum(count_dict.values())
    
    prob_dict = {}
    
    # for word in sentence:   
    #     for i in range(len(dataset_list) - 1):
    #         temp = (dataset_list[i], dataset_list[i+1]) # Tuples are easier to reuse than nested lists
    #         if not word in prob_dict_bigram:
    #             prob_dict_bigram[word] = 1
    #         else:
    #             prob_dict_bigram[word] += 1
            
            
    return 1

def probAllModels(sentence):
    unigramCounts = {}
    bigramCounts = {}
    trigramCounts = {}
    
    for word in sentence:
        for text in dataset_list:
            if not word in unigramCounts:
                if word == '<s>' or word == '</s>':
                    unigramCounts[word] = len(dataset_list)
                else:
                    unigramCounts[word] = text.count(word)
            else:
                unigramCounts[word] += 1
                
    for word in sentence:   
        for i in range(len(dataset_list) - 1):
            temp = (dataset_list[i], dataset_list[i+1]) # Tuples are easier to reuse than nested lists
            if not word in bigramCounts:
                bigramCounts[word] = 1
            else:
                bigramCounts[word] += 1
    
    print("Probabilty Sentence:\n", unigramCounts)
    
def ppl(sentence): # Perplexity
        logprobs = [float(sentence.split()[1])]
        logP = sum(logprobs)
        N = len(logprobs)
        HW = (-1/N) * logP
        perplexity = math.exp(HW)
        print(str(perplexity))
        return perplexity

dataset_list = dataset("C:/Users/Berat/Documents/GitHub/NLP-Assignments/Assignment1/CBTest/data/cbt_train.txt")

unigram = Ngram(1)

# prob(["<s>", "Merhaba", "Dunya", "Dunya", "sana", "diyorum?","</s>"])
# prob(["<s> Merhaba", "Merhaba Dunya", "Dunya Dunya", "Dunya sana", "sana diyorum?","diyorum? </s>"])

# bigram = Ngram(2)
# trigram = Ngram(3)

# print(unigram)

# print(prob(unigram))
# print(prob(bigram))
# print(prob(trigram))


# for text in data:
#     text = processData(text)
#     print(Ngram(text, 1))

