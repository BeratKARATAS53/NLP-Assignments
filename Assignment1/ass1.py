import math
import sys
import re

def dataset(folderPath): # Read Dataset from this folderpath
    dataset_list = []
    for word in open(folderPath):
        token = word.strip().lower()
        if token:
            dataset_list.append(token)
            
    return dataset_list

def processData(data): # Process every single line by 're' library
    data = re.sub(r'[^A-Za-z. ]', '', data)
    
    result = data.split()
    result.insert(0, '<s>')
    result.append('</s>')
    
    return result

def processData2(data): # Process every single line by string replace method
    data = data.replace(',','')
    data = data.replace('(','')
    data = data.replace(')','')
    data = data.replace('/','')
    data = data.replace('``','')
    data = data.replace('"','')
    data = data.replace('?','')
    data = data.replace('!','')
    data = data.replace(':','')
    data = data.replace(';','')
    data = data.replace('*','')
    
    result = data.split()
    result.insert(0, '<s>')
    result.append('</s>')
    
    return result

def Ngram(data, n): # NGram Models
    ngrams_list = []
 
    for number in range(0, len(data)):
        ngram = ' '.join(data[number:number + n])
        ngrams_list.append(ngram)
 
    return ngrams_list

def prob(sentence): # MLE Probability Function
    count_dict = {}
    
    for word in sentence:
        for text in dataset_list:
            if not word in count_dict:
                if word == '<s>' or word == '</s>':
                    count_dict[word] = len(dataset_list)
                else:
                    count_dict[word] = 1
            else:
                count_dict[word] += text.count(word)
                
    print("Probabilty Sentence:\n", count_dict())
    
    prob_dict = {}
    
    for key, value in count_dict.items():
        first_word = key[0]
        prob = unigrams[first_word] / total_corpus
    # for word in sentence:   
    #     for i in range(len(dataset_list) - 1):
    #         temp = (dataset_list[i], dataset_list[i+1]) # Tuples are easier to reuse than nested lists
    #         if not word in prob_dict_bigram:
    #             prob_dict_bigram[word] = 1
    #         else:
    #             prob_dict_bigram[word] += 1
            
            
    return 1

def ppl(sentence): # Perplexity
        logprobs = [float(sentence.split()[1])]
        logP = sum(logprobs)
        N = len(logprobs)
        HW = (-1/N) * logP
        perplexity = math.exp(HW)
        print(str(perplexity))
        return perplexity

dataset_list = dataset("C:/Users/Berat/Documents/GitHub/NLP Assignments/Assignment1/CBTest/data/cbt_train.txt")

text = processData(dataset_list[1])
unigram = Ngram(text, 1)
bigram = Ngram(text, 2)
trigram = Ngram(text, 3)

print(prob(unigram))
print(prob(bigram))
print(prob(trigram))


# for text in data:
#     text = processData(text)
#     print(Ngram(text, 1))

