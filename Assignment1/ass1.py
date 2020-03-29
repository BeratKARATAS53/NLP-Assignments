import math
import random
import bisect

import sys

import re
import json


# The file in which the perplexities of each sentence calculated according to 3 models are written.
file = open('perplexity.txt', 'w') 

def dataset(folderPath): # Read Dataset from this path.
    print("Reading Dataset...")
    
    dataset_list = [] # Each element equals one line in the file.
    for word in open(folderPath):
        token = word.strip()
        if token:
            token = processData(token)
            listToString = ' '.join([str(elem) for elem in token]) 
            
            dataset_list.append(listToString)
            
    return dataset_list

"""
I delete all punctuation except for - and _,
I shrink all the words,
And I add <s> (start token) at the beginning of the sentence and </s> (finish token) at the end.
"""
def processData(data): # Process every single line by 're' library.
    data = data.lower()
    data = data.replace("|", " ") # I separate words that are adjacent. Like that > ancestors|baby|boy|everyone
    data = re.sub(r'[^A-Za-z. _-]', '', data) # 
    
    result = data.split()
    result.insert(0, '<s>')
    result.append('</s>')
    
    return result


total_corpus = [] # Total Words Number in the Dataset

def Ngram(n): # Generate NGram Models
    print("Creating",n,"gram Model...")
    
    ngrams_list = []
    corpus = 0
    
    for data in dataset_list: # 
        data = data.split()
        corpus += len(data)
        sentence_ngram = []
        for number in range(0, len(data)): 
            ngram = ' '.join(data[number:number + n])
            sentence_ngram.append(ngram)
        ngrams_list.append(sentence_ngram)
        
    total_corpus.append(corpus)
    
    return ngrams_list

# Structures where the amounts of each word in a sentence are kept.
unigramCounts = {}
bigramCounts = {}
trigramCounts = {}
    
# Structures where the probabilities of each word in a sentence are kept.
unigramProbs = {}
bigramProbs = {}
trigramProbs = {}

# The logarithm of the probabilities of each sentence.
log_dict = {}
    
"""
The prob function processes the sentence it takes as a parameter according to the unigram, bigram and trigram models,
and calculates the probability according to these models and calculates their logarithm.
It also writes the result to log_dict.
"""
def prob(sentence):
    print("Calculating Probabilities...")
    
    # If the sentence has not been previously processed, it converts it into an array for easier working.
    if type(sentence) == str:
        sentence = sentence.split()
    
    # Calculating the amounts of each word according to the unigram and write to the unigramCounts dictionary.
    for word in sentence:
        unigramCounts[word] = sum([data.count(word) for data in unigram_list])
    
    # Calculating the probabilities of each word according to the unigram and write to the unigramProbs dictionary.
    for k,v in unigramCounts.items():
        first_word = k
        first_word_count = unigramCounts[first_word]
        uni_prob = unigramCounts[first_word] / total_corpus[0]
        unigramProbs[first_word] = uni_prob
    
    # Calculating the amounts of each word according to the bigram and write to the bigramCounts dictionary.
    for data in bigram_list:
        for i in range(len(sentence) - 1):
            temp = (sentence[i], sentence[i+1])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in bigramCounts:
                    bigramCounts[listToString] = 1
                else:
                    bigramCounts[listToString] += 1
    
    # Calculating the probabilities of each word according to the bigram and write to the bigramProbs dictionary.
    for k,v in bigramCounts.items():
        first_word = k.split()
        first_word = first_word[0]
        first_word_count = unigramCounts[first_word]
        bi_probs = bigramCounts[k] / unigramCounts[first_word]
        bigramProbs[k] = bi_probs
    
    # Calculating the amounts of each word according to the trigram and write to the trigramCounts dictionary.
    for data in trigram_list:
        for i in range(len(sentence) - 2):
            temp = (sentence[i], sentence[i+1], sentence[i+2])
            listToString = ' '.join([str(elem) for elem in temp])
            if listToString in data:
                if listToString not in trigramCounts:
                    trigramCounts[listToString] = 1
                else:
                    trigramCounts[listToString] += 1
        
    # Calculating the probabilities of each word according to the trigram and write to the trigramProbs dictionary.
    for k,v in trigramCounts.items():
        first_word = k.split()
        first_two_word = (first_word[0], first_word[1])
        listToString = ' '.join([str(elem) for elem in first_two_word]) 
            
        first_two_word_count = bigramCounts[listToString]
        tri_probs = trigramCounts[k] / bigramCounts[listToString]
        trigramProbs[k] = tri_probs
    
    # Calculating logarithms of probabilities calculated according to each model and writing to the log_dict dictionary.
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
    
    # As a result, I return the logarithm of the probabilities so that I can use it when calculating perplexity.
    return log_dict


# Structures where the smooth probabilities of each word in a sentence are kept.
unigramSProbs = {}
bigramSProbs = {}
trigramSProbs = {}

# The logarithm of the smooth probabilities of each sentence.
logS_dict = {}

"""
The sprob function processes the sentence it takes as a parameter according to the unigram, bigram and trigram models,
and calculates the smooth probability according to these models and calculates their logarithm.
It also writes the result to logS_dict.
"""
def sprob(sentence):
    print("Calculating Smooth Probabilities...")
    
    # If the sentence has not been previously processed, it converts it into an array for easier working.
    if type(sentence) == str: 
        sentence = sentence.split()
    
    unigramCounts.clear()
    bigramCounts.clear()
    trigramCounts.clear()
    
    unigram_unique_count = {}
                
    # Calculating the amounts of each word according to the unigram and write to the unigramCounts dictionary.
    # And I calculate the total number of unique words in the dataset, so V size.
    for text in unigram_list:
        for word in sentence:
            if word in text:
                if word not in unigramCounts:
                    unigramCounts[word] = 1
                    unigram_unique_count[word] = 1
                else:
                    unigramCounts[word] += 1
    
    unique_word_count = sum(unigram_unique_count.values())
    
    # Calculating the smooth probabilities of each word according to the unigram and write to the unigramSProbs dictionary.
    for k,v in unigramCounts.items():
        first_word = k
        first_word_count = unigramCounts[first_word]
        uni_prob = (unigramCounts[first_word] + 1) / (total_corpus[0] + unique_word_count) # Smooth: P(x) = C(x) + 1 / C(total) + V
        unigramSProbs[first_word] = uni_prob
             
    # Calculating the amounts of each word according to the bigram and write to the bigramCounts dictionary.
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
    
    # Calculating the smooth probabilities of each word according to the bigram and write to the bigramSProbs dictionary.
    for k,v in bigramCounts.items():
        first_word = k.split()
        first_word = first_word[0]
        first_word_count = unigramCounts[first_word]
        bi_probs = (bigramCounts[k] + 1) / (unigramCounts[first_word] + unique_word_count)
        bigramSProbs[k] = bi_probs
    
    # Calculating the amounts of each word according to the trigram and write to the trigramCounts dictionary.
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
        
    # Calculating the smooth probabilities of each word according to the trigram and write to the trigramSProbs dictionary.
    for k,v in trigramCounts.items():
        first_word = k.split()
        first_two_word = (first_word[0], first_word[1])
        listToString = ' '.join([str(elem) for elem in first_two_word]) 
            
        first_two_word_count = bigramCounts[listToString]
        tri_probs = (trigramCounts[k] + 1) / (bigramCounts[listToString] + unique_word_count)
        trigramSProbs[k] = tri_probs
    
    # Calculating logarithms of smooth probabilities calculated according to each model and writing to the logS_dict dictionary.
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
    
    # As a result, I return the logarithm of the smooth probabilities so that I can use it when calculating perplexity.
    return logS_dict 


"""
The ppl function firstly calculates the normal probability of the sentence given as a parameter.
If the probability is 0 for any model, this time only calculates the smooth probability for that model.
And it writes these results into the perplexity_dict dictionary.

Perplexity Formula: 2^-((1/N) * log(sentence))
"""
def ppl(sentence): # Perplexity
    print("Calculating Perplexities...")
    
    perplexity_dict = {}
    
    prob_dict = prob(sentence)
    
    """ Unigram Perplexity:
        According to the unigram model of the sentence, I calculate directly since the perplexity is not likely to be 0. """
    log_probs_uni = prob_dict["unigram"] # log(sentence)
    
    len_uni = len(unigramProbs) # N
    HW_uni = (-1/len_uni) * log_probs_uni # -((1/N) * log(sentence))
    perplexity_uni = math.exp(HW_uni) # 2^-((1/N) * log(sentence))
    
    perplexity_dict["unigram"] = perplexity_uni
        
    """ Bigram Perplexity:
        If the probability of the sentence is 0 according to the bigram model, the smooth probability is calculated, 
        otherwise, the normal probability is processed. """
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
    
    """ Trigram Perplexity:
        If the probability of the sentence is 0 according to the trigram model, the smooth probability is calculated, 
        otherwise, the normal probability is processed. """
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
    
    """ After calculating the Perplexity, all previously used structures are cleared,
        in order to make the correct calculation in the next sentence. """
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
    
    # The sentence and its perplexity are written to the file.
    file.write(sentence + "\n")
    file.write(json.dumps(perplexity_dict) + "\n")
    return perplexity_dict

# It was created according to the unigram model that will be used to create words; count, probability, and gan-chart structures.
uni_count = {}
uni_prob = {}
uni_chart = {}

# It was created according to the bigram model that will be used to create words; count, probability, and gan-chart structures.
bi_count = {}
bi_prob = {}
bi_chart = {}
next_bi = []

# It was created according to the trigram model that will be used to create words; count, probability, and gan-chart structures.
tri_count = {}
tri_prob = {}
tri_chart = {}
next_tri = []

# Parameters taken from the Generate function
word_count = []
word_length = []

""" 
tri_step[]: In the trigram model I created, I only process according to the bigram in the first step, 
since there is only one <s> token at the beginning of the sentence. In other steps, I continue from trigram.
Here is the variable I use to check if the current step is the first step. 

recursive_count[]: With this variable I hold each step number, 
I complete the function if the current step number has reached the total number of words to be produced. """
tri_step = [0]
recursive_count = [0]

# Word generation function. I wrote the details below as they vary on each model basis.
def next(word, n):    
    if n == 1: # Unigram
        """ I create the sentence in Unigram as follows:
        First of all, I place the probabilities of each word in the dataset on a fictitious 
        Gantt chart, which I call uni_chart, and select a word among them with the number I randomly created. 
        And I repeat this selection as much as the maximum number of words. """
        
        uni_sentences = []
        # Calculating the amounts of each word according to the unigram and write to the uni_count dictionary.
        for data in unigram_list:
            for text in data:
                if text not in uni_count:
                    uni_count[text] = 1
                else:
                    uni_count[text] += 1
        
        # Calculating the probabilities of each word according to the unigram and write to the uni_prob dictionary.
        for k,v in uni_count.items():
            first_word = k
            first_word_count = uni_count[first_word]
            uniProb = uni_count[first_word] / total_corpus[0]
            uni_prob[first_word] = uniProb
        
        # I place the probabilities in the chart in the range 0-1.
        print("Calculate Unigram Charts...")
        first_word = list(uni_prob.keys())[0]
        for k,v in uni_prob.items():
            if k == first_word:
                uni_chart[k] = v
            else:
                k_index = list(uni_prob.keys()).index(k)
                previous_key = list(uni_prob.keys())[k_index-1]
                uni_chart[k] = uni_chart[previous_key] + v
        
        # In order to work easier, I put the probabilities as keys.
        uni_chart_switched = {y:x for x,y in uni_chart.items()}
        
        for a in range(word_count[0]):
            next_uni = [] # I add each word I choose here and then I turn it into a sentence.
            for b in range(word_length[0]):
                rand = random.random()
                
                # The bisect library allows me to find the value closest to the random number from the chart.
                index = bisect.bisect_left(list(uni_chart_switched.keys()), rand) # Return an index.
                value = uni_chart_switched.get(list(uni_chart_switched)[index]) # I find the word according to index.
                next_uni.append(value)
                
                if value == "</s>": # If the word is a finish token, the sentence is complete.
                    break
            
            listToString = ' '.join([str(elem) for elem in next_uni])
            uni_sentences.append(listToString)
        
        print("Calculated Unigram Sentences...")
        return uni_sentences # As a result, I return the sentences I created.
    
    elif n == 2: # Bigram
        """ In Bigram, I create the sentence as follows:
        First of all, I keep the probability table of the words come after my parameter value <s>. 
        And I place them in a fictitious Gantt chart, which I call bi_chart, and select a word 
        from them with the number I randomly created. And I repeat this selection process until 
        I reach the maximum number of words or encounter the </s> token. """
        
        if (recursive_count[0] == word_length[0]) or (word == "</s>"):
            bi_sentences = ' '.join([str(elem) for elem in next_bi])
            
            # Clear all variables
            recursive_count[0] = 0
            next_bi.clear()
                        
            return bi_sentences
        else:
            # In the next word selection, I clear the structures so that the previous words do not cause confusion.
            bi_count.clear()
            bi_prob.clear()
            bi_chart.clear()
            
            # Calculating the amounts of each word according to the bigram and write to the bi_count dictionary.
            for data in bigram_list:
                for i in range(len(data)):
                    text = data[i].split()
                    if text[0] == word:
                        if data[i] not in bi_count:
                            bi_count[data[i]] = 1
                        else:
                            bi_count[data[i]] += 1
            
            # Calculating the probabilities of each word according to the bigram and write to the bi_prob dictionary.
            for k,v in bi_count.items():
                first_word = k.split()
                first_word = first_word[0]
                first_word_count = uni_count[first_word]
                bi_probs = bi_count[k] / uni_count[first_word]
                bi_prob[k] = bi_probs
            
            # I place the probabilities in the chart in the range 0-1.
            print("Calculate Bigram Charts...:",recursive_count[0])
            first_word = list(bi_prob.keys())[0]
            for k,v in bi_prob.items():
                if k == first_word:
                    bi_chart[k] = v
                else:
                    k_index = list(bi_prob.keys()).index(k)
                    previous_key = list(bi_prob.keys())[k_index-1]
                    bi_chart[k] = bi_chart[previous_key] + v
            
            # In order to work easier, I put the probabilities as keys.
            bi_chart_switched = {y:x for x,y in bi_chart.items()}
            
            rand = random.random()
            
            # The bisect library allows me to find the value closest to the random number from the chart.
            index = bisect.bisect_left(list(bi_chart_switched.keys()), rand) # Return an index.
            value = bi_chart_switched.get(list(bi_chart_switched)[index]) # I find the word according to index.
            
            """
            "value" value consists of two words like 'they have',
            I add the second word to the sentence for the next step and send the second word to the next() function. """
            next_bi.append(value.split()[1])
            
            recursive_count[0] = recursive_count[0] + 1 # Increase a step number
            return next(next_bi[-1], 2)
    
    else: # Trigram
        """
        In Trigram, I create the sentence as follows:
        In the first step, I keep the probability table of the words come after my parameter value <s>. 
        I place them in a fictitious Gantt chart. And I select a word among them by the number I randomly created. 
        This process takes place only with the help of the bigram model in the first step. 
        I combine this word and the value in the parameter and send it to the next() function again.
        In the second step, I have a binary word, like "they have". 
        Now I place the probability of each word in the data set that comes after this binary word into a fictitious gan chart, 
        which I call tri_chart, and select a word among them with the number I randomly created.
        And I repeat this selection process until I reach the maximum number of words or encounter the </s> token.
        """
        
        # If my parameter value has a </s> token, I check this to complete the function.
        finish_token = []
        if len(word.split()) == 3:
            finish_token.append(word.split()[2])
        elif len(word.split()) == 2:
            finish_token.append(word.split()[1])
        else:
            finish_token.append(word)
            
        if (recursive_count[0] == word_length[0]) or (finish_token[0] == "</s>"):
            tri_sentences = ' '.join([str(elem.split()[0]) for elem in next_tri])
            
            # Clear all variables
            recursive_count[0] = 0
            tri_step[0] = 0
            next_tri.clear()
                        
            return tri_sentences
        else:
            # In the next word selection, I clear the structures so that the previous words do not cause confusion.
            bi_count.clear()
            bi_prob.clear()
            tri_count.clear()
            tri_prob.clear()
            tri_chart.clear()
            
            # If I am in the first step, as I mentioned at the beginning, I choose the second word for the bigram model.
            if tri_step[0] == 0:
                # Calculating the amounts of each word according to the bigram and write to the bi_count dictionary.
                for data in bigram_list:
                    for i in range(len(data)):
                        text = data[i].split()
                        if text[0] == word:
                            if data[i] not in bi_count:
                                bi_count[data[i]] = 1
                            else:
                                bi_count[data[i]] += 1
                  
                # Calculating the probabilities of each word according to the bigram and write to the bi_prob dictionary.
                for k,v in bi_count.items():
                    first_word = k.split()
                    first_word = first_word[0]
                    first_word_count = uni_count[first_word]
                    bi_probs = bi_count[k] / uni_count[first_word]
                    bi_prob[k] = bi_probs

                # I place the probabilities in the chart in the range 0-1.
                print("Calculate Bigram Charts...:",recursive_count[0])
                first_word = list(bi_prob.keys())[0]
                for k,v in bi_prob.items():
                    if k == first_word:
                        bi_chart[k] = v
                    else:
                        k_index = list(bi_prob.keys()).index(k)
                        previous_key = list(bi_prob.keys())[k_index-1]
                        bi_chart[k] = bi_chart[previous_key] + v
                
                # In order to work easier, I put the probabilities as keys.
                bi_chart_switched = {y:x for x,y in bi_chart.items()}
                
                rand = random.random()
                
                # The bisect library allows me to find the value closest to the random number from the chart.
                index = bisect.bisect_left(list(bi_chart_switched.keys()), rand) # Return an index.
                value = bi_chart_switched.get(list(bi_chart_switched)[index]) # I find the word according to index.
                
                new_word = word + " " + value.split()[1]
                next_tri.append(new_word)
                
                tri_step[0] = 1
                recursive_count[0] = recursive_count[0] + 1
            
            # Calculating the amounts of each word according to the bigram and write to the bi_count dictionary.
            for data in bigram_list:
                for i in range(len(data)):
                    text = data[i].split()
                    if text[0] == next_tri[-1].split()[0]:
                        if data[i] not in bi_count:
                            bi_count[data[i]] = 1
                        else:
                            bi_count[data[i]] += 1
            
            # Calculating the probabilities of each word according to the bigram and write to the bi_prob dictionary.
            for k,v in bi_count.items():
                first_word = k.split()
                first_word = first_word[0]
                first_word_count = uni_count[first_word]
                bi_probs = bi_count[k] / uni_count[first_word]
                bi_prob[k] = bi_probs
            
            # Calculating the amounts of each word according to the trigram and write to the tri_count dictionary.
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
            
            # Calculating the probabilities of each word according to the trigram and write to the tri_prob dictionary.
            for k,v in tri_count.items():
                first_word = k.split()
                first_two_word = (first_word[0], first_word[1])
                listToString = ' '.join([str(elem) for elem in first_two_word]) 
                        
                first_two_word_count = bi_count[listToString]
                tri_probs = tri_count[k] / bi_count[listToString]
                tri_prob[k] = tri_probs
            
            # I place the probabilities in the chart in the range 0-1.
            first_word = list(tri_prob.keys())[0]
            for k,v in tri_prob.items():
                if k == first_word:
                    tri_chart[k] = v
                else:
                    k_index = list(tri_prob.keys()).index(k)
                    previous_key = list(tri_prob.keys())[k_index-1]
                    tri_chart[k] = tri_chart[previous_key] + v
            
            # In order to work easier, I put the probabilities as keys.
            tri_chart_switched = {y:x for x,y in tri_chart.items()}
                
            rand = random.random()
            
            # The bisect library allows me to find the value closest to the random number from the chart.
            index = bisect.bisect_left(list(tri_chart_switched.keys()), rand) # Return an index.
            value = ""
            if index >= len(list(tri_chart_switched)):
                value += tri_chart_switched.get(list(tri_chart_switched)[index - 1])
            else:
                value += tri_chart_switched.get(list(tri_chart_switched)[index])
            
            """
            "value" value consists of three words like 'they have reading',
            I add the third word to the sentence for the next step and 
            send the second and third words ("have reading") to the next() function. """
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


# The structure in which the sentences created according to each model are kept.
generate_list = {}

"""
Generate function produces sentences for 3 different models with 2 parameters (length, count). """
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
    generate_list["Trigram"] = TrigramSentences
    
    # The sentences are written to the file.
    file.write("Generating Sentences")
    file.write(json.dumps(generate_list))

# Reading a dataset
dataset_list = dataset("C:/Users/Berat/Documents/GitHub/NLP-Assignments/Assignment1/cbt_train.txt")

# Generating Models
unigram_list = Ngram(1)
bigram_list = Ngram(2)
trigram_list = Ngram(3)

# Generating Sentences
generate(20,5)

# Calculating Perplexities
for k,v in generate_list.items():
    if k == "Unigram":
        file.write("Unigram Sentences:\n")
        for i in range(word_count[0]):
            ppl(v[i])
    elif k == "Bigram":
        file.write("Bigram Sentences:\n")
        for i in range(word_count[0]):
            ppl(v[i])
    else:
        file.write("Trigram Sentences:\n")
        for i in range(word_count[0]):
            ppl(v[i])
