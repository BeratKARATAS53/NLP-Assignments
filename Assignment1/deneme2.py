 
import collections 
import bisect
import random

unigram = [["a","b","a","a","a","a"],["b","a","c","d"],["d","a"]]
bigram = [["a b","b a","a b"],["b a","c d"],["d a"]]
trigram = [["a b b","b b a"],["b b a","c c d"],["d b a"]]

uni_dict2 = collections.OrderedDict()
uni_dict = {"a": 0.2111111879, "b": 0.32121212, "c": 0.111111111111, "d": 0.252345612}
sent = "a b b a"
sent = sent.split()

unigram_dict = {}
bigram_dict = {}
trigram_dict = {}

next_uni = []


# for text in unigram:
#     for word in text:
#         if word not in unigram_dict:
#             unigram_dict[word] = 1
#         else:
#             unigram_dict[word] += 1

# print(sum(unigram_dict.values()))
# for k,v in uni_dict.items():
#     first_word = list(uni_dict.keys())[0]
#     if k == first_word:
#         unigram_dict[k] = v
#     else:
#         k_index = list(uni_dict.keys()).index(k)
#         previous_key = list(uni_dict.keys())[k_index-1]
#         unigram_dict[k] = unigram_dict[previous_key] + v

# unigram_dict = {y:x for x,y in unigram_dict.items()}

# for i in range(4):
#     rand = random.random()
#     print(rand)
#     index = bisect.bisect_left(list(unigram_dict.keys()), rand)
#     print(index)
#     if index == len(uni_randoms):
#         print(index)
#         index -= 1
#     value = unigram_dict.get(list(unigram_dict)[index])
#     next_uni.append(value)

# print(next_uni)
# print(uni_dict)
# print(type(sent))
# if type(sent) == str:
#     sent = sent.split()

# for dat in unigram:
#     for sen in sent:
#         if sen in dat:
#             if sen not in unigram_dict:
#                 unigram_dict[sen] = dat.count(sen)
#             else:
#                 unigram_dict[sen] += 1

# for dat in unigram:
#     for da in dat:
#         if da not in unigram_dict:
#             unigram_dict[da] = 1
#         else:
#             unigram_dict[da] += 1
                
# print(unigram_dict)
# total_corpus = sum(unigram_dict.values())
# print(total_corpus)

# unigramProbs = {}

# for k,v in unigram_dict.items():
#     first_word = k
#     first_word_count = unigram_dict[first_word]
#     uni_prob = unigram_dict[first_word] / total_corpus
#     unigramProbs[first_word] = uni_prob

# print(unigramProbs)
# for i in range(len(sent) - 1):
#     temp = (sent[i], sent[i+1])
#     listToString = ' '.join([str(elem) for elem in temp])
#     print(listToString)
#     bigram_dict[listToString] = sum([data.count(listToString) for data in bigram])

# for dat in bigram:
#     for i in range(len(sent) - 1):
#         temp = (sent[i], sent[i+1])
#         listToString = ' '.join([str(elem) for elem in temp])
#         if listToString in dat:
#             print(listToString)
#             if listToString not in bigram_dict:
#                 bigram_dict[listToString] = dat.count(listToString)
#             else:
#                 bigram_dict[listToString] += 1
#         print(bigram_dict)

for i in range(len(sent) - 2):
    temp = (sent[i], sent[i+1], sent[i+2])
    listToString = ' '.join([str(elem) for elem in temp])
    print(listToString)
    trigram_dict[listToString] = sum([data.count(listToString) for data in trigram])
print(trigram_dict)
