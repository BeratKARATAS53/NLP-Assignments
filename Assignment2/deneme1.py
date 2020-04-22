# from pytorch_pretrained_bert import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
# VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
# tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
# idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

# class NerDataset(data.Dataset):
#     def __init__(self, fpath):
#         """
#         fpath: [train|valid|test].txt
#         """
#         entries = open(fpath, 'r').read().strip().split("\n\n")
#         sents, tags_li = [], [] # list of lists
#         for entry in entries:
#             words = [line.split()[0] for line in entry.splitlines()]
#             tags = ([line.split()[-1] for line in entry.splitlines()])
#             sents.append(["[CLS]"] + words + ["[SEP]"])
#             tags_li.append(["<PAD>"] + tags + ["<PAD>"])
#         self.sents, self.tags_li = sents, tags_li

#     def __len__(self):
#         return len(self.sents)

#     def __getitem__(self, idx):
#         words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

#         # We give credits only to the first piece.
#         x, y = [], [] # list of ids
#         is_heads = [] # list. 1: the token is the first piece of a word
#         for w, t in zip(words, tags):
#             tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
#             xx = tokenizer.convert_tokens_to_ids(tokens)

#             is_head = [1] + [0]*(len(tokens) - 1)

#             t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
#             yy = [tag2idx[each] for each in t]  # (T,)

#             x.extend(xx)
#             is_heads.extend(is_head)
#             y.extend(yy)

#         assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

#         # seqlen
#         seqlen = len(y)

#         # to string
#         words = " ".join(words)
#         tags = " ".join(tags)
#         return words, x, is_heads, tags, y, seqlen


# def pad(batch):
#     '''Pads to the longest sample'''
#     f = lambda x: [sample[x] for sample in batch]
#     words = f(0)
#     is_heads = f(2)
#     tags = f(3)
#     seqlens = f(-1)
#     maxlen = np.array(seqlens).max()

#     f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
#     x = f(1, maxlen)
#     y = f(-2, maxlen)


#     f = torch.LongTensor

#     return words, f(x), is_heads, tags, f(y), seqlens
import numpy as np
from collections import Counter
import math
import sys


class HMM:
    
    transition_tags = [["<s> T","T </s>"],["<s> S","S A","A A","A C","C G","G </s>"],["<s> C","C A","A S","S </s>"],["<s> C","C A","A </s>"],["<s> A","A T","T C","C </s>"]]
    emission_tags = [["T S","X Y"],["T A G","X Z Y"],["A G C","X Y Z"],["A C","Z Y"],["G A","X Y"],["A T C","X Z Y"],["C G T","X Y Z"],["S S","Z Y"]]
    sentences = ["X X","X Y Z","Z X","X","Z Z Y","C"]
    test_sentences = {}
    
    V_size = 0
    
    def HMM(self):
        each_tag_counts = {}
        
        transition_tag_counts = {}
        transition_prob = {}
        
        array_to_npArr = np.asarray(self.transition_tags)

        for tags in array_to_npArr:
            each_counts = dict(Counter(tags))
        
            for k,v in each_counts.items():
                if k not in transition_tag_counts:
                    transition_tag_counts[k] = v
                else:
                    transition_tag_counts[k] += 1
        # print(transition_tag_counts)
        
        for tags in array_to_npArr:
            for tag in tags:
                tag = tag.split()
                each_counts = dict(Counter(tag))
            
                for k,v in each_counts.items():
                    if k not in each_tag_counts:
                        each_tag_counts[k] = v
                    else:
                        each_tag_counts[k] += 1
                        
        # print(each_tag_counts)
        
        for k,v in transition_tag_counts.items():
            first_tag = k.split()
            first_tag = first_tag[0]
            first_tag_count = each_tag_counts[first_tag]
            trans_prob = transition_tag_counts[k] / each_tag_counts[first_tag]
            transition_prob[k] = round(trans_prob,2)
        
        # print(transition_prob)
        
        emissionTag_Word_dict = {}
        emission_word_counts = {}
        emission_prob = {}
        
        for sentence in self.emission_tags:
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
                        
        print(emission_word_counts)
        
        for k1,v1 in emission_word_counts.items():
            counter = sum(map(Counter, emission_word_counts.values()), Counter())
            count_dict = dict(counter)
            
            em_prob = {}
            for k2,v2 in v1.items():
                    emis_prob = v2 / count_dict[k2]
                    em_prob[k2] = round(emis_prob,2)
                            
            emission_prob[k1] = em_prob
            
            self.V_size += len(v1)
            
        print(emission_prob)

        return transition_prob, emission_prob
        
    def viterbi(self, transition_prob, emission_prob, test_sentences):
        transition_tags = list(emission_prob.keys())
        
        transition_tags.sort()
        transition_tags.insert(0,"<s>")
        transition_tags.append("</s>")
        print("Unique Transition Tags: ",transition_tags)

        transition_matrix_len = len(transition_tags)

        transition_matrix = np.zeros((transition_matrix_len-1,transition_matrix_len-1))

        for i in range(transition_matrix_len):
            uniqe_tag = transition_tags[i]
            for j in range((transition_matrix_len)-1):
                tag = uniqe_tag + " " + transition_tags[j+1]
                if tag in transition_prob:
                    transition_matrix[j][i] = transition_prob[tag]
        
        print("Transition:\n",transition_matrix)
        
        emission_tags = list(emission_prob.keys())
        emission_tags.sort()
        print("Unique Emission Tags: ",emission_tags)

        row_count = len(emission_tags)
        
        for sentences in test_sentences:
            sentences = sentences.split()
            column_count = len(sentences)
            emission_matrix = np.zeros((row_count,column_count))
            i = 0
            for word in sentences:
                for tag in emission_tags:
                    j = emission_tags.index(tag)
                    if word in emission_prob[tag]:
                        emission_matrix[j][i] = emission_prob[tag][word]
                    else:
                        emission_prob[tag][word] = 1 / (len(sentences) + self.V_size)
                        emission_matrix[j][i] = emission_prob[tag][word]
                i += 1
                print(sentences, "\n", emission_matrix)
                        
            
            tag_path_array = []
            # Start State
            viterbi_matrix = np.zeros((row_count+1,column_count+1))
            for tag in emission_tags:
                i = emission_tags.index(tag)
                viterbi_matrix[i][0] = transition_matrix[i][0]
            
            # tag_path_array.append(emission_tags[np.argmax(viterbi_matrix[:,0])])
            # print("Viterbi:\n", viterbi_matrix)
            
            for word in sentences:
                i = sentences.index(word)
                if i == 0:
                    for tag in emission_tags:
                        j = emission_tags.index(tag)
                                        
                        print("viterbi: ",viterbi_matrix[j][i],
                                ", emission: ",emission_matrix[j][i])
                                                
                        result = transition_matrix[j][i] * emission_matrix[j][i] # (?)
                        print("result: ",result)
                                        
                        viterbi_matrix[j][i] = result
                else:
                    for tag in emission_tags:
                        j = emission_tags.index(tag)
                        each_cell = np.zeros(len(emission_tags))
                        for k in range(len(emission_tags)):
                                            
                            print("Transition: ",transition_tags[k+1],"-",transition_tags[j+1],
                                    ", Emission: ",emission_tags[j],"-",word,
                                    ", viterbi: ",viterbi_matrix[k][i-1],
                                    ", transition: ",transition_matrix[j][k+1],
                                    ", emission: ",emission_matrix[j][i])
                                            
                            result = viterbi_matrix[k][i-1] * transition_matrix[j][k+1] * emission_matrix[j][i]
                            print("result: ",result)
                            each_cell[k] = result
                        viterbi_matrix[j][i] = max(each_cell)
                
                        
            end_result = np.zeros(len(emission_tags))
            for tag in emission_tags:
                i = emission_tags.index(tag)
                result = viterbi_matrix[i][len(viterbi_matrix[0])-2] * transition_matrix[len(transition_matrix)-1][i+1]
                
                # print("viterbi: ",viterbi_matrix[i][len(viterbi_matrix[0])-2])
                # print("transition: ",transition_matrix[i+1][len(transition_matrix[0])-1] )
                
                end_result[i] = result
                
                
            viterbi_matrix[viterbi_matrix == 0] = -1
            
            for i in range(len(viterbi_matrix[0])-1):
                tag_path_array.append(emission_tags[np.argmax(viterbi_matrix[:,i])])
            
            viterbi_matrix[len(viterbi_matrix)-1][len(viterbi_matrix[0])-1] = max(end_result)
            
            print(sentences,"-",tag_path_array, "\n",viterbi_matrix)
    
classHMM = HMM()

model_result = classHMM.HMM()
classHMM.viterbi(model_result[0],model_result[1],classHMM.sentences)