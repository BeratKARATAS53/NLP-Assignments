import math
import sys
import re

import numpy as np
from collections import Counter

class HMM():
    
    datasets = [] # Its 2d array contains train and test datasets. First index train set, second index test set.
    V_size = 0 # Total unique words in train dataset.
    
    """
    **Arguments**:
    
        :param input_folder: File path to read Train and Test datasets
        :type input_folder: A string

    **Arg. Example**:
    
        >>> input_folder = ./Assignment2/dataset/
        
    **Explanation**:
    
        I read the file line by line.
        I split each line according to the empty character and 
        I get the elements of the resulting array in the first and last index. 
        The ones in the first index represent the tags, 
        and the ones in the last index represent the sentence.
        And finally, I add the resulting train and test sequences to the class variable called dataset.
        
        >>> dataset array:
            [('O O B-LOC O O O O B-PER O O O O', 'SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .'), 
             ('B-PER I-PER', 'Nadim Ladki'),...]
        
    """
    def dataset(self, input_folder):
        # Read a Train data!
        train_data = open(input_folder+"train.txt", 'r')
        lines = []; words = []; tags = []
        for line in train_data:
            word = line.strip().split(' ')[0] # First string in line
            tag = line.strip().split(' ')[-1] # Last string in line
            if word != "-DOCSTART-": # I don't read a line, which is started to -DOCSTART-
                if not line.strip(): # When every come empty line, I add the tags and words in lines array.
                    t = ' '.join([tag for tag in tags if len(tag) > 0]) 
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append((t, w))
                    words = []
                    tags = []
                words.append(word)
                tags.append(tag)

        self.datasets.append(lines)
        
        # Read a Test data!
        test_data = open(input_folder+"test.txt", 'r')
        lines = []; words = []; tags = []
        for line in test_data:
            word = line.strip().split(' ')[0] # First string in line
            tag = line.strip().split(' ')[-1] # Last string in line
            if word != "-DOCSTART-": # I don't read a line, which is started to -DOCSTART-
                if not line.strip(): # When every come empty line, I add the tags and words in lines array.
                    t = ' '.join([tag for tag in tags if len(tag) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append((t,w))
                    words = []
                    tags = []
                words.append(word)
                tags.append(tag)
            
        self.datasets.append(lines)

    
    """
    **Arguments**:
    
        :param dataset: A Train Dataset
        :type dataset: A list or tuple

    **Arg. Example**:
    
        >>> dataset = [('B-ORG O B-MISC O O O B-MISC O O', 'EU rejects German call to boycott British lamb .'), 
                       ('B-PER I-PER', 'Peter Blackburn'), ...]
        
    **Explanation**:
    
        >>> Note: Örnek değerleri gerçeği yansıtmamaktadır. Sadece örnek olması açısından yazılmıştır.
        
        Step.1 - Create A Transition Probability:
        (I don't calculate initial probability. Because I added start (<s>) token to my transition_tags.)
            I have 4 structures:
              * each_tag_counts; I calculate the count of each tag in the train dataset 
                and add the result to this array.
                >>> Example: {'B-ORG':185, 'O':35007, 'B-MISC':87, ...}
              * transition_tags; I reprocess each tag in a sentence according to the bigram model 
                and add the result to this array.
                >>> Example: [['<s> B-ORG', 'B-ORG O', 'O B-MISC', 'B-MISC </s>'], ...]
              * transition_tag_counts; I calculate the counts of each pair in transition_tags 
                and add the result to this dictionary.
                >>> Example: {'<s> B-ORG':1250, 'B-ORG O':245, 'O B-MISC':56, ...}
              * transition_prob; Finally, I calculate the probabilities of each pair in transition_tags
                and add the result to this dictionary.
                >>> Example: {'<s> B-ORG':0.55, 'B-ORG O':0.04, 'O B-MISC':0.0023, ...}
            
        Step.2 - Create A Emission Probability:
            I have 3 structures:
              * emissionTag_Word_dict; I classify each word in a sentence 
                according to the tag to which it is linked and add the result to this dictionary.
                >>> Example: {'B-ORG': ['EPR', 'EU', 'European', ...], 'B-LOC': ['BRUSSELS', 'Germany', ...], ...}
              * emission_word_counts; The value part of each tag in emissionTag_Word_dict 
                is the word list with that tag. 
                Here I calculate the counts of the words in this list and add the result to this dictionary.
                >>> Example: {'B-ORG': {'EPR':3, 'EU':5, 'European':2, ...}, 'B-LOC': {'BRUSSELS':12, 'Germany':3, ...}, ...}
              * emission_prob; I calculate the probability of the words in emission_word_counts structure 
                and add the result to this dictionary.
                >>> Example: {'B-ORG': {'EPR':0.2, 'EU':0.4, 'European':0.14, ...}, 'B-LOC': {'BRUSSELS':0.4, 'Germany':0.1, ...}, ...}
             
    """
    def HMM(self, dataset):
        # Tranition Probability Calculating
        each_tag_counts = {}
        
        transition_tags = [] 
        transition_tag_counts = {}
        transition_prob = {}
        
        # Calculate Transition Tags use Bigram Model.
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
        
        array_to_npArr = np.asarray(transition_tags)

        # Calculate Every Binary Transition Tag Counts
        for tags in array_to_npArr:
            each_counts = dict(Counter(tags))
        
            for k,v in each_counts.items():
                if k not in transition_tag_counts:
                    transition_tag_counts[k] = v
                else:
                    transition_tag_counts[k] += 1
        
        # Calculate Each Transition Tag Counts. 
        for tags in array_to_npArr:
            for tag in tags:
                tag = tag.split()
                each_counts = dict(Counter(tag))
            
                for k,v in each_counts.items():
                    if k not in each_tag_counts:
                        each_tag_counts[k] = v
                    else:
                        each_tag_counts[k] += 1
        
        # Calculate Transition Tag Probabilities
        for k,v in transition_tag_counts.items():
            first_tag = k.split()
            first_tag = first_tag[0]
            first_tag_count = each_tag_counts[first_tag]
            trans_prob = transition_tag_counts[k] / each_tag_counts[first_tag]
            transition_prob[k] = trans_prob
        
        
        # Emission Probability Calculating
        emissionTag_Word_dict = {}
        emission_word_counts = {}
        emission_prob = {}
        
        # Calculate Emission Tags with Word.
        for sentence in dataset:
            tags = sentence[0].split()
            words = sentence[1].split()
            
            for i in range(len(tags)):
                if tags[i] not in emissionTag_Word_dict:
                    arr = [words[i]]
                    emissionTag_Word_dict[tags[i]] = arr
                else:
                    emissionTag_Word_dict[tags[i]].append(words[i])
                    
        # Calculate Emission Word Counts.
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
                        
        # Calculate Emission Word Probability.
        for k1,v1 in emission_word_counts.items():
            total_corpus = sum(emission_word_counts[k1].values())
            
            em_prob = {}
            for k2,v2 in v1.items():
                emis_prob = v2 / total_corpus
                em_prob[k2] = emis_prob
                
            emission_prob[k1] = em_prob
            self.V_size += len(v1)
        
        # Return 2 probabilities; transition_prob and emission_prob
        return transition_prob, emission_prob

    
    """
    **Arguments**:
    
        :param transition_prob: A Transition Probability
        :type transition_prob: A dictionary
        
        :param emission_prob: A Emission Probability
        :type emission_prob: A dictionary
        
        :param test_data: A Test dataset
        :type test_data: A list or tuple

    **Arg. Example**:
    
        >>> transition_prob = {'<s> B-ORG':0.55, 'B-ORG O':0.04, 'O B-MISC':0.0023, ...}
        >>> emission_prob = {'B-ORG': {'EPR':0.2, 'EU':0.4, 'European':0.14, ...}, 'B-LOC': {'BRUSSELS':0.4, 'Germany':0.1, ...}, ...}
        >>> test_data = [('B-ORG O B-MISC O O O B-MISC O O', 'EU rejects German call to boycott British lamb .'), 
                         ('B-PER I-PER', 'Peter Blackburn'), ...]
        
    **Explanation**:
    
        ... Note: Sample values ​​do not reflect reality. It is written only as an example.
        
        Step.1 - Changing Probability Dictionaries to Probability Matrix:
            To calculate the Viterbi algorithm, I have converted dictionaries 
            with transition and emission probes into a 2D matrix.
            >>> transition_prob >> transition_matrix: column; first tags, row; second tags
                   <s> | B-LOC |   O  | B-ORG | ......
            B-LOC [0.2 | 0.11  | 0.25 | 0.    | ......]
            O     [0.4 | 0.11  | 0.   | 0.    | ......]
            B-ORG [0.  | 0.    | 0.12 | 0.    | ......]
            ....  [... | ......| .....| ......| ......]
            ....  [... | ......| .....| ......| ......]
            ....  [... | ......| .....| ......| ......] 
            
            >>> emission_prob >> emission_matrix: column; words, row; tags
                   EU   | Germany |  The  | have  | ......
            B-LOC [0.25 | 0.11    | 0.5   | 0.    | ......]
            O     [0.14 | 0.18    | 0.7   | 0.    | ......]
            B-ORG [0.2  | 0.      | 0.1   | 0.    | ......]
            ....  [.... | ........| ......| ......| ......]
            ....  [.... | ........| ......| ......| ......]
            ....  [.... | ........| ......| ......| ......] 
        
        
        Step.2 - Calculate Viterbi:
            ** Example Viterbi Matrix: **
            >>> viterbi_matrix: column: each step; 1 to len(sentence), row; tags
                  start | step-1  | step-2 | step-3 | ......
            B-LOC [0.2  | 0.025   | 0.089  | 0.11   | ......]
            O     [0.4  | 0.1056  | 0.07   | 0.4    | ......]
            B-ORG [0.2  | 0.02    | 0.12   | 0.23   | ......]
            ....  [.... | ........| .......| .......| ......]
            ....  [.... | ........| .......| .......| ......]
            ....  [.... | ........| .......| .......| ......]
            
            
            First of all, I calculate emission prob for each sentence from test_sentences 
            and add them in the emission_matrix, as I mentioned above.
            Next, I fill each cell in the viterbi_matrix with the formula in the below:
            >>> Formül: viterbi_matrix[k][i-1] * transition_matrix[j][k+1] * emission_matrix[j][i-1]
              * viterbi_matrix[k][i-1]: Probability of come the previous tag in Viterbi matrix.
                In the example above, the possibility of the B-LOC tag coming in the previous step for step-2: 0.025 
              * transition_matrix[j][k+1]: The probability that the tag in the cell 
              that it is in comes after the tag I chose in viterbi_matrix[k][i-1].
                In the example above, the probability that comes B-ORG after the O tag: 0.12
              * emission_matrix[j][i-1]: The probability that the word in the step it is in 
              is inside the tag in the cell where it is located.
                In the example above, the possibility of the word 'The' in the B-LOC tag: 0.5
                
            In this way, I fill each cell with the total number of tags, take the highest of them
            and write them to viterbi_matrix[j][i].
            >>> viterbi_matrix[j][i] = max(each_cell)
            
            Then I add the tag corresponding to the highest value in each column of viterbi_matrix,
            which I filled with these calculations, to the predict_tags list.
            
            Now I have guessed tags. Finally, I send this to the accuracy () function and measure the accuracy of the model.
             
    """
    def viterbi(self, transition_prob, emission_prob, test_data):
        # test_data list has 2 index. First one is tags and the other one is sentences.
        test_sentences_tags = [sentences[0] for sentences in test_data]
        
        # The keys of the emission probe correspond to the tags in my data set. 
        # I will do the placement according to these tags while creating the Matrix.
        transition_tags = list(emission_prob.keys())
        
        transition_tags.sort()
        transition_tags.insert(0,"<s>")

        # Transition probability was a dictionary. I turned this into a matrix for easier processing.
        transition_matrix_len = len(transition_tags)
        transition_matrix = np.zeros((transition_matrix_len-1,transition_matrix_len))
        for i in range(transition_matrix_len):
            uniqe_tag = transition_tags[i]
            for j in range((transition_matrix_len)-1):
                tag = uniqe_tag + " " + transition_tags[j+1]
                if tag in transition_prob:
                    transition_matrix[j][i] = transition_prob[tag]
                    
        
        # The keys of the emission probe correspond to the tags in my data set. 
        # I will do the placement according to these tags while creating the Matrix.
        emission_tags = list(emission_prob.keys())
        emission_tags.sort()

        predict_tags = []
        
        # Emission probability was a dictionary. I turned this into a matrix for easier processing.
        row_count = len(emission_tags)
        for test_sentences in test_data:
            sentences = test_sentences[1]
            sentences = sentences.split()
            
            column_count = len(sentences)
            emission_matrix = np.zeros((row_count,column_count))
            for word in sentences:
                i = sentences.index(word)
                for tag in emission_tags:
                    j = emission_tags.index(tag)
                    
                    """ In this section;
                        If the word from the test sentence is in my train data set, 
                        I draw the probability from emission_prob.
                        If that word is not in my train data set, 
                        I calculating smooth probability it.
                    """
                    if word in emission_prob[tag]:
                        emission_matrix[j][i] = emission_prob[tag][word]
                    else:
                        emission_prob[tag][word] = 1 / (len(sentences) + self.V_size)
                        emission_matrix[j][i] = emission_prob[tag][word]
            
            
            tag_path_array = [] # The predicted tag list of that sentence in the Viterbi matrix.
            
            viterbi_matrix = np.zeros((row_count,column_count+2))
            """ The first column of the Viterbi matrix is ​​an initial probability. 
            I'm writing this to the Matrix. """
            for tag in emission_tags:
                i = emission_tags.index(tag)
                viterbi_matrix[i][0] = transition_matrix[i][0]
            
            """ 
                Now it's time to fill the matrix for each word
                If we imagine the Viterbi matrix;
                  'i' columns,
                  'j' rows,
                  'each_cell' represents each cell.
                                  
                  The process 'viterbi_matrix [j] [i] = max (each_cell)' selects the highest value
                   found in each cell and prints it in that cell.
            """
            for word in sentences:
                i = sentences.index(word)
                i = i + 1
                for tag in emission_tags:
                    j = emission_tags.index(tag)
                    each_cell = np.zeros(len(emission_tags))
                    if emission_matrix[j][i-1] != 0.0:
                        for k in range(len(emission_tags)):
                            if viterbi_matrix[k][i-1] != 0.0:
                                if transition_matrix[j][k+1] != 0.0:
                                        
                                    # print("Transition: ",transition_tags[k+1],"-",transition_tags[j+1],
                                        #     ", Emission: ",emission_tags[j],"-",word,
                                        #     ", viterbi: ",viterbi_matrix[k][i-1],
                                        #     ", transition: ",transition_matrix[j][k+1],
                                        #     ", emission: ",emission_matrix[j][i-1])
                                        
                                    result = viterbi_matrix[k][i-1] * transition_matrix[j][k+1] * emission_matrix[j][i-1]
                                    each_cell[k] = result
                        
                    viterbi_matrix[j][i] = max(each_cell)
                    
            """ When the end of the sentence is reached,
                I apply the following formula to calculate the 'end' (bottom right cell) in the Viterbi matrix. """
            end_result = np.zeros(len(emission_tags))
            for tag in emission_tags:
                i = emission_tags.index(tag)
                result = viterbi_matrix[i][len(viterbi_matrix[0])-2] * transition_matrix[len(transition_matrix)-1][i+1]
                
                # print("viterbi: ",viterbi_matrix[i][len(viterbi_matrix[0])-2])
                # print("transition: ",transition_matrix[i+1][len(transition_matrix[0])-1] )
                
                end_result[i]
                
            viterbi_matrix[viterbi_matrix == 0] = -1
            viterbi_matrix[len(viterbi_matrix)-1][len(viterbi_matrix[0])-1] = max(end_result)
            
            for i in range(len(viterbi_matrix[0])-2):
                argmax = np.argmax(viterbi_matrix[:,i+1])
                if argmax == 0:
                    argmax = 8
                tag_path_array.append(emission_tags[argmax])
        

            predict_tags.append(tag_path_array)
        
        """ I send both the tags I read from the test dataset 
        and the tags I guess to the accuracy function. """
        self.accuracy(test_sentences_tags, predict_tags)
            
            

    """
    **Arguments**:
    
        :param test_sentences_tags: Tag list of each sentence that I obtained from the test dataset
        :type test_sentences_tags: A list or tuple
        
        :param predict_tags: The list of tags I guessed for each sentence I get from the test dataset
        :type predict_tags: A list or tuple

    **Arg. Example**:
    
        ... Note: Sample values ​​do not reflect reality. It is written only as an example.
        
        >>> test_sentences_tags = [('B-ORG O B-MISC O O O B-MISC O O'), ('B-PER I-PER'), ...]
        >>> predict_tags = [('B-LOC O 0 O O O B-MISC O O'), ('0 I-LOC'), ...]
        
    **Explanation**:
    
        My function parameters are 2-dimensional arrays. In this respect, 
        I processing each 1-dimensional tag array in the for a loop.
        I convert the arrays in my hand to np-array for easier operation.
        With the formula below, I assign the number of matching tags 
        in the binary "test_sentences_tags [i], predict_tags [i]" to the total_match_tag variable.
        >>> total_match_tag += np.sum(test_sentences_tags[i] == predict_tags[i])
        
        And by dividing the total number of matching tags by the total number of tags, I calculate the accuracy.
             
    """
    def accuracy(self, test_sentences_tags, predict_tags):
        total_match_tag = 0
        total_tags = 0
        
        file = open('submission.txt', 'w')
        file.write("Id,Category\n")
        
        index = 1
        for i in range(len(test_sentences_tags)):
            test_sentences_tags[i] = test_sentences_tags[i].split()
            test_sentences_tags[i] = np.asarray(test_sentences_tags[i])
            
            predict_tags[i] = np.asarray(predict_tags[i])
            
            # for x in range(len(predict_tags[i])):
            #     file.write(str(index)+","+predict_tags[i][x]+"\n")
            #     index += 1
                
            testSent = ' '.join([str(elem) for elem in test_sentences_tags[i]])
            predict = ' '.join([str(elem) for elem in predict_tags[i]])
            file.write(testSent+"\n"+predict+"\n----------\n")
            
            total_match_tag += np.sum(test_sentences_tags[i] == predict_tags[i])
            total_tags += len(test_sentences_tags[i])

classHMM = HMM()

classHMM.dataset("./Assignment2/dataset/")

train = classHMM.datasets[0]
test = classHMM.datasets[1]

model = classHMM.HMM(train)

viterbi = classHMM.viterbi(model[0], model[1], test)