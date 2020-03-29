import re 

def bigramEstimation(file):
    '''A very basic solution for the sake of illustration.
       It can be calculated in a more sophesticated way.
       '''

    lst = [] # This will contain the tokens
    unigrams = {} # for unigrams and their counts
    bigrams = {} # for bigrams and their counts

    # 1. Read the textfile, split it into a list
    text = open(file, 'r').read()
    newString = text.lower()
    newString = re.sub(r'[^A-Za-z. _-]', '', newString)
    lst = newString.strip().split()
    print ('Read ', len(lst), ' tokens...')

    print(lst)
    
    del text # No further need for text var

    # 2. Generate unigrams frequencies
    for l in lst:
        if not l in unigrams:
            unigrams[l] = 1
        else:
            unigrams[l] += 1

    print ('Generated ', len(unigrams), ' unigrams...'  )

    # 3. Generate bigrams with frequencies
    for i in range(len(lst) - 1):
        temp = (lst[i], lst[i+1]) # Tuples are easier to reuse than nested lists
        print(temp)
        if not temp in bigrams:
            bigrams[temp] = 1
        else:
            bigrams[temp] += 1

    print ('Generated ', len(bigrams), ' bigrams...')

    # Now Hidden Markov Model
    # bigramProb = (Count(bigram) / Count(first_word)) + (Count(first_word)/ total_words_in_corpus)
    # A few things we need to keep in mind
    total_corpus = sum(unigrams.values())
    # You can add smoothed estimation if you want


    print ("Calculating bigram probabilities and saving to file...")

    # Comment the following 4 lines if you do not want the header in the file. 
    with open("bigrams.txt", 'a') as out:
        out.write('Bigram' + '\t' + 'Bigram Count' + '\t' + 'Uni Count' + '\t' + 'Bigram Prob')
        out.write('\n')
        out.close()


    # for k,v in bigrams.items():
    #     # first_word = helle in ('hello', 'world')
    #     first_word = k[0]
    #     first_word_count = unigrams[first_word]
    #     bi_prob = bigrams[k] / unigrams[first_word]
    #     uni_prob = unigrams[first_word] / total_corpus

    #     final_prob = bi_prob + uni_prob
    #     with open("bigrams.txt", 'a') as out:
    #         out.write(k[0] + ' ' + k[1] + '\t' + str(v) + '\t' + str(first_word_count) + '\t' + str(final_prob)) # Delete whatever you don't want to print into a file
    #         out.write('\n')
    #         out.close()




# Callings
bigramEstimation("C:/Users/Berat/Documents/GitHub/NLP-Assignments/Assignment1/CBTest/data/cbt_train.txt")