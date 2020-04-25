import sys
import re

import math
import random
from collections import defaultdict

import numpy as np
from collections import Counter

class CYK():
    
    def __init__(self):
        # rules_dict is list that use for generate a random sentences. It's mixed of cfg_rules and vocabulary lists.
        self.rules_dict = defaultdict(list)
    
    """
    **Arguments**:
    
        :param input_folder: File path to read Train and Test datasets
        :type input_folder: A string
        
    **Arg. Example**:
    
        >>> input_folder = ./Assignment3/cfg.gr
        
    **Explanation**:
    
        This function have 3 steps.
        
        >>> Step-1:
            Dosyadan ilk kelimesi '#' ve 'ROOT' olmayan satırları okuyorum ve 'lines' array'ine yazıyorum.
        
        >>> Step-2:
            Elimdeki 'lines' array'inde hem cfg rules'lar hem de vocabulary'ler var. CYKParser() fonksiyonunda kullanmam için bunları ayırmam lazım.
            O yüzden her bir satırı bir for döngüsünde kuralların right-hand-side (rhs) kısmına bakarak ayırma işlemini gerçekleştiriyorum. Eğer rhs kısmı küçük harften oluşuyorsa
            o satır vocabulary'yi temil eder, o yüzden o satırı 'vocabulary' dict'ine ekliyorum. Diğer durumda ise 'rules' dict'ine ekliyorum.
            
        >>> Step-3:
            randsentence() fonksiyonunda daha kolay kullanabilmek için rules ve vocabulary dictionary'lerini rules_dict adındaki defaultdict türündeki yapıya birleştiriyorum.
            Neden daha kolay olduğunu randsentence() fonksiyonunda bahsedeceğim.
            
        Sonuç olarak elimde 3 tane dict var:
            ... rules = {       'VP': ['Verb NP'], 
                                'Noun': ['Adj Noun'], 
                                'S': ['NP VP'], 
                                'PP': ['Prep NP'], 
                                'NP': ['Det Noun', 'Pronoun', 'NP PP']  }
            
            ... vocabulary =  { 'Prep': ['with', 'on', 'under', 'in', 'to', 'from'], 
                                'Verb': ['ate', 'wanted', 'kissed', 'washed', 'pickled', 'is', 'prefer', 'like', 'need', 'want'], 
                                'Noun': ['president', 'sandwich', 'pickle', 'mouse', 'floor'], 
                                'Adj': ['fine', 'delicious', 'beautiful', 'old'], 
                                'Pronoun': ['me', 'i', 'you', 'it'], 
                                'Det': ['the', 'a', 'every', 'this', 'that']    } 
                                
            ... rules_dict =  { 'S': [('NP', 'VP')], 
                                'PP': [('Prep', 'NP')], 
                                'Adj': [('fine',), ('delicious',), ('beautiful',), ('old',)], 
                                'VP': [('Verb', 'NP')], 
                                'NP': [('Det', 'Noun'), ('Pronoun',), ('NP', 'PP')],
                                'Noun': [('Adj', 'Noun'), ('president',), ('sandwich',), ('pickle',), ('mouse',), ('floor',)], 
                                'Verb': [('ate',), ('wanted',), ('kissed',), ('washed',), ('pickled',), ('is',), ('prefer',), ('like',), ('need',), ('want',)], 
                                'Det': [('the',), ('a',), ('every',), ('this',), ('that',)], 
                                'Pronoun': [('me',), ('i',), ('you',), ('it',)],
                                'Prep': [('with',), ('on',), ('under',), ('in',), ('to',), ('from',)]   }
        
    """
    def rules(self, folder_path):
        """ Step-1: Read a cfg rules file. """
        cfg_rules = open(folder_path, 'r')
        lines = []
        for line in cfg_rules:
            word = line.strip().split('\t')
            each_char = word[0].split(' ')
            if each_char[0] != '#':
                if each_char[0] != 'ROOT':
                    if line.strip():
                        lines.append(word)
        
        """ Step-2: Lines split to rules and vocabulary dicts. """
        rules = {}
        vocabulary = {}
        for line in lines:
            lhs = line[0].strip()
            rhs = line[1]
            for word in rhs.split():
                word = word.strip()
                if word.islower(): # vocabulary
                    if lhs not in vocabulary:
                        vocabulary[lhs] = [word]
                    else:
                        vocabulary[lhs].append(word)
                else:
                    if lhs not in rules:
                        rules[lhs] = [rhs]
                    else:
                        if rhs not in rules[lhs]:
                            rules[lhs].append(rhs)
                    
        self.cfg_rules = rules
        self.cfg_vocabs = vocabulary
        
        """ Step-3: Mixed rules and vocabulary dicts to rules_dict. """
        for key, value in rules.items():
            for each in value:
                self.rules_dict[key].append(tuple(each.split()))
            
        for key, value in vocabulary.items():
            for each in value:
                self.rules_dict[key].append(tuple(each.split()))
    
    
    """
    **Arguments**:

        :param symbol: non-terminal variable such as 'S', 'NP', 'Verb', ...
        :type symbol: str
        
        :return sentence: A random generate sentence
        :type sentence: str
        
    **Explanation**:
    
        randsentence() fonksiyonunu recursive olarak tanımladım. 
        Öncelikle başlangıç sembol'ü olan 'S' değerini alıp rules_dict'den o non_terminal değerinin sahip olduğu rastgele bir terminal değeri seçiyorum.
        Sonrasında bu terminal değerinin içindeki her bir kelimeyi kontrol ediyorum;
            ... Eğer bu kelime bir non-terminal değer ise bu değeri tekrardan fonksiyona gönderiyorum.
            ... Eğer non-terminal değil ise bu değerin bir kelime olduğunu anlıyorum ve bunu 'sentence' string'ine ekliyorum.
        
        !!  Note: Normalde satır 140'daki if bloğu ile cümledeki kelime sayısının 6'yı geçmemesini sağlamam gerekiyordu. Ama bu durum bazen sağlanmıyor.
            Sebebi ise fonksiyon her recursive olarak çağırıldığında sentence değerinin sıfırlanması. 
            Buna rağmen kaldırmadım, çünkü oluşturduğum 10 cümleden 1-2 tanesi 7 ve üzeri kelime sayısına sahip oluyor.
            Eğer kaldırırsam 10 cümleden sadece 1-2 tanesi istediğim 6 ve altı kelime sayısına sahip oluyor.
            Bu yüzden o kısım %100 verimde çalışmasa da istediğime yakın sonuçlar alıyorum.
        
        !!! Dosya okunurken bahsettiğim ve neden elimdeki 'rules' ve 'vocabulary' dict'lerini birleştirdiğimin cevabı randsentence() fonksiyonunun yapısını recursive olarak tanımlamamdır.
            Yani fonksiyonu recursive olarak çağırdığım her adımda gelen symbol değerinin rules mu yoksa vocabulary mi olduğunu kontrol etmek yerine bu 2 dict'i birleştirmek daha kolay işlem yapmamı sağladı.
    
    """
    def randsentence(self, symbol):
        sentence = ''
            
        # Step-1: Selected the first rule from 'S' key 
        rand_rule = random.choice(self.rules_dict[symbol])

        for each_rule in rand_rule:
            # Cümledeki kelime sayısı 3'den büyük 6'dan küçük olsun koşulu ama %90 verimle çalışmıyor.
            if (len(sentence.split()) > 3) and (len(sentence.split()) < 6):
                break
            elif each_rule in self.rules_dict: # Eğer seçtiğim kelime bir non-terminal değer ise fonksiyona geri gönder.
                sentence += self.randsentence(each_rule)
            else: # Eğer seçtiğim kelime bir terminal değer ise sentence'a ekleme yap.
                sentence += each_rule + ' '
        
        return sentence
        
    """
    **Arguments**:

        :param generate_sentence: A Random Generated Sentence
        :type generate_sentence: str
        
        :return cyk_matrix: O cümlenin cyk matrix'i
        :return cyk_matrix: 2d np array
        
    **Arg. Example**:
    
        >>> generate_sentence: the mouse on every pickle 
        >>> cyk_matrix:
                [['Det' 'Noun' 'Prep' 'Det' 'Noun']
                ['NP' 'X' 'X' 'NP' None]
                ['X' 'X' 'PP' None None]
                ['X' 'X' None None None]
                ['NP' None None None None]]
                
    **Explanation**:

        >>> Step-1:  ------ [line 252]
            CYK parser algoritmasına başlamadan önce elimdeki cümlenin her bir kelimesin hangi türe ait olduğunu bulmam lazım. Bu kısımda vocabulary dict'inden yararlanıyorum.
            Ve sonucu sentence_type array'ine yazıyorum.
        
        >>> Step-2:   ------ [line 267-274]
            CYK matrix'imin ilk satırını cümledeki kelimelerimin türleriyle dolduruyorum. İlk adımı ayırmış olmamın sebebi diğer adımlarda bir formül uygulayacak olmam.
            
        >>> Other Steps:
            Şimdi diğer adımlarda aşağıdaki formülü uyguluyorum:
            ... Formula: Xrow,column = (Xm,column)(Xrow-(m+1),column+m+1) ---- m: satır sayısı aralığı [0-row], row: satır sayısı [0-count(sentence)], column: sütun sayısı [0-count(sentence)]

            ... Formula Example:
                >>> First Row: X1,0 = (X0,0)(X0,1) ---------------------    m: 0, row: 1, column: 0
                >>> Second Row X2,0 = (X0,0)(X0,1) U (X1,0)(X0,2) ----    m: [0-1], row: 2, column: 0
                .
                .
                .
        
    **CYK Parser Vizualization**:

    ... sentence: the mouse kissed a mouse 
        >>> row=0 | 'the', 'mouse', 'kissed', 'a', 'mouse'
        >>> row=1 | 'the mouse', 'mouse kissed', 'kissed a', 'a mouse'
        >>> row=2 | 'the mouse kissed', 'mouse kissed a', 'kissed a mouse'
        >>> row=3 | 'the mouse kissed a', 'mouse kissed a mouse'
        >>> row=4 | 'the mouse kissed a mouse'
    
    ------------------------------------------
    ... Step-1: --- row=0                                 
        'the  mouse   kissed  a    mouse'
        'Det   Noun    Verb  Det   Noun'
                     
    ... Step-2: --- row=1
        'the mouse'     - X1,0 = (X0,0)(X0,1) -> (Det)(Noun)  => NP
        'mouse kissed'  - X1,1 = (X0,1)(X0,2) -> (Noun)(Verb) => Empty (X)
        
    Step-1                                            Step-2
       _______                                           _______
    4 |_______|________                               4 |_______|________
    3 |_______|________|________                      3 |_______|________|________
    2 |_______|________|________|_____                2 |_______|________|________|_____ 
    1 |_______|________|________|_____|_______        1 |___NP__|____X___|____X___|_NP__|_______
    0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__|       0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__|
         the    mouse    kissed    a    mouse              the    mouse    kissed    a    mouse
    ------------------------------------------
    *********
    ------------------------------------------
    ... Step-3: --- row=2
        'the mouse kissed'  - X2,0 = (X0,0)(X1,1) U (X1,0)(X0,2) -> (Det)(X) U (NP)(Verb) => Empty (X)
        'mouse kissed a'    - X2,1 = (X0,1)(X1,2) U (X1,1)(X0,3) -> (Noun)(X) U (X)(Det)  => Empty (X)
        
    ... Step-4: --- row=3
        'the mouse kissed a'    - X3,0 = (X0,0)(X2,1) U (X1,0)(X1,2) U (X2,0)(X0,3) -> (Det)(X) U (NP)(X) U (X)(Verb)   => Empty (X)
        'mouse kissed a mouse'  - X3,1 = (X0,1)(X2,2) U (X1,1)(X1,3) U (X2,1)(X0,4) -> (Noun)(VP) U (X)(NP) U (X)(Det)  => Empty (X)
        
     Step-3                                                  Step-4
       _______                                           _______
    4 |_______|________                               4 |_______|________
    3 |_______|________|________                      3 |___X___|___X____|________
    2 |___X___|___X____|___VP___|_____                2 |___X___|___X____|___VP___|_____
    1 |___NP__|___X____|___X____|_NP__|_______        1 |___NP__|____X___|____X___|_NP__|_______
    0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__|       0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__|
         the    mouse    kissed    a    mouse              the    mouse    kissed    a    mouse
    ------------------------------------------
    *********
    ------------------------------------------
    ... Step-5: --- row=4
        'the mouse kissed a mouse'  - X4,0 = (X0,0)(X3,1) U (X1,0)(X2,2) U (X2,0)(X1,3) U (X3,0)(X0,4) -> (Det)(X) U (NP)(VP) U (X)(NP) U (X)(Noun) => S
        
     Step-5
       _______ 
    4 |___S___|________
    3 |___X___|___X____|________ 
    2 |___X___|___X____|___VP___|_____ 
    1 |___NP__|___X____|___X____|_NP__|_______ 
    0 |__Det__|__Noun__|__Verb__|_Det_|_Noun__| 
         the    mouse    kissed    a    mouse      

    ------------------------------------------
         
    This is my cyk_matrix:
        [       0      1      2       3        4
        0    ['Det' 'Noun' 'Verb'   'Det'   'Noun']
        1    ['NP'  'X'     'X'     'NP'     None]
        2    ['X'   'X'     'VP'    None     None]
        3    ['X'   'X'     None    None     None]
        4    ['S'    None   None    None     None]
        ]
    """
    def CYKParser(self, generated_sentence):
        sentence_type = []
        for word in generated_sentence.split(): # Step-1
            sentence_type.append([key for key, value in self.cfg_vocabs.items() if word in value])
        
        for i in range(len(sentence_type)):
            sentence_type[i] = sentence_type[i][0]
            
        # Started filling the cyk_matrix
        generated_sentence = generated_sentence.split()
        length = len(generated_sentence)
        cyk_matrix = np.empty((length, length), dtype=object)
        
        sentence_type_dict = {}
        
        for row in range(length):
            index = length - row
            if row == 0: # Step-2: Filling the first row
                for column in range(index):
                    word = generated_sentence[column:row+column+1]
                    t = ' '.join([tag for tag in word if len(tag) > 0])
                    
                    cyk_matrix[row][column] = sentence_type[column]
                    sentence_type_dict[t] = sentence_type[column]
            else: # Other Steps
                for column in range(index):
                    word = generated_sentence[column:row+column+1]
                    t = ' '.join([tag for tag in word if len(tag) > 0])
                    
                    result = []
                    for m in range(row):
                        x1 = cyk_matrix[m][column]
                        x2 = cyk_matrix[row-(m+1)][column+m+1]
                        result_cell = x1 + " " + x2
                        result.append(result_cell)
                    
                    self.result_cell = 'X'
                    for k,v in self.cfg_rules.items():
                        for res in result:
                            if res in v:
                                self.result_cell = k
                        
                    cyk_matrix[row][column] = self.result_cell
        
        return cyk_matrix


classCYK = CYK()

classCYK.rules("./Assignment3/cfg.gr")

file_output = open("output.txt","w")

random_sentences = []
for i in range(10):
    sentence = classCYK.randsentence('S')
    file_output.write(sentence+"\n")
    random_sentences.append(sentence)

file_output.close()

cyk_parser = []
for rand_sentence in random_sentences:
    cyk_parser.append(classCYK.CYKParser(rand_sentence))

i = 0
for parse in cyk_parser:
    print(random_sentences[i])
    i += 1
    print(parse)
    if 'S' in parse[len(parse)-1][0]:
        print("It's in this language")
    else:
        print("It's not in this language!")