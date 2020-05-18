from collections import defaultdict
a = 'ELLLE'
print(len(a))
a = 'ELLLE eellle'
print(len(a))
a = 'ELLLE eellle'
print(len(a.split()))

a = {'Adj': ['fine', 'delicious', 'beautiful', 'old'], 'Verb': ['ate', 'wanted', 'kissed', 'washed', 'pickled', 'is', 'prefer', 'like', 'need', 'want'], 'Pronoun': ['me', 'i', 'you', 'it'], 'Det': ['the', 'a', 'every', 'this', 'that'], 'Noun': ['president', 'sandwich', 'pickle', 'mouse', 'floor'], 'Prep': ['with', 'on', 'under', 'in', 'to', 'from']}

print(a)
aa = defaultdict(list)
for k,v in a.items():
    for r in v:
        aa[k].append(tuple(r.split()))
print(aa)