tags = [[1,2],[3,4],[4,5,6]]

tag2 = []

bigram = (' '.join(tags[number:number + 1]) for number in range(0, len(tags)))
print(bigram)