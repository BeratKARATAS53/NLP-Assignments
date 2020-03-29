# trigram_list = [["a b c", "a d c"],["b c c", "c d a", "a d e"],["b c d", "c d a", "a d c", "a b c", "a d c"]]

# tri_count = {}

# for data in trigram_list:
#     for i in range(len(data)):
#         text = data[i].split()
#         first_two_word = text[0] + " " + text[1]
#         if first_two_word == "a d":
#             if data[i] not in tri_count:
#                 tri_count[data[i]] = 1
#             else:
#                 tri_count[data[i]] += 1
                

# print(tri_count)
# a= 2
# new = []
# if a >= 2:
#     new.append(1)
# else:
#     new.append(2)

# new = [["word i","and i","word b","i word"],["word i","i word","word i"],["word a"]]
# n_c = {}
# for ne in new:
#     if "word i" in ne:
#         if "word i" not in n_c:
#             n_c["word i"] = 1
#         else:
#             n_c["word i"] += 1

n = 100
m = 200
v = 100
print(n/m)
print((n+1)/(m+v))