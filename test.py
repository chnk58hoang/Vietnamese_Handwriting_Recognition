import sentencepiece as spm


x = [[1,2,3,0],[1,2,0,0],[1,0,0,2]]
clean_x = []
for xx in x:
    xx = [k for k in xx if k != 0]
    clean_x.append(xx)

print(clean_x)