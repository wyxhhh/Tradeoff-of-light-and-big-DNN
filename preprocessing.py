import csv
import os
import numpy as np
import random


# a process for converting raw data to csv. format which can be used directly for training 
header = ["class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9", "class10", "comp1", "comp2", "size", "score", "label"]
qfile = 'training_data.csv'
f = open(qfile, "w", newline='')
writer = csv.writer(f)
writer.writerows([header])
lamda = 0.1
p = 0.5

for n in range(1, 9):
    for m in range(10000, 70000, 10000):
        name = "./weights/training_data_Mini"+str(n)+'_'+str(m)+".txt"
        f = open(name, 'r')
        y = list()
        while True:
            lines = f.readline()
            if not lines:
                break
            li = eval(lines)
            x = list()
            for i in range(10):
                x.append(li[i])
            # the normalization step
            x.append(li[10]/668830.0)
            x.append(li[11]/109295.0)
            x.append(li[12]/60000.0)
            if li[13] == 1:
                x.append(li[14] - lamda*li[15])
            else:
                x.append(li[14] - lamda*li[15] - p)
            x.append(li[13])
            y.append(x)
        writer.writerows(y)

