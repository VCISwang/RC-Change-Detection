import os
import numpy as np
import math
import random
random.seed(10)

file_list       = os.path.join("train(org).txt")
img_name_list   = np.loadtxt(file_list, dtype=str)
random.shuffle(img_name_list)

N_total = len(img_name_list)
print(f'Total number of images in the dataset: {N_total}')

for per in range(10,90,10):
    print(f'===== Creating {per}% split =====')
    N_sup = math.floor((per/100)*len(img_name_list))
    print(f'Number of supervised images: {N_sup}.')

    sup_img_name_list   = img_name_list[0:N_sup]
    unsup_img_name_list = img_name_list[N_sup:]

    textfile_sup = open(str(per)+"_train_supervised.txt", "w")
    for element in sup_img_name_list:
        textfile_sup.write(element + "\n")
    textfile_sup.close()

    textfile_unsup = open(str(per)+"_train_unsupervised.txt", "w")
    for element in unsup_img_name_list:
        textfile_unsup.write(element + "\n")
    textfile_unsup.close()

    
    
    

   

