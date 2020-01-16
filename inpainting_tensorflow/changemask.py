import numpy as np

def changeeye(mask):
    for i in range(512):
        for j in range(512):
            if mask[i][j]==1:
                for k in range(30,50):
#                     if i-k>=0 and mask[i-k][j]==0:
#                         mask[i-k][j]=2
                    if j-k>=0 and mask[i][j-k]==0:
                        mask[i][j-k]=2
                    if j+k<512 and mask[i][j+k]==0:
                        mask[i][j+k]=2
                for k in range(30,50):
                    if i+k<512 and mask[i+k][j]==0:
                        mask[i+k][j]=3
    for i in range(512):
        for j in range(512):
            if mask[i][j]==3:
                for k in range(1,10):
                    if j-k>=0 and mask[i][j-k]==0:
                        mask[i][j-k]=4
                    if j+k<512 and mask[i][j+k]==0:
                        mask[i][j+k]=4
            if mask[i][j]==2:
                for k in range(20,35):
                    if i+k<512 and mask[i+k][j]==0:
                        mask[i+k][j]=5
    for i in range(512):
        for j in range(512):
            if mask[i][j]==1:
                mask[i][j]=0
            elif mask[i][j]==2:
                mask[i][j]=1
            elif mask[i][j]==3:
                mask[i][j]=1
            elif mask[i][j]==4:
                mask[i][j]=1
            elif mask[i][j]==5:
                mask[i][j]=1
    return mask

def changehead(mask):
    for i in range(512):
        for j in range(512):
            if mask[i][j]==1:
                for k in range(30,50):
#                     if i-k>=0 and mask[i-k][j]==0:
#                         mask[i-k][j]=2
                    if j-k>=0 and mask[i][j-k]==0:
                        mask[i][j-k]=2
                    if j+k<512 and mask[i][j+k]==0:
                        mask[i][j+k]=2
    for i in range(512):
        for j in range(512):
            if mask[i][j]==2:
                for k in range(30,45):
                    if i-k>0 and mask[i+k][j]==0:
                        mask[i-k][j]=3
    for i in range(512):
        for j in range(512):
            if mask[i][j]==1:
                mask[i][j]=0
            elif mask[i][j]==2:
                mask[i][j]=0
            elif mask[i][j]==3:
                mask[i][j]=1
    return mask

def changenose(mask):
    for i in range(512):
        for j in range(512):
            if mask[i][j]==1:
                for k in range(1,10):
                    if i-k>=0 and mask[i-k][j]==0:
                        mask[i-k][j]=2
                    if i+k<512 and mask[i+k][j]==0:
                        mask[i+k][j]=2
                    if j-k>=0 and mask[i][j-k]==0:
                        mask[i][j-k]=2
                    if j+k<512 and mask[i][j+k]==0:
                        mask[i][j+k]=2
    for i in range(512):
        for j in range(512):
            if mask[i][j]==1:
                mask[i][j]=1
            elif mask[i][j]==2:
                mask[i][j]=1
    return mask

def changemouth(mask):
    for i in range(512):
        for j in range(512):
            if mask[i][j]==1:
                for k in range(1,10):
                    if i-k>=0 and mask[i-k][j]==0:
                        mask[i-k][j]=2
                    if i+k<512 and mask[i+k][j]==0:
                        mask[i+k][j]=2
                    if j-k>=0 and mask[i][j-k]==0:
                        mask[i][j-k]=2
                    if j+k<512 and mask[i][j+k]==0:
                        mask[i][j+k]=2
    for i in range(512):
        for j in range(512):
            if mask[i][j]==1:
                mask[i][j]=1
            elif mask[i][j]==2:
                mask[i][j]=1
    return mask