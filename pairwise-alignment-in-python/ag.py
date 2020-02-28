import time
from concurrent.futures import ThreadPoolExecutor as TPE
import numpy as np
match_award      = 10
mismatch_penalty = -5
gap_penalty      = -5 # both for opening and extanding

def match_score(alpha, beta):
    if alpha == beta:
        return match_award
    elif alpha == '-' or beta == '-':
        return gap_penalty
    else:
        return mismatch_penalty

def finalize(align1, align2):
    i,j = 0,0 
    #calcuate identity, score and aligned sequeces
    symbol = ''
    identity = 0
    for i in range(0,len(align1)):
        # if two AAs are the same, then output the letter
        if align1[i] == align2[i]:
            identity = identity + 1
    identity = float(identity) / len(align1) * 100
    return(identity)

def needle(seq1, seq2):
    return(finalize(seq1,seq2))
