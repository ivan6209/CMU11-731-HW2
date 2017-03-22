from collections import Counter, defaultdict
import numpy as np
import sys

with open(sys.argv[2], 'r') as f:
    lines = f.readlines()
validen = [[w.lower() for w in l.strip().split()] for l in lines]
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
validde = [[w.lower() for w in l.strip().split()] for l in lines]

with open(sys.argv[3], 'r') as f:
    lines = f.readlines()
alignments = [ [ tuple(map(int, a_b.split('-'))) for a_b in l.strip().split()] for l in lines]



def quasi_consec(tp, aligned_words):
    if len(tp) < 2:
        return True
    tp = [aligned_words[idx] for idx in tp]
    for jj in range(len(tp)-1):
        if tp[jj+1]-tp[jj] != 1:
            return False
    return True

def phrase_extract(e, f, alignment, max_len=3):
    extracted_phrases = []
    eAlignmentDict = defaultdict(list)
    fAlignmentDict = defaultdict(list)
    for a,b in alignment:
        eAlignmentDict[a].append(b)
        fAlignmentDict[b].append(a)
    f_aligned_words = set([b for _,b in alignment])
    f_aligned_words = dict(zip(sorted(list(f_aligned_words)), range(len(f_aligned_words))))
    # Loop over all substrings in the E
    for i1 in range(len(e)):
        tp = []
        for i2 in range(i1, min(len(e), i1+max_len)):
            tp.extend(eAlignmentDict[i2])
            tp = sorted(list(set(tp)))
            # Get all positions in F that correspond to the substring from i1 to i2 in E (inclusive)
            if len(tp) != 0 and quasi_consec(tp, f_aligned_words):
                j1, j2 = tp[0], tp[-1]
                if j2-j1 >= max_len:
                    continue
                sp = []
                for f_ in tp:
                    sp.extend(fAlignmentDict[f_])
                sp = sorted(list(set(sp)))
                # Get all positions in E that correspond to the substring from j1 to j2 in F (inclusive)

                if len(sp) != 0 and set(sp).issubset(range(i1,i2+1)): # Check that all elements in sp fall between i1 and i2 (inclusive)
                    e_phrase = e[i1:i2+1]
                    f_phrase = f[j1:j2+1]
                    extracted_phrases.append((e_phrase, f_phrase))
              # Extend source phrase by adding unaligned words
                    while j1 >= 0 and len(fAlignmentDict[j1])==0: # Check that j1 is unaligned
                        j_prime = j2
                        while j_prime < min(len(f), j1+max_len) and len(fAlignmentDict[j])==0: # Check that j2 is unaligned
                            f_phrase = f[j1:j_prime+1]
                            extracted_phrases.append((e_phrase, f_phrase))
                            j_prime += 1
                        j1 -= 1

    return extracted_phrases


phrase_dict = defaultdict(list)
for jj, alignment in enumerate(alignments):
    phrases = phrase_extract(validen[jj], validde[jj], alignment)
    for e, f in phrases:
        phrase_dict[' '.join(e)].append(' '.join(f))


phrases = []
for e, fs in phrase_dict.items():
    lfs = len(fs)
    if lfs == 1:
        phrases.append((e, fs[0], 0.0))
    else:
        lfs = float(lfs)
        counter = Counter(fs)
        for f, ct in counter.items():
            phrases.append((e, f, -np.log(ct/lfs)))



with open(sys.argv[4], 'w') as f:
    for phrase in phrases:
        f.write('%s\t%s\t%.4f\n'%(phrase[1], phrase[0], phrase[2]))
    
