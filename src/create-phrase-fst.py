import sys
from collections import defaultdict
import numpy as np

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
lines = [l.strip().split('\t') for l in lines]
phrases2 = []
for l in lines:
    phrases2.append((l[1], l[0], float(l[2])))


with open(sys.argv[2], 'w') as f:
    f_state = defaultdict(lambda:len(f_state))
    for phrase in phrases2:
        previous_state = f_state['init']
        prefix=''
        for f_ in phrase[1].split():
            if prefix+f_ in f_state:
                previous_state = f_state[prefix+f_]
            else:
                state = f_state[prefix+f_]
                f.write("%d %d %s <eps>\n"%(previous_state, state, f_))
                previous_state = state
            prefix += ' '+f_
        prefix += ':'
        for e_ in phrase[0].split():
            if prefix+e_ in f_state:
                previous_state = f_state[prefix+e_]
            else:
                state = f_state[prefix+e_]
                f.write("%d %d <eps> %s\n"%(previous_state, state, e_))
                previous_state = state
            prefix += ' '+e_
            
        f.write("%d %d <eps> <eps> %.4f\n"%(previous_state, f_state['init'], np.abs(phrase[2])) )
    f.write("0 0 </s> </s>\n")
    f.write("0 0 <unk> <unk>\n")
    f.write("0")
