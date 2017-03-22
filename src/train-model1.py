
# coding: utf-8

# In[1]:

import pywrapfst as fst
import pickle, time, logging, logging.handlers, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np


# In[2]:

log = logging.getLogger('')
log.setLevel(logging.INFO)
logFormat = logging.Formatter("%(asctime)s-%(levelname)s: %(message)s")  # %(name)s 

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logFormat)
log.addHandler(ch)

fh = logging.handlers.RotatingFileHandler('../log/mt8-init.log', maxBytes=(1048576*5), backupCount=7)
fh.setFormatter(logFormat)
log.addHandler(fh)

# In[3]:

maxTrain = None
maxValid = None

with open(sys.argv[2], 'r') as file:
    lines = file.readlines()
trainen0 = [[w.lower() for w in l.strip().split()] for l in lines[:maxTrain]]
with open(sys.argv[1], 'r') as file:
    lines = file.readlines()
trainde0 = [[w.lower() for w in l.strip().split()] for l in lines[:maxTrain]]

with open('../data/en-de/valid.en-de.low.en', 'r') as file:
    lines = file.readlines()
validen0 = [[w.lower() for w in l.strip().split()] for l in lines]
with open('../data/en-de/valid.en-de.low.de', 'r') as file:
    lines = file.readlines()
validde0 = [[w.lower() for w in l.strip().split()] for l in lines]

# In[4]:

# filter out low frequency words
N = 1

def getMapping(data, N=0):
    print("in getMapping, N=%d"%N)
    w2id=defaultdict(lambda:0)
    w2id["<unk>"] = 0
    w2id["<s>"] = 1
    w2id["</s>"] = 2
    id2w=defaultdict(lambda:"<default>")
    id2w[0] = "<unk>"
    id2w[1] = "<s>"
    id2w[2] = "</s>"

    
    unigramCt = Counter()
    for l in data:
        unigramCt.update(l)
    for word, freq in unigramCt.items():
        if freq > N:
            wid = len(w2id)
            id2w[wid] = word
            w2id[word] = wid
    unigramCt = Counter()
    return w2id, id2w
w2iden, id2wen =  getMapping(trainen0, N)
w2idde, id2wde =  getMapping(trainde0, N)
print('de vocab size = %d, en vocab size = %d' % (len(w2idde), len(w2iden)))

# In[ ]:

trainde = [[id2wde[w2idde[w]] for w in sent] for sent in trainde0]
trainen = [[id2wen[w2iden[w]] for w in sent] for sent in trainen0]
validde = [[id2wde[w2idde[w]] for w in sent] for sent in validde0]
validen = [[id2wen[w2iden[w]] for w in sent] for sent in validen0]

# In[8]:


'''
  Alignment: P( E | F) = Σ_θ P( θ, F | E) (Equation 98)
  IBM model 1: P( θ, F | E)
  (1) Initialize θ[i,j] = 1 / (|E| + 1) (i for E and j for F) (Equation 100) 
  (2) Expectation-Maximization (EM)
    [E] C[i,j] =  θ[i,j] / Σ_i θ[i,j] (Equation 110)
    [M] θ[i,j] =  C[i,j] / Σ_j C[i,j] (Equation 107)
  (3) Calculate data likelihood (Equation 106)
'''

class IBM():
    def __init__(self, trainData, validData, dicts, log, saveModelPath='.' ):
        
        self.trainData = trainData
        self.validData = validData
        self.w2ide, self.w2idf,  self.id2we, self.id2wf = dicts  
        self.epsilon = 1.0 / (len(self.w2ide)+1)
        self.theta_ef = defaultdict(lambda: self.epsilon)   # theta_ef[(e,f)] = P(f|e), work with count_ef
        self.theta_fe = defaultdict(lambda: self.epsilon)   # theta_fe[(f,e)] = P(e|f), work with count_fe
        self.init_iter = 0
        self.log = log
        self.saveModelPath = saveModelPath
                
    def train(self, max_iter=5, tic=''):
        for iterCt in range(self.init_iter, self.init_iter+max_iter):
            self.log.info(str(iterCt)+',E')
            # (2) [E] C[i,j] = θ[i,j] / Σ_i θ[i,j] (Equation 110)
            count_ef = {}
            count_e = {}
            count_fe = {}
            count_f = {}
            
            for src, tgt in self.trainData:
                for e in tgt+['<null>']:
                    if e not in count_ef:
                        count_ef[e] = {}
                    if e not in count_e:
                        count_e[e] = 0.0
                for f in src:
                    tmp = 0.0
                    for e in tgt+['<null>']:
                        tmp += self.theta_ef[(e,f)]
                        if f not in count_ef[e]:
                            count_ef[e][f] = 0.0
                    for e in tgt+['<null>']:
                        count_ef[e][f] += self.theta_ef[(e,f)] / tmp
                        count_e[e] += self.theta_ef[(e,f)] / tmp
                        
                for f in src+['<null>']:
                    if f not in count_fe:
                        count_fe[f] = {}
                    if f not in count_f:
                        count_f[f] = 0.0
                for e in tgt:
                    tmp = 0.0
                    for f in src+['<null>']:
                        tmp += self.theta_fe[(f,e)]
                        if e not in count_fe[f]:
                            count_fe[f][e] = 0.0
                    for f in src+['<null>']:
                        count_fe[f][e] += self.theta_fe[(f,e)] / tmp
                        count_f[f] += self.theta_fe[(f,e)] / tmp                        
                        
            
            self.log.info(str(iterCt)+',M')
            # (2) [M] θ[i,j] =  C[i,j] / Σ_j C[i,j] (Equation 107)            
            for e in count_ef: #                
                for f in count_ef[e]:
                    self.theta_ef[(e,f)] = count_ef[e][f] / count_e[e]
                    #print('diff', tmp = count_e[e])
            for f in count_fe: #                
                for e in count_fe[f]:
                    self.theta_fe[(f,e)] = count_fe[f][e] / count_f[f]
                    #print('diff', tmp = count_e[e])        
            self.count_e = count_e
            self.count_f = count_f
            
            savePath = self.saveModelPath+'/mt8-_'+tic+'_'+str(iterCt)+'.pkl'
            self.log.info('saving model file to: %s'%savePath)            
            with open(savePath, 'wb') as f:
                pickle.dump((dict(self.theta_ef), self.count_e, dict(self.theta_fe), self.count_f), f)            
            self.log.info('calculating log likelihood')  
            
            ll = self.logLikelihood_f(self.validData)  
            self.log.info('ll on validation=%.4f', ll)

    def logLikelihood_e(self, data):
        total = 0.0
        for idx, (src, tgt) in enumerate(data):
            ll = np.log(1.0) # - len(src) * np.log(len(tgt)+1)
            for e in tgt:
                tmp = 0
                for f in src+['<null>']: 
                    tmp += self.theta_fe[(f,e)]
                ll += np.log(tmp/(len(src)+1))
            total += ll / len(tgt)
        return total/len(data)            
            

    def logLikelihood_f(self, data):
        total = 0.0
        for idx, (src, tgt) in enumerate(data):
            ll = np.log(1.0) # - len(src) * np.log(len(tgt)+1)
            for f in src:
                tmp = 0
                for e in tgt+['<null>']: 
                    tmp += self.theta_ef[(e, f)]
                ll += np.log(tmp/(len(tgt)+1))
            total += ll / len(src)
        return total/len(data)
    def load(self, path, init_iter):
        with open(path, 'rb') as f:
            thetaef_counte_thetafe_countf = pickle.load(f)
        theta_ef, self.count_e, theta_fe, self.count_f = thetaef_counte_thetafe_countf
        
        self.theta_ef = defaultdict(lambda: self.epsilon)
        for k, v in theta_ef.items():
            self.theta_ef[k]=v
        self.theta_fe = defaultdict(lambda: self.epsilon)
        for k, v in theta_fe.items():
            self.theta_fe[k]=v
        self.init_iter = init_iter
        self.log.info('loading %s completed'%path)
        
    def align1(self, bitext):  # wrong direction. remove
        alignments = []
        for src, tgt in bitext:
            alignment = []
            for i, e in enumerate(tgt):
                # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)
                max_j, max_prob = None, -np.inf
                for j, f in enumerate(src):
                    prob = self.theta_ef[(e,f)] * self.count_e[e]
                    if max_prob < prob:
                        max_j, max_prob = j, prob
                assert max_j is not None
                if max_j < len(src):
                    alignment.append((i, max_j))
            alignments.append(alignment)
        return alignments
    def align_intersection(self, bitext, allowNull=True):
        alignments_f = self.align_f(bitext, False)
        alignments_e = self.align_e(bitext, False)
        return [list(set(f).intersection(e)) for f,e in zip(alignments_f,alignments_e)]
    
    def align(self, bitext, allowNull=True):
        addNull = ['<null>'] if allowNull else []
        alignments = []
        for src, tgt in bitext:
            alignment = []
            for j, f in enumerate(src):
                # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)
                max_i, max_prob = None, -np.inf
                for i, e in enumerate(tgt+addNull):
                    prob = self.theta_ef[(e,f)] * self.theta_fe[(f,e)]
                    if max_prob < prob:
                        max_i, max_prob = i, prob
                assert max_i is not None
                if max_i < len(tgt):
                    alignment.append((max_i, j))
            alignments.append(alignment)
        return alignments
    def align_f(self, bitext, allowNull=True):
        addNull = ['<null>'] if allowNull else []
        alignments = []
        for src, tgt in bitext:
            alignment = []
            for j, f in enumerate(src):
                # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)
                max_i, max_prob = None, -np.inf
                for i, e in enumerate(tgt+addNull):
                    prob = self.theta_ef[(e,f)] * self.count_e[e]
                    if max_prob < prob:
                        max_i, max_prob = i, prob
                assert max_i is not None
                if max_i < len(tgt):
                    alignment.append((max_i, j))
            alignments.append(alignment)
        return alignments
    def align_e(self, bitext, allowNull=True):
        addNull = ['<null>'] if allowNull else []
        alignments = []
        for src, tgt in bitext:
            alignment = []
            for i, e in enumerate(tgt):
                # ARGMAX_j θ[i,j] or other alignment in Section 11.6 (e.g., Intersection, Union, etc)
                max_j, max_prob = None, -np.inf
                for j, f in enumerate(src+addNull):
                    prob = self.theta_fe[(f,e)] * self.count_f[f]
                    if max_prob < prob:
                        max_j, max_prob = j, prob
                assert max_j is not None
                if max_j < len(src):
                    alignment.append((i, max_j))
            alignments.append(alignment)
        return alignments
    def translate_e2f(self, tgt):
        src = []
        for e in tgt:
            f_, prob_ = '<error>', 0.0
            for f in self.w2idf:
                prob = self.theta_ef[(e,f)]
                if  prob> prob_:
                    f_, prob_ = f, prob
            src.append(f_)
        return src



    
tic = str(time.time()).split('.')[0]
print('tic=',tic)

fh.close()
fh = logging.handlers.RotatingFileHandler('../log/mt8-'+tic+'.log', maxBytes=(1048576*5), backupCount=7)
fh.setFormatter(logFormat)
log.addHandler(fh)

ibm = IBM(zip(trainde, trainen),zip(validde, validen), dicts=(w2iden, w2idde,  id2wen, id2wde), log=log ,saveModelPath='../model')
#ibm.load('../model/mt8-_1490142762_5.pkl', init_iter=5)
ibm.train(max_iter=8, tic=tic)

print('training done. tic=',tic)
alignments = ibm.align(zip(trainde, trainen))
with open(sys.argv[3], 'w') as f:
    for alignment in alignments:
        alignment_str = ' '.join([ '%d-%d'%(i_j[0],i_j[1]) for i_j in alignment])
        f.write(alignment_str+'\n')

print('alignment done. tic=',tic)
