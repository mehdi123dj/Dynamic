import numpy as np
import scipy as sc 
import math as math
import networkx as nx
import operator 
from utils import strength, rand_top_wight

class Segmentation_temporal: 
    
    def __init__(self,W):
        self.W = W.copy()
        self.dense = dict()
        self.tot = dict()
        self.randtot =dict()
        self.std_dev = dict()
        self.liss = dict()
        self.Qality = dict()
        self.Q_std_dev =dict()
    
    def process(self,seed,wid,Nrep):
        sortie=dict()
        u=0
        compare_qual=[]
        minstrength=math.inf
        S=self.W.size(weight='weight')
        if S==0:
            self.dense = None
            self.tot = None
            self.randtot = None
            self.std_dev = None
            self.liss = None
            self.Qality = None
            self.Q_std_dev = None
        else:   
            while self.W.size(weight='weight')>0.:
                weights=[self.W[u][v]['weight'] for u,v in self.W.edges()]
                S=self.W.size(weight='weight')
                strengthlist={node : strength(self.W,node) for node in self.W.nodes()}
                trille=sorted(strengthlist.items(), key=operator.itemgetter(1),reverse=True) 
                res=[trille[i][0] for i in range(len(trille))]   
                tot=[]
                A=[]
                if S==0:
                    break 
                for node in (res):
                    A.append(node)
                    tot.append((self.W.subgraph(A).size(weight='weight'))*1./S)
                mean=[]
                for i in range(Nrep):
                    degree_sequence=[]
                    for node in self.W.nodes():
                        degree_sequence.append(self.W.degree(node))
                    randW=nx.configuration_model(degree_sequence, create_using=self.W.copy())
                    randW=rand_top_wight(randW,weights)
                    RS=randW.size(weight='weight')
                    randstrengthlist={node : strength(randW,node) for node in randW.nodes()}      
                    randtrille=sorted(randstrengthlist.items(), key=operator.itemgetter(1),reverse=True)
                    randres=[randtrille[i][0] for i in range(len(randtrille))]
                    randA=[]
                    randtot_i=[]
                    for node in (randres):
                        randA.append(node)
                        if RS==0: 
                            randtot_i.append(0)
                        else:
                            randtot_i.append((randW.subgraph(randA).size(weight='weight'))*1./RS)
                    mean.append(randtot_i)
                Q_list=[sum([(tot[j]-mean[i][j])*1./len(tot) for j in range(len(tot))]) for i in range(Nrep)]
                Q_mean=sum(Q_list)*1./len(Q_list)
                Q_std_dev=sum([(q-Q_mean)**2 for q in Q_list])*1./(len(Q_list)-1)
                randtot_mean=np.zeros(len(tot))
                for i in range(Nrep):
                    act=mean[i]    
                    for j in range(len(act)):
                        randtot_mean[j]+=act[j]*1./Nrep
                randtot=list(randtot_mean)
        
                std_dev=[]
                for i in range(len(self.W)):
                    std_i=0
                    for iteration in range(Nrep):
                        std_i+=(mean[iteration][i])**2
                    std_dev.append(math.sqrt(round(abs((1./Nrep)*(std_i)-(randtot[i])**2),10)))        
                diff=[tot[i]-randtot[i] for i in range(len(tot))]
                liss=diff
                if max(diff)>0.0:

                    minode=np.argmax(liss)+wid-1
                    ND=[]                      
                    for i in range(minode+1,len(res)):
                        ND.append(res[i])

                    DW=self.W.subgraph(ND)
                    Dense=[x for x in self.W if x not in ND]
                    mini=min([strength(self.W,node) for node in Dense])
                    for node in ND:
                        if strength(self.W,node)==mini:
                            Dense.append(node)
                    Quality=Q_mean
                    self.dense[u] = self.W.subgraph(Dense)
                    self.tot[u] = tot
                    self.randtot[u] = randtot
                    self.std_dev[u] = std_dev
                    self.liss[u] = liss
                    self.Qality[u] = Quality
                    self.Q_std_dev[u] = math.sqrt(Q_std_dev)
                    if len(Dense)>0:
                        self.W=DW
                        u+=1
                    else:
                        break
                else:
                    break

