import numpy as np
import scipy as sc 
import math as math
import networkx as nx
import operator 
from utils import strength, rand_top_wight
from tqdm import tqdm
from utils import edge_constr_dynamic

class Segmentation_temporal: 
    
    def __init__(self,W):
        self.W_original = W.copy()
        self.W = W
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

    def assemble(self,thresh):
        Dense = []
        ND = []
        if self.dense == None: 
            self.D = []
            self.ND = list(self.W_original.nodes())
        else : 
            for i in self.dense.keys():
                if self.Qality[i]> thresh : 
                    Dense += list(self.dense[i].nodes())
            self.D = Dense
            self.ND = list(set(self.W_original.nodes())-set(Dense))
    


class Temporal_vectors:

    def __init__(self,T_d,day_0,largeur):
        self.slide = largeur
        G_t=dict()
        t_max=max(T_d)
        t1=0
        t2=largeur
        while t2<=t_max:
            G=nx.Graph()
            edge_list=[(edge[1],edge[2]) for edge in day_0 if t2>edge[0]>=t1]
            G.add_edges_from(edge_list)
            G_t[t1]=G
            t1+=largeur
            t2+=largeur

        G=nx.Graph()
        edge_list=[(edge[1],edge[2]) for edge in day_0 if t_max>=edge[0]>=t1]
        G.add_edges_from(edge_list)
        G_t[t1]=G
        self.G_t = G_t

    def get_Itrich_class_vecs(self,T_uv,classe,thresh,wid,Nrep,seed):
        
        RC_t=dict()
        D_t=dict()
        ND_t=dict()
        W_t=dict()
        thresh=0.
        print("Running Itrich")
        for t in tqdm(self.G_t):
            t2=t+self.slide
            W=edge_constr_dynamic(T_uv,t,t2)
            segmente_temp=Segmentation_temporal(W)
            segmente_temp.process(seed,wid,Nrep)
            segmente_temp.assemble(thresh)

            W_t[t] = W
            D_t[t] = segmente_temp.D
            ND_t[t] = segmente_temp.ND
            if segmente_temp.dense == None :  
                RC_t[t] = dict()
            else : 
                RC_t[t] = {i : list(segmente_temp.dense[i].nodes()) for i in segmente_temp.dense}
        
        Itrich_node_record_temp=dict()
        for node in classe:
            Itrich_node_record_temp[node]=dict()
            for t in self.G_t:
                if node in D_t[t]:
                    Itrich_node_record_temp[node][t]=1
                elif node in ND_t[t]:
                    Itrich_node_record_temp[node][t]=0
                else : 
                    Itrich_node_record_temp[node][t]=-1      
        return Itrich_node_record_temp

    def get_degree_vecs(self,classe):

        Degree_node_record_temp = dict()
        print("Running Degree calculation")
        for node in tqdm(classe):
            Degree_node_record_temp[node] = dict()
            for t in self.G_t:
                G = self.G_t[t]
                if node in G:
                    Degree_node_record_temp[node][t] = G.degree(node)
                else : 
                    Degree_node_record_temp[node][t] = 0
        return Degree_node_record_temp
  
    def get_core_class_vecs(self,classe,k_thresh):
        Core_node_record_temp = dict()
        print("Running K-core class calculation")
        for node in tqdm(classe):
            Core_node_record_temp[node] = dict()
            for t in self.G_t:
                G = self.G_t[t]
                core_num =  nx.core_number(G)
                if node in G:
                    if core_num[node] > k_thresh:
                        Core_node_record_temp[node][t] = 1
                    else : 
                        Core_node_record_temp[node][t] = 0
                else : 
                    Core_node_record_temp[node][t] = -1
        return Core_node_record_temp

    def get_core_number_vecs(self,classe):
        Core_node_number_temp = dict()
        print("Running K-core number calculation")
        for node in tqdm(classe):
            Core_node_number_temp[node] = dict()
            for t in self.G_t:
                G = self.G_t[t]
                core_num =  nx.core_number(G)
                if node in G:
                        Core_node_number_temp[node][t] = core_num[node]
                else : 
                    Core_node_number_temp[node][t] = 0
        return Core_node_number_temp

    def get_count(self,Itrich_node_record_temp,snap_to_window,E_t):
        D_count=dict()
        ND_count=dict()
        presents = dict()
        for t in E_t:
            presents[t] = list(set([u for u,v in E_t[t]]+[v for u,v in E_t[t]]))
        for node in Itrich_classe_vects.keys():
            D_count[node] = [Itrich_node_record_temp[node][snap_to_window[t]] if node in presents[t] else -1 for t in E_t].count(1)
            ND_count[node] =[Itrich_node_record_temp[node][snap_to_window[t]] if node in presents[t] else -1 for t in E_t].count(0)
        return {"Dense": D_count, "ND" : ND_count }
