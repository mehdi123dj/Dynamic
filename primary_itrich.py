import numpy as np
import scipy as sc 
import random as random
import math as math
import networkx as nx
from utils import duree, edge_constr_dynamic, edge_constr
from segment import Segmentation_temporal
import os 
import pickle 
from tqdm import tqdm 
import argparse

data_dir='./data/primary_school_data'
res_dir='./results'



def main(): 

    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-d",
        "--day",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-wd",
        "--width",
        default=300,
        type=int,
    )

    parser.add_argument(
        "-sm",
        "--smoothing",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-n",
        "--number-null",
        default=25,
        type=int,
    )

    parser.add_argument(
        "-s",
        "--seed",
        default=random.randint(0,1000),
        type=int,
    )
    args = parser.parse_args()


    file = open(os.path.join(data_dir,'classe'), 'rb')
    classe = pickle.load(file)
    file.close()

    file = open(os.path.join(data_dir,'t0'), 'rb')
    t0 = pickle.load(file)
    file.close()

    file = open(os.path.join(data_dir,'day_0'), 'rb')
    day_0 = pickle.load(file)
    file.close()

    file = open(os.path.join(data_dir,'T_d'), 'rb')
    T_d = pickle.load(file)
    file.close()

    ##chose the day over which to carry the calculation
    d=args.day
    ######chose the window size######
    largeur=args.width
    #chose weather there is overlap or not if slide = largeure, it means there is no overlap between windows
    slide=largeur
    N_null = args.number_null    ##The number of null models to average the higher the more precise and slower
    smoothing=args.smoothing     ##No smoothing when value is 1
    seed= args.seed
    


    #########            Code execution          #######################
    G_t=dict()
    t_max=max(T_d[d])
    t1=0
    t2=largeur
    while t2<=t_max:
        G=nx.Graph()
        edge_list=[(edge[1],edge[2]) for edge in day_0[d] if t2>edge[0]>=t1]
        G.add_edges_from(edge_list)
        G_t[t1]=G
        t1+=slide
        t2+=slide

    G=nx.Graph()
    edge_list=[(edge[1],edge[2]) for edge in day_0[d] if t_max>=edge[0]>=t1]
    G.add_edges_from(edge_list)
    G_t[t1]=G

    ####################################################################
    ####################################################################

    T_uv={(u,v):set() for u in classe for v in classe} 

    for t,u,v in day_0[d]:
        T_uv[(u,v)].add(t)
        T_uv[(v,u)].add(t)

    ####################################################################
    ####################################################################

    for t in tqdm(G_t):
        t2=t+largeur
        #W=edge_constr_dynamic(T_uv,t1,t2)
        W=edge_constr(G_t[t])
        f = open(os.path.join(res_dir,"Input_W_primary_school","W_day_%d_t0_%d.txt"%(d,t)), "w")
        f.write("day %d \t t_0= %d \n"%(d,t))
        for e in W.edges():
            u=e[0]
            v=e[1]
            f.write(str(u)+'\t'+str(v)+'\t'+str(W.get_edge_data(u,v)['weight'])+'\n')
        f.close()

        segmente_temp=Segmentation_temporal(W)
        segmente_temp.process(seed,smoothing,N_null)
        
        f = open(os.path.join(res_dir,"RC_primary_school","segmentation_day_%d_t0_%d.txt"%(d,t)), "w")
        f.write('t %d \n' %(t))
        if segmente_temp.dense != None:
            for i in segmente_temp.dense:
                f.write('RC ')
                for node in list(segmente_temp.dense[i].nodes()):
                    f.write('%d '%(node))
                f.write('\n')
                f.write('Q %f %f \n'%(segmente_temp.Qality[i],segmente_temp.Q_std_dev[i]))

        else :
            f.write('RC \n')
            f.write('Q \n')
        f.close()


if __name__ == "__main__":
    main()