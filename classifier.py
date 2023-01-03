import numpy as np
import scipy as sc 
import random as random
import math as math
import networkx as nx
from utils import duree, edge_constr_dynamic, edge_constr
from segment import Segmentation_temporal, Temporal_vectors
import os 
import pickle 
from tqdm import tqdm 
import argparse
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


##chose the day over which to carry the calculation
seed = 1
d1 = 1
d2 = 2
largeur = 300
data_dir='./data/highschool_data'

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

file = open(os.path.join(data_dir,'G_t_d'), 'rb')
G_t_d = pickle.load(file)
file.close()

file = open(os.path.join(data_dir,'T_uv_d'), 'rb')
T_uv_d = pickle.load(file)
file.close()

file = open(os.path.join(data_dir,'E_t_d'), 'rb')
E_t_d = pickle.load(file)
file.close()

file = open(os.path.join(data_dir,'snap_to_window_d'), 'rb')
snap_to_window_d = pickle.load(file)
file.close()

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        default="high",
        choices = ["high","prim"],
        type=str,
    )

    parser.add_argument(
        "-l",
        "--largeur",
        default=300,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--day",
        default=1,
        type=int,
    )    

    args = parser.parse_args()
    largeur = args.largeur
    if args.type == "high":
        data_dir='./data/highschool_data'
    elif args.type == "prim":
        data_dir='./data/primary_school_data'
    else : 
        raise Exception("Not a valid type")

    seed = 1
    d1 = args.day
    d2 = args.day+1

    if args.type == "high" and args.day not in range(1,6):
        raise Exception("Not a valid day for highschool data set")
    elif args.type == "prim" and args.day not in range(1,3):
        raise Exception("Not a valid day for primary school data set")
    else : 
        for d in range(d1,d2):
            
            G_t = G_t_d[d]
            T_uv = T_uv_d[d]
            Temps = sorted(list(T_d[d]))
            E_t= E_t_d[d]
            snap_to_window = snap_to_window_d[d]
            # si trop long, les calculer une fois pour toute et pickle les resultats 
            ####################################################################
            ####################################################################
            ####################################################################

            Vects = Temporal_vectors(T_d[d],day_0[d],largeur)
            Deg_vects = Vects.get_degree_vecs(classe)
            Core_classe_vects = Vects.get_core_class_vecs(classe,1)
            Core_num_vects = Vects.get_core_number_vecs(classe)
            Itrich_classe_vects = Vects.get_Itrich_class_vecs(T_uv,classe,0,1,10,seed)

            Itrich_node_record_temp = Itrich_classe_vects
            I=len(classe)
            J=len(Temps)
            ItRich_mat=np.zeros((I,J))
            degree_mat=np.zeros((I,J))
            core_mat=np.zeros((I,J))


            ordred_classes=list(set(classe.values()))
            rank=dict()
            for i in range(len(classe)):
                rank[sorted(list(classe.keys()))[i]]=i
            grouped_by_rank=dict()
            for c in ordred_classes:
                grouped_by_rank[c]=[rank[node] for node in classe if classe[node]==c]
            grouped_by_rank
            good_rank=[]
            for c in grouped_by_rank:
                good_rank.extend(grouped_by_rank[c])
            ranked_nodes=[]
            for r in good_rank:
                for node in classe:
                    if rank[node]==r:
                        ranked_nodes.append(node)

            for i in range(len(classe)):
                node=ranked_nodes[i]
                ItRich_mat[i]+=np.array([Itrich_node_record_temp[node][snap_to_window[t]]
                            if node in list(set([item for v in E_t[t] for item in v])) else -1 for t in Temps])

                degree_mat[i]+=np.array([Deg_vects[node][snap_to_window[t]]
                            if node in list(set([item for v in E_t[t] for item in v])) else -1 for t in Temps])

                core_mat[i]+=np.array([Core_classe_vects[node][snap_to_window[t]]
                            if node in list(set([item for v in E_t[t] for item in v])) else -1 for t in Temps])



            y = np.array([ordred_classes.index(classe[node]) for node in ranked_nodes])
            Itrich_train, Itrich_test, y_itrich_train, y_itrich_test = train_test_split(ItRich_mat, y, test_size=0.2, random_state=seed)
            degree_train, degree_test, y_degree_train, y_degree_test = train_test_split(degree_mat, y, test_size=0.2, random_state=seed)
            core_train, core_test, y_core_train, y_core_test = train_test_split(core_mat, y, test_size=0.2, random_state=seed)


            #clf =  RandomForestClassifier(max_depth=max_depth,n_estimators = nestim, random_state=seed)
            #clf =  svm.SVC()
            clf = LogisticRegression(random_state=seed, max_iter = 10000)


            clf_itrich = clf 
            clf_itrich.fit(Itrich_train, y_itrich_train)
            y_itrich_res = clf_itrich.predict(Itrich_test)
            itrich_accuracy = accuracy_score(y_itrich_test, y_itrich_res)


            clf_degree = clf
            clf_degree.fit(degree_train, y_degree_train)
            y_degree_res = clf_degree.predict(degree_test)
            degree_accuracy = accuracy_score(y_degree_test, y_degree_res)


            clf_core = clf
            clf_core.fit(core_train, y_core_train)
            y_core_res = clf_itrich.predict(core_test)
            core_accuracy = accuracy_score(y_core_test,y_core_res)

            print("results for day",d)
            print("accuracy for itrich vect :",itrich_accuracy)
            print("accuracy for K-core vect :",core_accuracy)
            print("accuracy for degree vect :",degree_accuracy)

if __name__ == "__main__":
    main()