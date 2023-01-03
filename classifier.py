import numpy as np
import random as random
from segment import Temporal_vectors
import os 
import pickle 
from tqdm import tqdm 
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

seed = 0

def load_pickles(data_dir,file_names):
    loaded_objects = []
    for file_name in file_names:
        file = open(os.path.join(data_dir,file_name), 'rb')
        loaded_objects.append(pickle.load(file))
        file.close()
    return loaded_objects


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

    file_names = ['classe','t0','day_0','T_d','G_t_d','T_uv_d','E_t_d','snap_to_window_d']
    classe, t0, day_0, T_d, G_t_d, T_uv_d, E_t_d, snap_to_window_d = load_pickles(data_dir,file_names)

    d1 = args.day
    d2 = args.day+1

    if args.type == "high" and args.day not in range(1,6):
        raise Exception("Not a valid day for highschool data set")
    elif args.type == "prim" and args.day not in range(1,3):
        raise Exception("Not a valid day for primary school data set")
    else : 
        for d in range(d1,d2):
            print(" ")
            print("Processing "+args.type+" data for day "+str(d))
            print(" ")
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
            core_num_mat=np.zeros((I,J))

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

                core_num_mat[i]+=np.array([Core_num_vects[node][snap_to_window[t]]
                            if node in list(set([item for v in E_t[t] for item in v])) else -1 for t in Temps])
                

            y = np.array([ordred_classes.index(classe[node]) for node in ranked_nodes])
            Itrich_train, Itrich_test, y_itrich_train, y_itrich_test = train_test_split(ItRich_mat, y, test_size=0.2, random_state=seed)
            degree_train, degree_test, y_degree_train, y_degree_test = train_test_split(degree_mat, y, test_size=0.2, random_state=seed)
            core_train, core_test, y_core_train, y_core_test = train_test_split(core_mat, y, test_size=0.2, random_state=seed)
            core_num_train, core_num_test, y_core_num_train, y_core_num_test = train_test_split(core_num_mat, y, test_size=0.2, random_state=seed)

            #clf =  RandomForestClassifier(max_depth=max_depth,n_estimators = nestim, random_state=seed)
            #clf =  svm.SVC()
            clf = LogisticRegression(random_state = seed, max_iter = 10000)


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

            clf_core_num = clf
            clf_core_num.fit(core_num_train, y_core_num_train)
            y_core_num_res = clf_itrich.predict(core_num_test)
            core_num_accuracy = accuracy_score(y_core_num_test,y_core_num_res)

            print("results for day",d)
            print("accuracy for itrich class vect :",itrich_accuracy)
            print("accuracy for K-core class vect :",core_accuracy)
            print("accuracy for degree vect :",degree_accuracy)
            print("accuracy for K-core number vect :",core_num_accuracy)

if __name__ == "__main__":
    main()