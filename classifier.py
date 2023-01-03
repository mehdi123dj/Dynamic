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
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

seed = 0

def load_pickles(data_dir,file_names):
    loaded_objects = []
    for file_name in file_names:
        file = open(os.path.join(data_dir,file_name), 'rb')
        loaded_objects.append(pickle.load(file))
        file.close()
    return loaded_objects


def classifier_test(clf,X_train,Y_train,X_test,Y_test):
    clf.fit(X_train, Y_train)
    Y_res = clf.predict(X_test)
    accuracy = np.round(accuracy_score(Y_test, Y_res),2)
    recall = np.round(recall_score(Y_test, Y_res, average = "macro"),2)
    precision = np.round(precision_score(Y_test, Y_res, average = "macro"),2)
    return (accuracy, recall, precision)
             

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
    parser.add_argument(
        "-n",
        "--nrep",
        default=5,
        type=int,
    )    
    parser.add_argument(
        "-th",
        "--threshold",
        default=0.,
        type=float,
    )    

    parser.add_argument(
        "-w",
        "--width",
        default=1,
        type=int,
    )    

    parser.add_argument(
        "-ts",
        "--test",
        default=.2,
        type=float,
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
    Nrep = args.nrep
    thresh = args.threshold
    wid = args.width
    test_size = args.test_size

    if args.type == "high" and args.day not in range(1,6):
        raise Exception("Not a valid day for highschool data set")
    elif args.type == "prim" and args.day not in range(1,3):
        raise Exception("Not a valid day for primary school data set")
    else : 
        for d in range(d1,d2):
            ####################################################################
            print(" ")
            print("Processing "+args.type+" school data for day "+str(d))
            print(" ")
            G_t = G_t_d[d]
            T_uv = T_uv_d[d]
            Temps = sorted(list(T_d[d]))
            E_t= E_t_d[d]
            snap_to_window = snap_to_window_d[d]
            ####################################################################
            Vects = Temporal_vectors(T_d[d],day_0[d],largeur)
            Deg_vects = Vects.get_degree_vecs(classe)
            Core_classe_vects = Vects.get_core_class_vecs(classe,1)
            Core_num_vects = Vects.get_core_number_vecs(classe)
            Itrich_classe_vects = Vects.get_Itrich_class_vecs(T_uv,classe,thresh,wid,Nrep,seed)
            ####################################################################
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
            ####################################################################
            y = np.array([ordred_classes.index(classe[node]) for node in ranked_nodes])
            Itrich_train, Itrich_test, y_itrich_train, y_itrich_test = train_test_split(ItRich_mat, y, test_size=test_size, random_state=seed)
            degree_train, degree_test, y_degree_train, y_degree_test = train_test_split(degree_mat, y, test_size=test_size, random_state=seed)
            core_train, core_test, y_core_train, y_core_test = train_test_split(core_mat, y, test_size=test_size, random_state=seed)
            core_num_train, core_num_test, y_core_num_train, y_core_num_test = train_test_split(core_num_mat, y, test_size=test_size, random_state=seed)

            #clf =  RandomForestClassifier(max_depth=max_depth,n_estimators = nestim, random_state=seed)
            #clf =  svm.SVC()

            clf = LogisticRegression(random_state = seed, max_iter = 10000)
            itrich_accuracy, itrich_recall, itrich_precision = classifier_test(clf,Itrich_train,y_itrich_train,Itrich_test,y_itrich_test)
            degree_accuracy, degree_recall, degree_precision = classifier_test(clf,degree_train,y_degree_train,degree_test,y_degree_test)
            core_class_accuracy, core_class_recall, core_class_precision = classifier_test(clf,core_train,y_core_train,core_test,y_core_test)
            core_num_accuracy, core_num_recall, core_num_precision = classifier_test(clf,core_num_train,y_core_num_train,core_num_test,y_core_num_test)

            print(" ")
            print("Accuracy/Recall/Precision for day",d)
            print(" " )
            print("ItRich cls ",itrich_accuracy,itrich_recall,itrich_precision)
            print("Degree num ",degree_accuracy, degree_recall, degree_precision )
            print("K-core num ",core_num_accuracy, core_num_recall, core_num_precision )
            print("K-core cls ",core_class_accuracy, core_class_recall, core_class_precision )
            ####################################################################




if __name__ == "__main__":
    main()