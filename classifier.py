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
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from utils import load_pickles, res_to_str, 
seed = 0



def classifier_test(clf,X_train,Y_train,X_test,Y_test):
    clf.fit(X_train, Y_train)
    Y_res = clf.predict(X_test)
    f1 = np.round(f1_score(Y_test, Y_res, average = "macro",zero_division=0),2)
    recall = np.round(recall_score(Y_test, Y_res, average = "macro",zero_division=0),2)
    precision = np.round(precision_score(Y_test, Y_res, average = "macro",zero_division=0),2)
    return (f1, recall, precision)
             
def compute_clf_sats(clf,nclassifier,X,Y,test_size):
    f1_score_l, recall_l, precision_l = [],[],[]
    for i in range(nclassifier):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random.randint(0,10000))
        f1, recall, precision = classifier_test(clf,X_train,y_train,X_test,y_test)
        f1_score_l.append(f1)
        recall_l.append(recall)
        precision_l.append(precision)
    res = {"F1-Score": {"Mean":np.round(np.mean(f1_score_l),2),"Std":np.round(np.std(f1_score_l),2)},
        "Recall": {"Mean":np.round(np.mean(recall_l),2),"Std":np.round(np.std(recall_l),2)},
        "Precision": {"Mean":np.round(np.mean(precision_l),2),"Std":np.round(np.std(precision_l),2)}}
    return res


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
        "-nc",
        "--nclassifier",
        default=25,
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
    test_size = args.test
    nclassifier = args.nclassifier

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
            Itrich_classe_vects = Vects.get_Itrich_class_vecs(T_uv,classe,thresh,wid,Nrep)
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

            X_itrich = ItRich_mat
            X_degree = degree_mat
            X_core = core_mat
            X_core_num = core_num_mat
            y = np.array([ordred_classes.index(classe[node]) for node in ranked_nodes])

            clf1 =  RandomForestClassifier(max_depth=30,n_estimators = 100, random_state=seed)
            clf2 =  svm.SVC(random_state = seed)
            clf3 = LogisticRegression(random_state = seed, max_iter = 10000)
            classifiers = [clf1,clf2,clf3]
            clf_names = ["Random forest"," Support vector"," Logistic Regression"]
            print("------------------------------------------------------")
            for i, clf in enumerate(classifiers):
                itrich_res = compute_clf_sats(clf,nclassifier,ItRich_mat,y,test_size)
                degree_res = compute_clf_sats(clf,nclassifier,degree_mat,y,test_size)
                core_class_res = compute_clf_sats(clf,nclassifier,core_mat,y,test_size)
                core_num_res = compute_clf_sats(clf,nclassifier,core_num_mat,y,test_size)

                print("Classifier :"+clf_names[i]) 
                print("------------------------------------------------------")
                print(12*" "+"F1-Score"+4*" "+"Recall"+10*" "+"Precision")
                print("ItRich cls ",res_to_str(itrich_res))
                print("Degree num ",res_to_str(degree_res) )
                print("K-core num ",res_to_str(core_num_res) )
                print("K-core cls ",res_to_str(core_class_res) )
                print("------------------------------------------------------")
            ####################################################################




if __name__ == "__main__":
    main()