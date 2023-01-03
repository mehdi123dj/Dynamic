import pickle 
import os 
import argparse
import networkx as nx
dir='./data'


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

    args = parser.parse_args()
    largeur = args.largeur

    if args.type == "high":
        d1 = 1
        d2 = 6 
        folder_name = "highschool_data"
        file_name = "High-School_data.txt"
        day_min=54040
        split = " "
    elif args.type == "prim":
        d1 = 1
        d2 = 3
        folder_name = "primary_school_data"
        file_name = "primaryschool.txt"
        day_min=54940
        split = "\t"
    else : 
        raise Exception("Not a valid type")




    # Create Highschool data 
    mon_fichier=open(os.path.join(dir,folder_name,file_name), "r")
    contenu=mon_fichier.readlines()
    mon_fichier.close()


    classe=dict()
    temp_edge=[]
    for l in range(len(contenu)):
        ligne = contenu[l]

        chaine=(ligne.split(split))
        t=int(chaine[0])
        u=int(chaine[1])
        v=int(chaine[2])
        edge=(t,u,v)
        temp_edge.append(edge)
        classe[u]=chaine[3]
        classe[v]=chaine[4][:-1]


    
    file = open(os.path.join(dir,folder_name,"classe"), 'wb')
    pickle.dump(classe, file)
    file.close()

    day=dict()
    i=0
    for d in range(d1,d2):
        j=[]
        while i < len(temp_edge)-1 and temp_edge[i+1][0]-temp_edge[i][0]<day_min:
            j.append(temp_edge[i])
            i+=1
        day[d]=j
        i+=1

    t0={i : day[i][0][0] for i in range(d1,d2)}
    file = open(os.path.join(dir,folder_name,"t0"), 'wb')
    pickle.dump(t0, file)
    file.close()

    day_0={d:[(day[d][i][0]-t0[d],day[d][i][1],day[d][i][2]) for i in range(len(day[d]))]
        for d in range(d1,d2)}
    file = open(os.path.join(dir,folder_name,"day_0"), 'wb')
    pickle.dump(day_0, file)
    file.close()


    T_d={d : set([t for t,u,v in day_0[d]]) for d in range(d1,d2)}
    file = open(os.path.join(dir,folder_name,"T_d"), 'wb')
    pickle.dump(T_d, file)
    file.close()

    G_t_d=dict()
    slide = largeur 
    for d in range(d1,d2):
        G_t_d[d] = dict()
        #########            Code execution          #######################
        t_max=max(T_d[d])
        t1=0
        t2=largeur
        while t2<=t_max:
            G=nx.Graph()
            edge_list=[(edge[1],edge[2]) for edge in day_0[d] if t2>edge[0]>=t1]
            G.add_edges_from(edge_list)
            G_t_d[d][t1]=G
            t1+=slide
            t2+=slide

        G=nx.Graph()
        edge_list=[(edge[1],edge[2]) for edge in day_0[d] if t_max>=edge[0]>=t1]
        G.add_edges_from(edge_list)
        G_t_d[d][t1]=G

    file = open(os.path.join(dir,folder_name,"G_t_d"), 'wb')
    pickle.dump(G_t_d, file)
    file.close()

    T_uv_d=dict()
    for d in range(d1,d2):
        T_uv_d[d]={(u,v):set() for u in classe for v in classe} 

        for t,u,v in day_0[d]:
            T_uv_d[d][(u,v)].add(t)
            T_uv_d[d][(v,u)].add(t)

    file = open(os.path.join(dir,folder_name,"T_uv_d"), 'wb')
    pickle.dump(T_uv_d, file)
    file.close()

    E_t_d=dict()
    for d in range(d1,d2):
        E_t_d[d] = dict()
        for t in sorted(list(T_d[d])):
            E_t_d[d][t]=[]
            for e in T_uv_d[d]:
                if t in T_uv_d[d][e]:
                    E_t_d[d][t].append((min(e),max(e)))
            E_t_d[d][t]=set(E_t_d[d][t])  

    file = open(os.path.join(dir,folder_name,"E_t_d"), 'wb')
    pickle.dump(E_t_d, file)
    file.close()


    snap_to_window_d=dict()
    for d in range(d1,d2):
        snap_to_window_d[d] = dict()
        for t in sorted(list(T_d[d])): 
            for tt in G_t_d[d]:
                if tt+largeur>t>=tt:
                    snap_to_window_d[d][t]=tt

    file = open(os.path.join(dir,folder_name,"snap_to_window_d"), 'wb')
    pickle.dump(snap_to_window_d, file)
    file.close()


if __name__ == "__main__":
    main()