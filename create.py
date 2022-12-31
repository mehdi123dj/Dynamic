import pickle 
import os 
dir='./data'

# Create Highschool data 
mon_fichier=open(os.path.join(dir,"highschool_data","High-School_data.txt"), "r")
contenu=mon_fichier.readlines()
mon_fichier.close()

temp_edge=[]
classe=dict()
for l in range(len(contenu)):
    ligne = contenu[l]
    chaine=(ligne.split(" "))
    t=int(chaine[0])
    u=int(chaine[1])
    v=int(chaine[2])
    edge=(t,u,v)
    temp_edge.append(edge)
    classe[u]=chaine[3]
    classe[v]=chaine[4][:-1]

file = open('./data/highschool_data/classe', 'wb')
pickle.dump(classe, file)
file.close()

day_min=54040
day=dict()
i=0
for d in range(1,6):
    j=[]
    while i < len(temp_edge)-1 and temp_edge[i+1][0]-temp_edge[i][0]<day_min:
        j.append(temp_edge[i])
        i+=1
    day[d]=j
    i+=1

t0={i : day[i][0][0] for i in range(1,6)}
file = open('./data/highschool_data/t0', 'wb')
pickle.dump(t0, file)
file.close()

day_0={d:[(day[d][i][0]-t0[d],day[d][i][1],day[d][i][2]) for i in range(len(day[d]))]
    for d in range(1,6)}
file = open('./data/highschool_data/day_0', 'wb')
pickle.dump(day_0, file)
file.close()


T_d={d : set([t for t,u,v in day_0[d]]) for d in range(1,6)}
file = open('./data/highschool_data/T_d', 'wb')
pickle.dump(T_d, file)
file.close()


# Create Primary School data 
mon_fichier=open(os.path.join(dir,"primary_school_data","primaryschool.txt"), "r")
contenu=mon_fichier.readlines()
mon_fichier.close()

temp_edge=[]
classe=dict()
for l in range(len(contenu)):
    ligne = contenu[l]
    chaine=(ligne.split("\t"))
    u=int(chaine[1])
    v=int(chaine[2])
    edge=(int(chaine[0]),u,v,)
    temp_edge.append(edge)
    classe[u]=chaine[3]
    classe[v]=chaine[4][:-1]

file = open('./data/primary_school_data/classe', 'wb')
pickle.dump(classe, file)
file.close()

day_min=54940
day=dict()
i=0
for d in range(1,3):
    j=[]
    while i < len(temp_edge)-1 and temp_edge[i+1][0]-temp_edge[i][0]<day_min:
        j.append(temp_edge[i])
        i+=1
    day[d]=j
    i+=1

t0={i : day[i][0][0] for i in range(1,3)}
file = open('./data/primary_school_data/t0', 'wb')
pickle.dump(t0, file)
file.close()

day_0={d:[(day[d][i][0]-t0[d],day[d][i][1],day[d][i][2]) for i in range(len(day[d]))]
    for d in range(1,3)}

file = open('./data/primary_school_data/day_0', 'wb')
pickle.dump(day_0, file)
file.close()


T_d={d : set([t for t,u,v in day_0[d]]) for d in range(1,3)}
file = open('./data/primary_school_data/T_d', 'wb')
pickle.dump(T_d, file)
file.close()

