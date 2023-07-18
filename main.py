import random
import time
import numpy as np

def read_matrix_from_file(filename):
    with open(filename, 'r') as archivo:
        # Leer las líneas del archivo
        lineas = archivo.readlines()

        # Obtener el tamaño de la matriz y el valor m
        n, m = map(int, lineas[0].split())
        # Crear una matriz vacía
        matriz = [[0.0] * n for _ in range(n)]

        # Leer los valores de la mitad superior de la matriz
        for linea in lineas[1:]:
            i, j, valor = map(float, linea.split())
            i = int(i)
            j = int(j)
            matriz[i][j] = valor

    # Convertir la mitad superior en matriz simétrica
    for i in range(n):
        for j in range(i + 1, n):
            matriz[j][i] = matriz[i][j]
    return matriz, m ,n

def MaxSum(sol,M):
    return sum(M[i][j] for i in sol for j in sol) / 2

def MaxMin(sol,M):
    return min(M[i][j] for i in sol for j in sol if M[i][j]>0)

def f_obj(sol,M, obj):
    if obj==0:
        return MaxMin(sol,M)
    if obj==1:
        return MaxSum(sol,M)

def min_dist(sol,S,M):
    dist = [min(M[i][j] for j in sol if i != j) for i in S]
    return dist

def actualisecontrib(inn, sol, contrib, obj, N, M):
    if obj == 1:
        contrib = [contrib[i] + M[i][inn] for i in N]
    elif obj == 0:
        contrib = [min(contrib[i], M[i][inn]) for i in N]
    contrib[inn] = 0
    contrib = [0 if i in sol else contrib[i] for i in N]
    return contrib

def Construction (M,m,alpha,obj):
#Iniciem amb les primeres dades
    n=len(M) #longitud del vector insol, m al problema
    sol=set()
    N=list(range(n)) #Tots els elements possibles (del 0 al n-1) en un vector
    CL=set(range(n))

#PRIMER ELEMENT ALEATORI
    i=random.choice(N)
    sol.add(i) #Afegim el element a la solucio
    CL.remove(i)

#INICIALITZAR LES CONTRIBUCIONS
    contrib = [M[j][i] for j in N]

#BUCLE PRINCIPAL
    while len(sol) < m:
        gmin = min(numero for numero in contrib if numero > 0)
        gmax = max(contrib)
        RCL = {elemento for elemento in CL if contrib[elemento] >= gmin + alpha * (gmax - gmin)}
        inn=random.choice(list(RCL))
        CL.remove(inn)
        contrib=actualisecontrib(inn,sol,contrib,obj,N, M)
        sol.add(inn)
    return sol

def LocalSearch(M, sol, obj):
    N = set(range(len(M)))
    S=sol

    while True:
        dist_S = min_dist(sol, S, M)
        dist_NS = min_dist(sol, N-S, M)
        dist_S1 = [[j, d] for j, d in zip(sol, dist_S)]
        dist_NS1 = [[j, d] for j, d in zip(N-S, dist_NS)]

        # Ordenar los elementos de N-S por distancia descendente
        sorted_NS = sorted(dist_NS1, key=lambda x: x[1], reverse=True)

        # Ordenar los elementos de S por distancia ascendente
        sorted_S = sorted(dist_S1, key=lambda x: x[1])

        new_sol=sol
        new_sol.add(sorted_NS[0][0])
        new_sol.remove(sorted_S[0][0])
        if f_obj(sol,M,obj)<f_obj(new_sol,M,obj):
            sol=new_sol
        else:
            break
    return sol



def GRASP(M,m,alpha,n_iter):
    start_time=time.time()
    solucions_ND=[]
    obj=0
    #for _ in range(n_iter):
    while True:
        sol=Construction(M,m,alpha,obj)
        sol=LocalSearch(M,sol,obj)
        sum_sol=MaxSum(sol,M)
        min_sol=MaxMin(sol,M)
        dominada=False
        for s in solucions_ND:
            if sum_sol<=s[1] and min_sol<=s[2]:
                dominada=True
                break
        if not dominada:
            solucions_ND = [s for s in solucions_ND if (sum_sol<s[1] or min_sol<s[2])]
            solucions_ND.append([sol, sum_sol, min_sol])
        if obj == 1:
            obj = 0
        elif obj == 0:
            obj = 1
        tiempo_transcurrido = time.time() - start_time
        if tiempo_transcurrido >= duracion:
            break
    return solucions_ND

alpha=0.75
n_iter=10000
duracion=100


def calculate_hypervolume(points):
    # Agregar el punto de referencia (0, 0) al conjunto de puntos
    reference_point = np.array([0, 0])
    points = np.vstack((points, reference_point))

    # Ordenar los puntos según el primer objetivo (en orden descendente)
    points = points[np.argsort(-points[:, 0])]

    hypervolume = 0.0
    previous_point = None

    # Calcular el hypervolume
    for point in points:
        if previous_point is not None:
            hypervolume += (previous_point[0] - point[0]) * (previous_point[1] - reference_point[1])
        previous_point = point

    return hypervolume



import pandas as pd

resultados=[]
for j in range(1, 146):
    print('Instancia ' + str(j))
    [M, m, n] = read_matrix_from_file(
        '/Users/salvaclerigues/PycharmProjects/TFG/GRASP/Todos gkd/a_' + str(j) + '.txt')
    start_time = time.time()
    p = GRASP(M, m, alpha, n_iter)
    end_time = time.time()
    final_time = end_time - start_time

    k = len(p)
    valors = []
    for i in range(k):
        p[i][0] = {elemento + 1 for elemento in p[i][0]}
        valors.append((p[i][1], p[i][2]))
    valors = np.array(valors)
    print(valors)

    n_sol=k
    hv=calculate_hypervolume(valors)
    resultados.append([j, final_time, n_sol, hv])
df = pd.DataFrame(resultados, columns=['Instancia', 'Tiempo', 'Cardinalidad', 'Hipervolumen'])
print(df)
df.to_excel('GRASPtotst100.xlsx', index=False)