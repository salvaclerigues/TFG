import random
import copy
import time
import numpy as np
import pymoo

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

def randomPopulation(Pop, n, m):
    Population = []
    while len(Population) < Pop:
        i = set(random.sample(range(n), m))
        if i not in Population:
            Population.append(i)
    return Population

def maxsum(sol,M):
    return sum(M[i][j] for i in sol for j in sol) / 2

def maxmin(sol,M):
    return min(M[i][j] for i in sol for j in sol if M[i][j]>0)

def domina(individuo1, individuo2, M):
    # True si in1 domina in2, false si no
    maxmin_individuo1 = maxmin(individuo1, M)
    maxmin_individuo2 = maxmin(individuo2, M)
    maxsum_individuo1 = maxsum(individuo1, M)
    maxsum_individuo2 = maxsum(individuo2, M)

    if maxmin_individuo1 >= maxmin_individuo2 and maxsum_individuo1 >= maxsum_individuo2:
        return True
    else:
        return False

def fast_non_dominated_sort(P, M):
    F = [] # Lista para almacenar los frentes de dominancia
    Q = [] # Lista para añadir Fi para luego append a F
    n = [0] * len(P)  # Diccionario para almacenar la dominancia de cada individuo
    S = [[] for _ in range(len(P))] #Soluciones que i domina
    rank = [0] * len(P) #Nº de frente de Si
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j:
                if domina(P[i], P[j], M):
                    S[i].append(P[j])
                elif domina(P[j], P[i], M):
                    n[i] += 1
        if n[i] == 0:
            rank[i] = 1
            Q.append(P[i])
    F.append(Q)
    i = 0
    while (len(F[i]) > 0):
        Q = []
        for p in F[i]:
            for q in S[P.index(p)]:
                n[P.index(q)] -= 1
                if n[P.index(q)] == 0:
                    rank[P.index(q)] = i + 2
                    Q.append(q)
        i += 1
        F.append(Q)
    F.pop()
    return F,rank

def binary_tournament(P1, P2, P, rank):
    if rank[P.index(P1)] >= rank[P.index(P2)]:
        return P1
    else:
        return P2

def crossover(P1, P2):
    FixG = P1 & P2
    O1 = list(FixG)
    O2 = list(FixG)
    FreeG = (P1 | P2) - FixG
    while FreeG:
        g1 = random.choice(list(FreeG))
        O1.append(g1)
        FreeG.remove(g1)
        g2 = random.choice(list(FreeG))
        O2.append(g2)
        FreeG.remove(g2)
    return set(O1), set(O2)

def mutation(S, n, MtPr):
    N=set(range(n))
    NS=N-S
    for _ in range(len(S)):
        i=random.uniform(0,1)
        if i<MtPr:
            r = random.choice(list(S))
            a = random.choice(list(NS))
            S.add(a)
            S.remove(r)
            NS.add(r)
            NS.remove(a)
    return S

def executeCrossover(P, CxPr, rank, n, MtPr):
    P_Cx = []
    while len(P_Cx) < len(P):
        P11, P12, P21, P22 = random.sample(P, 4)
        P1 = binary_tournament(P11,P12,P, rank)
        P2 = binary_tournament(P21,P22,P, rank)

        i = random.uniform(0.5,1)
        if i < CxPr:
            O1, O2 = crossover(P1, P2)
        else:
            O1=copy.copy(P1)
            O2=copy.copy(P2)
        O1=mutation(O1, n, MtPr)
        O2=mutation(O2, n, MtPr)
        P_Cx.append(O1)
        P_Cx.append(O2)
    return P_Cx

def unirunicos(A, B):
    conjuntos_unicos = []
    Todos=A+B
    for conjunto in Todos:
        if conjunto not in conjuntos_unicos:
            conjuntos_unicos.append(conjunto)
    return conjuntos_unicos
def make_new_pop(Pt, CxPr, MtPr, n, M):
    F, rank = fast_non_dominated_sort(Pt, M)
    Pt = computeCrowdingDistance(Pt,M)
    new_pop = executeCrossover(Pt, CxPr, rank, n, MtPr)
    return new_pop

def computeCrowdingDistance(P, M):
    Val=[]
    for individual in P:
        Val.append([maxsum(individual, M), maxmin(individual, M)])
    l = len(P)
    distancias = [0.0] * l
    for obj in range(2):
        Puntos = [element[obj] for element in Val]
        maxval = max(Puntos)
        minval = min(Puntos)
        sorted_indices = sorted(range(l), key=lambda x: Puntos[x], reverse=True)
        distancias[sorted_indices[0]] = float('inf')
        distancias[sorted_indices[-1]] = float('inf')
        for i in range(1,l-1):
            if maxval-minval != 0:
                distancias[sorted_indices[i]]+=((Puntos[sorted_indices[i - 1]])- Puntos[sorted_indices[i + 1]])/(maxval-minval)
            elif maxval - minval == 0:
                distancias[sorted_indices[i]]+=((Puntos[sorted_indices[i - 1]])- Puntos[sorted_indices[i + 1]])
    sorted_P = sorted(P, key=lambda x: distancias[P.index(x)], reverse=True)
    return sorted_P

def NSGA_II(M, m, Pop, duracion, CxPr, MtPr):
    ND=[]
    n = len(M)
    P = randomPopulation(Pop, n, m)
    F, rank = fast_non_dominated_sort(P, M)
    Q = executeCrossover(P, CxPr, rank, n, MtPr)
    Pt = P
    gener=0
    #for i in range(Gen):
        #print(i)
    while True:
        gener+=1
        print(gener)
        R = unirunicos(Pt,Q)
        F, rank = fast_non_dominated_sort(R, M)
        Pt = []
        i = 0
        while (len(Pt) + len(F[i])) < Pop:
            Pt.extend(F[i])
            i+=1
        F_sorted = computeCrowdingDistance(F[i], M)
        i=0
        while len(Pt) < Pop:
            Pt.append(F_sorted[i])
            i+=1
        Q = make_new_pop(Pt, CxPr, MtPr, n, M)
        tiempo_transcurrido = time.time() - start_time
        if tiempo_transcurrido >= duracion:
            break
    F,rank = fast_non_dominated_sort(Pt,M)
    for element in F[0]:
        ND.append(element)
    return ND

Pop = 20
perc=0.4
Gen = 10
CxPr = 0.26
MtPr = 0.08
duracion=40


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
for j in range(1,146):
    print('Instància ' + str(j))
    [M, m, n] = read_matrix_from_file(
        '/Users/salvaclerigues/PycharmProjects/TFG/NSGA-II/Todos gkd/a_' + str(j) + '.txt')
    start_time = time.time()
    q = NSGA_II(M, m, Pop, duracion, CxPr, MtPr)
    #p = [list(s) for s in q]
    p=q
    end_time = time.time()
    final_time = end_time - start_time

    k = len(p)
    valors = []
    for i in range(k):
        valors.append((maxsum(p[i],M), maxmin(p[i],M)))
    valors = np.array(valors)
    print(valors)

    n_sol=k
    hv=calculate_hypervolume(valors)
    resultados.append([j, final_time, n_sol, hv])
df = pd.DataFrame(resultados, columns=['Instancia', 'Tiempo', 'Cardinalidad', 'Hipervolumen'])
print(df)
df.to_excel('NSGAtotst40pop20.xlsx', index=False)