import matplotlib.pyplot as plt
import math
import random
import time
import itertools
import urllib
import csv
import seaborn as sns;
import numpy as np
import copy
import statistics

#CALCULANDO DISTANCIA
distancia = math.dist

def tamanho_rota(rota):
  "A distância total percorrida entre dois pares consecutivos em uma rota"
  return sum(distancia(rota[i-1], rota[i]) for i in range(len(rota)))

#GERANDO UMA CLASSE PARA O PROBLEMA
class PRV:
    def __init__(self):
      self.clientes = []
      self.R = 0
      self.rotas = []

    def custo(self):
      return sum(tamanho_rota(x) for x in self.rotas)

    def __repr__(self):
      return f"Clientes = {len(self.clientes)-1}\tVeiculos = {self.R}\tCusto = {self.custo()}"
    def __str__(self):
      return f"Clientes = {len(self.clientes)-1}\tVeiculos = {self.R}\tCusto = {self.custo()}"

#GERANDO PROBLEMAS ALEATORIOS
def gerar_prv (n, r, largura=900, altura=900, rseed=233):
  "Faz um conjunto com n clientes, com coordenadas aleatórias no máximo (largura x alutura)"
  prv = PRV()
  random.seed(rseed)
  prv.R = r
  prv.clientes = [ [random.randrange(largura), random.randrange(altura)] for c in range(n+1) ]
  prv.clientes[0] = [largura//2, altura//2]
  return prv

#PLOTANDO ROTA
def plot_rota(rota, cor):
  x = []
  y = []
  for i in range(len(rota)):
    x.append(rota[i][0])
    y.append(rota[i][1])

  for i in range(len(rota)):
    plt.plot([x[i], x[i-1]], [y[i], y[i-1]], color=cor, linewidth=2)

  plt.scatter(x, y, color=cor, linewidth=3)

def plot_prv(prv):

  print(prv)

  palette = itertools.cycle(sns.color_palette("colorblind", prv.R))
  for rota in prv.rotas:
    plot_rota(rota, next(palette))

  plt.plot(prv.clientes[0][0], prv.clientes[0][1],
           marker="D", markersize=10, markeredgecolor="black", markerfacecolor="grey")
  plt.axis("scaled")

  plt.show()

#ALGORITMO CONSTRUTIVO
def kmeans(prv):
  max_iter = 100

  clientes = prv.clientes[1:len(prv.clientes)]

  # Inicializar os centroides aleatoriamente
  centroides = random.sample(clientes, prv.R)

  for i in range(max_iter):
    # Designar cada cliente para um veiculo
    prv.rotas = [[] for _ in range(prv.R)]
    for x in clientes:
        distances = [distancia(x,c) for c in centroides]
        centroide_mais_proximo = np.argmin(distances)
        prv.rotas[centroide_mais_proximo].append(x)

    # Atualizar centroides
    prox_centroides = []
    for c in prv.rotas:
        if len(c) > 0:
            novo_centro = np.mean(c, axis=0)
            prox_centroides.append(novo_centro)
        else:
            prox_centroides.append(centroides[np.random.choice(prv.R)])

    # Verificar convergência
    if np.allclose(centroides, prox_centroides): break
    centroides = prox_centroides
  return prv

def vizinho_mais_proximo(cidades):
    """Comece a rota na primeira cidade; a cada passo estenda a rota
    movendo-se da cidade anterior para a vizinha mais próxima
    que ainda não foi visitada."""
    primeira = cidades[0]
    rota = [primeira]
    inexploradas = cidades.copy()
    inexploradas.remove(primeira)
    while inexploradas:
        C = mais_proxima(rota[-1], inexploradas)
        rota.append(C)
        inexploradas.remove(C)
    return rota

def mais_proxima(c, cidades):
    "Encontre a cidade mais próxima da cidade c."
    return min(cidades, key=lambda x: distancia(c, x))

def solucao_inicial(prv):
  prv = kmeans(prv)
  for i in range(len(prv.rotas)):
    prv.rotas[i].insert(0, prv.clientes[0])
    prv.rotas[i] = vizinho_mais_proximo(prv.rotas[i])

  return prv

#MODIFICADOR
def encontrar(S, j):
  index = -1
  for i in range(S.R):
    try:
      index = S.rotas[i].index(S.clientes[j])
    except ValueError:
      pass
    if index >= 0: break
  return i, index

def trocar(S, i, j):
  ri, li = encontrar(S, i)
  rj, lj = encontrar(S, j)
  del S.rotas[ri][li]
  prv.rotas[ri].insert(li, S.clientes[j])
  del S.rotas[rj][lj]
  prv.rotas[rj].insert(lj, S.clientes[i])

def inverter(S, i, j):
  ri, li = encontrar(S, i)
  rj, lj = encontrar(S, j)
  if ri == rj:
    if li < lj:
       S.rotas[ri][li:lj+1] = reversed(S.rotas[ri][li:lj+1])
    else:
       S.rotas[rj][lj:li+1] = reversed(S.rotas[rj][lj:li+1])

def puxar(S, i, j):
  rj, lj = encontrar(S, j)
  del S.rotas[rj][lj]
  ri, li = encontrar(S, i)
  S.rotas[ri].insert(li, S.clientes[j])

def ciclar(prv, j):
  for i in range(len(prv.rotas)):
    try:
      index = prv.rotas[i].index(prv.clientes[j])
      if index == 1:
        prv.rotas[i-1].append(prv.clientes[j])
        del prv.rotas[i][index]
      else:
        prv.rotas[i][index], prv.rotas[i][index-1] = prv.rotas[i][index-1], prv.rotas[i][index]
    except ValueError:
      pass

def modificar_3ops(S, loop = 1):
    R = []
    list = []
    for x in range(loop):
        #random.seed(x + loop)
        list.append([random.randrange(len(S.clientes)-1) + 1, random.randrange(len(S.clientes)-1) + 1])

    for element in list:
        i = element[0]
        j = element[1]

        R = copy.deepcopy(S)
        if (i == j):
            ciclar(R, j)
        else:
            ri, li = encontrar(S, i)
            rj, lj = encontrar(S, j)
            if (ri == rj): inverter(R, i, j)
            else: puxar(S, i, j)
    return R

#RECOZIMENTO_SIMULADO
def resfriamento(prv, iteracao):
  return (len(prv.clientes)**2)*(0.999)**iteracao

def recozimento_simulado( prv, presfriamento = resfriamento, max_time = 2, max_iterations = 100000):
  start = time.time()
  iteracao = 1
  temperatura = presfriamento(prv, iteracao)
  S = solucao_inicial(prv)
  condicao = temperatura > 1/(prv.R*len(prv.clientes)) and time.time()-start < max_time and iteracao <= max_iterations
  while condicao :
    R = modificar_3ops(S)
    delta = R.custo() - S.custo()
    if delta < 0:
      S = copy.deepcopy(R)
      if S.custo() < prv.custo():
          prv = copy.deepcopy(S)
    else:
      probabilidade = np.exp(-delta / temperatura)
      if np.random.uniform() < probabilidade:
          S = copy.deepcopy(R)
    iteracao += 1
    temperatura = resfriamento(prv, iteracao)
    condicao = temperatura > 1/(prv.R*len(prv.clientes)) and time.time()-start < max_time and iteracao < max_iterations
  return prv

#SUBIDA DA COLINA(BUSCA LOCAL)
def subida_colina(S, max_time = 2, max_iterations = 10000):
    S_local = copy.deepcopy(S)

    start = time.time()
    iterations = 0

    while(time.time() - start < max_time and iterations < max_iterations):
        #random.seed(iterations + 579)
        R = modificar_3ops(S_local)

        if(S_local.custo() > R.custo()):
            S_local = R

        iterations += 1

    return S_local

#BUSCA LOCAL ITERADA
def perturbar(S, k):
    S_local = copy.deepcopy(S)
    size = len(S_local.clientes)
    mod_porcentagem = math.ceil(size * k/100)#90?

    for x in range(mod_porcentagem):
        S_local = modificar_3ops(S_local)

    return S_local

def busca_local_iterada(S, max_time = 2, max_iterations = 10000, number = 30, k = 10):
    S_best = copy.deepcopy(S)
    S_local = copy.deepcopy(S)
    base = copy.deepcopy(S)

    start = time.time()
    iterations = 0

    while(max_time > time.time() - start and max_iterations >= iterations):
        #BUSCA LOCAL
        time_hill = max_time/number
        start_hill = time.time()

        while(time_hill > time.time() - start_hill and max_time >= time.time() - start and max_iterations >= iterations):
            R = modificar_3ops(S_local)

            if(S_local.custo() > R.custo()):
                S_local = R

            iterations += 1

        #CRITÉRIO DE ACEITAÇÃO DA NOVA base
        if(base.custo() > S_local.custo()):
            base = copy.deepcopy(S_local)
            S_best = copy.deepcopy(base)
        elif(random.random() <= (max_time - (time.time() - start))/max_time):
            base = copy.deepcopy(S_local)

        #PERTURBAR
        S_local = perturbar(base, k)

        iterations += 1

    return S_best

#COMPARANDO ALGORITMOS
def benchmark(funcao, entrada):
    "Roda uma uma função e retorna um par (tempo, resultados)."
    t0           = time.process_time()
    resultados = [funcao(x) for x in entrada]
    t1           = time.process_time()
    tempo_medio  = (t1 - t0) / len(entrada)
    return (tempo_medio, resultados)

def benchmarks(algoritmos, entradas):
    "Gera uma tabela com os resultados dos algoritmos passados como entrada."
    for algo in algoritmos:
        tempo, res = benchmark(algo, entradas)
        custos = [ x.custo() for x in res ]
        # Exatrair métricas e gerar tabela
        print("{:>25} |{:7.1f} ±{:4.0f} ({:5.0f} a {:5.0f}) |{:7.3f} segs/instancia | {} ⨉ clientes={}/veiculos={}"
              .format(algo.__name__, statistics.mean(custos), statistics.stdev(custos), min(custos), max(custos),
                      tempo, len(entradas), len(entradas[0].clientes)-1, entradas[0].R))

def gera_instancias(num_instancias=10, num_clientes=30, num_veiculos=3):
    #seed = int(input("Digite semente: "))
    return tuple(gerar_prv(num_clientes, num_veiculos, rseed=r) for r in range(num_instancias))

#TESTANDO ILS PARA DIFERENTES VALORES DE NUMBER(APARENTEMENTE É MELHOR COM 20)
def benchmark_teste(entrada, number):
    "Roda uma uma função e retorna um par (tempo, resultados)."

    t0           = time.process_time()
    resultados = [busca_local_iterada(solucao_inicial(x), number = number) for x in entrada]
    t1           = time.process_time()
    tempo_medio  = (t1 - t0) / len(entrada)
    return (tempo_medio, resultados)

def teste_ILS_k():
    X = []
    Y = []
    for k in range(10, 100, 10):
        X.append(k)
        tempo, res = benchmark_teste(gera_instancias(10, 30, 3), 4)
        print(res)

        custos = [ x.custo() for x in res ]
        Y.append(statistics.mean(custos))
        # Exatrair métricas e gerar tabela
        print("{:>25} |{:7.1f} ±{:4.0f} ({:5.0f} a {:5.0f}) |{:7.3f} segs/instancia | {} ⨉ clientes={}/veiculos={}"
              .format(busca_local_iterada.__name__, statistics.mean(custos), statistics.stdev(custos), min(custos), max(custos),
                      tempo, len(res), len(res[0].clientes)-1, res[0].R))

    plt.plot(X, Y)
    plt.show()

#TESTANDO ILS PARA DIFERENTES VALORES DE PERTUBAÇÃO
def benchmark_teste_2(entrada, k):
    "Roda uma uma função e retorna um par (tempo, resultados)."
    t0           = time.process_time()
    resultados = [busca_local_iterada(solucao_inicial(x), k = k) for x in entrada]
    t1           = time.process_time()
    tempo_medio  = (t1 - t0) / len(entrada)
    return (tempo_medio, resultados)

def benchmarks_teste_2(entradas, k):
    "Gera uma tabela com os resultados dos algoritmos passados como entrada."

    tempo, res = benchmark_teste_2(entradas, k)
    custos = [x.custo() for x in res]
    # Exatrair métricas e gerar tabela
    print("{:>25} |{:7.1f} ±{:4.0f} ({:5.0f} a {:5.0f}) |{:7.3f} segs/instancia | {} ⨉ clientes={}/veiculos={}"
          .format(busca_local_iterada.__name__, statistics.mean(custos), statistics.stdev(custos), min(custos), max(custos),
                  tempo, len(entradas), len(entradas[0].clientes)-1, entradas[0].R))

inicial = solucao_inicial(gerar_prv(30, 3))

algoritmos = [recozimento_simulado, busca_local_iterada]
benchmarks(algoritmos, gera_instancias(100, 50, 4))
#plot_prv(inicial)
'''
for x in range(10, 100, 5):
    print(x)
    benchmarks_teste_2(gera_instancias(10, 30, 3), x)
'''
#teste_ILS_k()
#plot_prv(busca_local_iterada(inicial, max_time = 15, max_iterations = 30000000000000))
#plot_prv(recozimento_simulado(inicial, max_time = 15, max_iterations = 30000000000000))
