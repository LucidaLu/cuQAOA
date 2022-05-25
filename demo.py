from cuQAOA import cuQAOASim
from cuQAOA.maxcut import solve_maxcut, GraphType

from networkx import nx

N = 12
G = nx.erdos_renyi_graph(n=N, p=0.8)

cq = cuQAOASim([sum(1 if s >> x & 1 != s >> y & 1 else -1 for x, y in G.edges) for s in range(1 << N)])
print('random argument qaoa result =', (len(G.edges) - cq([1, 2, 3, 4], [5, 6, 7, 8])) / 2)
print('optimized argument qaoa result =', solve_maxcut(G, 4, gtype=GraphType.ERDOS_RENYI))
