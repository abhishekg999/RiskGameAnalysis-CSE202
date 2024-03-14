from risk import *

def random_subset(s):
    return {x for x in s if random.choice((True, False))}

def generate_test_case(max_set_size=7, max_target=7):
    set_size = random.randint(1, max_set_size)
    full_set = [random.randint(1, max_target) for _ in range(set_size)]
    subset = random_subset(full_set)
    target = sum(subset)
    return (full_set, target)

def SubsetSumReduction(s: List[int], t: int):
    """
    Reduction from SubsetSum to 1CnT.
    """

    V = list(range(len(s) + 1))
    E = [(0, w) for w in V[1:]]
    G = Graph(V, E)
    C = [{s} for s in V]
    P = [0, 1]
    
    # Each troop "defeated" worths 2 * (territory bonus + 1)
    scale = (TerritoryBonus(len(V)) + 1)

    CB = [0] + [v * scale for v in s]
    gi = GameInstance(G, C, P, CB)

    TO = [0] + [1] * len(s)
    A = [2*t + 1] + [2*v - 1 for v in s] 
    gs = GameState(TO, A)

    moves = Alg1CnTAttacker(gi, gs, 0)
    ans = []
    for m in moves[:-1]:
        ans.append(s[m[1] - 1])
    return ans


def HamiltonianPathReduction(G: Graph):
    """
    Reduction from HamiltonianPath to 1CMaxT.
    """

    g_map = {v: i + 1 for i, v in enumerate(G.V)}
    g_map_inv = {v: k for k, v in g_map.items()}

    V = list(range(len(G.V) + 1))
    E = [(g_map[u], g_map[v]) for u, v in G.E] + [(0, v) for v in V[1:]]
    G = Graph(V, E)
    C = [{v  for v in V}]
    P = [0, 1]
    CB = [0] + [1] * (len(V) - 1)
    gi = GameInstance(G, C, P, CB)

    TO = [0] + [1] * (len(V) - 1)
    A = [len(V)**2] + [1] * (len(V) - 1)
    gs = GameState(TO, A)

    moves = Alg1CMaxTAttacker(gi, gs, 0)
    return [g_map_inv[m[1]] for m in moves[:-1]]


if __name__ == "__main__":
    V = range(6)
    E = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (1, 5), (2, 5), (3, 4), (3, 5)]
    G = Graph(V, E)
    

    moves = HamiltonianPathReduction(G)
    print(moves)