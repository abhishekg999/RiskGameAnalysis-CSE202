from typing import List, Tuple, Set, Callable, NamedTuple
from collections import namedtuple
from functools import wraps
import random

def InternalGameFunction(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


class Graph:
    V: List[int]
    E: List[Tuple[int, int]]

    def __init__(self, V: List[int], E: List[Tuple[int, int]]):
        self.V = V
        self.E = E

    def __repr__(self) -> str:
        return f"Graph(V={self.V}, E={self.E})"

    def neighbors(self, v: int) -> List[int]:
        return [w for u, w in self.E if u == v]

"""
GameState = (G, C, P, Players, Armies)
G: Graph with |V| nodes and |E| edges representing the map
    - V: List of strings representing the territories
    - E: List of tuples representing the edges
        - E[i] = (u, v) if there is an edge between u and v
C: List of sets representing the continents and associated territories
P: List of players
Players: list representing number of armies each player has in each territory
    - Player[i] = j if v_i is controlled by player j
Armies: list representing number of armies in each territory
    - Armies[i] = j if v_i has j armies
"""
class GameState(NamedTuple):
    TerritoryOwner: List[int]
    Armies: List[int]

class GameInstance(NamedTuple):
    G: Graph
    C: List[Set[int]]
    P: List[int]
    ContinentBonus: List[int]



"""
Move = (s, t, c)
s: source territory
t: target territory
c: number of armies to move
"""
t_Move = Tuple[int, int, int] | None
t_DeterministicAttack = Callable[[int, int], Tuple[int, int]]

def Apply(instance: GameInstance, state: GameState, move: t_Move) -> GameState:
    if move is None:
        return state
    else:
        _TerritoryOwner, _Armies = state
        s, t, c = move

        assert _TerritoryOwner[s] != _TerritoryOwner[t]
        A, D = _Armies[s], _Armies[t]
        A_p, D_p = DeterministicAttack(A, D)

        TerritoryOwner = _TerritoryOwner.copy()
        Armies = _Armies.copy()
        Armies[s] = A_p
        Armies[t] = D_p
        
        if D_p == 0:
            TerritoryOwner[t] = TerritoryOwner[s]
            assert c < A_p
            Armies[s] -= c
            Armies[t] += c
        else:
            assert c == 0
            # Can't do anything since the attack failed
            pass
        return GameState(TerritoryOwner, Armies)

@InternalGameFunction
def TerritoryToContinent(instance: GameInstance, v: int) -> int:
    return instance.G.v.index(v)

@InternalGameFunction
def DeterministicAttack(A: int, D: int) -> Tuple[int, int]:
    L = min(A - 2, D)
    A_p = A - L
    D_p = D - L
    # print(f"Attack: ({A}, {D}) -> ({A_p}, {D_p})")
    return (A_p, D_p)

@InternalGameFunction
def ContinentBonus(instance: GameInstance, continent: int) -> int:
    return instance.ContinentBonus[continent]

@InternalGameFunction
def TerritoryBonus(count: int) -> int:
    return max(count // 3, 3)

def objective(instance: GameInstance, state: GameState, p: int) -> int:
    """
    Objective function for player p. 
    Maximize the number of armies gained by player p on the next turn. 
    """

    # Territory bonuses
    territories_owned = sum([1 for i in state.TerritoryOwner if i == p])
    territory_bonus = TerritoryBonus(territories_owned)

    # Continent bonuses
    continent_bonus = 0
    for i, continent in enumerate(instance.C):
        if all([state.TerritoryOwner[j] == p for j in continent]):
            continent_bonus += ContinentBonus(instance, i)
    return territory_bonus + continent_bonus


def _1CnTAttacker(instance: GameInstance, state: GameState, p: int) -> int:
    """
    Trivial MaxTAttacker algorithm for 1CnT.
    """

    stack = []
    attacks = []

    stack.append((state, attacks))
    best = (-1, None)
    while stack:
        state, attacks = stack.pop()

        if attacks and attacks[-1] is None:
            # check if we have a new best
            value = objective(instance, state, p)
            if value > best[0]:
                best = (value, attacks)
            continue
        
        # Try stopping here
        stack.append((state, attacks + [None]))

        # Or continue attacking
        for v, player in enumerate(state.TerritoryOwner):
            if player == p and state.Armies[v] > 2:
                for w in instance.G.neighbors(v):
                    if state.TerritoryOwner[w] != p:
                        A_p, D_p = DeterministicAttack(state.Armies[v], state.Armies[w])
                        # print(A_p, D_p)
                        if D_p == 0:
                            # If the attack will succeed, try moving all possible armies
                            _c = A_p
                            for c in range(1, _c):
                                newState = Apply(instance, state, (v, w, c))
                                _attacks = attacks + [(v, w, c)]
                                stack.append((newState, _attacks))
                        else:
                            assert A_p <= 2
                            newState = Apply(instance, state, (v, w, 0))
                            _attacks = attacks + [(v, w, 0)]
                            stack.append((newState, _attacks))
    return best


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

    _, moves = _1CnTAttacker(gi, gs, 0)
    ans = []
    for m in moves[:-1]:
        ans.append(s[m[1] - 1])
    return ans


def random_subset(s):
    return {x for x in s if random.choice((True, False))}


def generate_test_case(max_set_size=7, max_target=7):
    set_size = random.randint(1, max_set_size)
    full_set = [random.randint(1, max_target) for _ in range(set_size)]
    subset = random_subset(full_set)
    target = sum(subset)
    return (full_set, target)


if __name__ == "__main__":
    s, t = generate_test_case()
    print(s, t)
    ans = SubsetSumReduction(s, t)
    print(ans)
    assert sum(ans) == t
