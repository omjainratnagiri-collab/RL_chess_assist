
def td_targets(V, outcome, lam):
    """V: list of search scores at plies 0..T-1 (common frame). outcome:
    terminal value in the same frame. Returns G[0..T-1]."""
    T = len(V)
    G = [0.0] * (T + 1)
    G[T] = outcome
    for t in range(T - 1, -1, -1):
        next_v = V[t + 1] if t + 1 < T else outcome
        G[t] = (1 - lam) * next_v + lam * G[t + 1]
    return G[:T]

V = [0.1, 0.3, 0.2, 0.5]
outcome = 1.0

                                                                                     
g0 = td_targets(V, outcome, 0.0)
expected0 = [V[1], V[2], V[3], outcome]
print("lambda=0 (pure bootstrap):", g0, "expected:", expected0)
assert all(abs(a - b) < 1e-9 for a, b in zip(g0, expected0))

                                                                     
g1 = td_targets(V, outcome, 1.0)
expected1 = [outcome] * len(V)
print("lambda=1 (pure outcome):  ", g1, "expected:", expected1)
assert all(abs(a - b) < 1e-9 for a, b in zip(g1, expected1))

                                                                    
g07 = td_targets(V, outcome, 0.7)
print("lambda=0.7 (blend):       ", g07)
for a, b0, b1 in zip(g07, g0, g1):
    lo, hi = min(b0, b1), max(b0, b1)
    assert lo - 1e-9 <= a <= hi + 1e-9

print("\nALL TD(lambda) MATH CHECKS PASSED")