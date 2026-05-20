"""
Compute exact partition function for L=8 2D Ising model (PBC)
via transfer matrix method, and append results to etc/exactz.md.

lnZ convention matches exactz.md:
  - lnZ: discrete partition function ln Z_discrete
  - fix:  Hubbard-Stratonovich normalization constant
          fix = (N/2)*(offset + ln π) + (1/2)*logdet(K)
          where K = Adj/T + offset*I, offset = 0.1 - min_eigval(Adj/T)
  - The training target (loss*) = -(lnZ + fix) = -lnZ_continuous
"""
import re
import numpy as np

TEMPERATURES = [
    2.0, 2.1, 2.2, 2.269185314213022, 2.3, 2.4,
    2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0, 4.5, 5.0
]

L = 8
N = L * L


def exact_lnZ(L, T):
    K = 1.0 / T
    n = 2 ** L

    idx = np.arange(n, dtype=np.int32)
    spins = np.where((idx[:, None] >> np.arange(L)[None, :]) & 1, 1.0, -1.0)

    horiz = np.sum(spins * np.roll(spins, -1, axis=1), axis=1)
    vert = spins @ spins.T

    log_T = K * vert + 0.5 * K * horiz[:, None] + 0.5 * K * horiz[None, :]

    shift = log_T.max()
    T_mat = np.exp(log_T - shift)

    eigs = np.linalg.eigvalsh(T_mat)

    max_eig = np.abs(eigs).max()
    lnZ = L * shift + L * np.log(max_eig) + np.log(np.sum((eigs / max_eig) ** L))

    return lnZ


def build_adj(L):
    N = L * L
    Adj = np.zeros((N, N))
    for i in range(N):
        row, col = divmod(i, L)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            j = ((row + dr) % L) * L + (col + dc) % L
            Adj[i, j] = 1.0
    return Adj


def compute_hs_fix(L, T, Adj):
    """
    Compute the HS normalization constant so that:
        lnZ_continuous = lnZ_discrete + fix
    Matches exactly the K matrix construction in source/ising.py.
    """
    N = L * L
    K_raw = Adj / T
    w = np.linalg.eigvalsh(K_raw)
    offset = 0.1 - w.min()          # same as source/ising.py
    K = K_raw + np.eye(N) * offset
    sign, logdet = np.linalg.slogdet(K)
    fix = 0.5 * N * (offset + np.log(np.pi)) + 0.5 * logdet
    return fix


if __name__ == "__main__":
    Adj = build_adj(L)

    print(f"\n### Exact Z for Ising n={L}, T from 2.0 to 5.0\n")
    print(f"| {'T':<22} | {'lnZ':<22} | {'fix':<22} | {'sum/n':<12} |")
    print(f"| {'-'*22} | {'-'*22} | {'-'*22} | {'-'*12} |")

    rows = []
    for T in TEMPERATURES:
        lnZ = exact_lnZ(L, T)
        fix = compute_hs_fix(L, T, Adj)
        per_spin = (lnZ + fix) / N
        print(f"| {T:<22} | {lnZ:<22.15g} | {fix:<22.15g} | {per_spin:<12.8f} |")
        rows.append((T, lnZ, fix, per_spin))

    # Remove old (wrong) L=8 section from exactz.md, then append correct one
    out_path = "etc/exactz.md"
    with open(out_path, "r") as f:
        content = f.read()

    # Strip any existing L=8 section (appended by previous runs)
    content = re.sub(
        r"\n*### Exact Z for Ising n=8, T from.*",
        "",
        content,
        flags=re.DOTALL,
    ).rstrip()

    section = f"\n\n\n### Exact Z for Ising n={L}, T from 2.0 to 5.0\n\n"
    section += f"| T                 | $lnZ$              | fix                | sum/n      |\n"
    section += f"| ----------------- | ------------------ | ------------------ | ---------- |\n"
    for T, lnZ, fix, per_spin in rows:
        section += f"| {T:<17} | {lnZ:<18.15g} | {fix:<18.15g} | {per_spin:<10.8f} |\n"

    with open(out_path, "w") as f:
        f.write(content + section)

    print(f"\nReplaced L={L} section in {out_path}")
