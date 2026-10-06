# The knee in closed form, the direction as a theorem, and a first-order transition

Status: theory, computed 2026-10-05 on the exact objects of `vendor/gleam/kernel_general.py`
(validated against the frozen `prereg_e10.py` tables before anything else is printed).
No training, no seeds. Generator: `analysis/knee_formula.py` (`--slices` for the full table,
`analysis/knee_formula_table.json`); bistability scans: `analysis/bistability.py`.
Everything below is a statement about the linear-Gaussian family at the LSTD(λ) /
averaged-update fixed point; it is not a statement about PPO's finite-sample dynamics.

## 1. The exact GAE(λ) gradient has a two-factor form

For `u_t = -k v_t + eta_t`, `eta ~ N(0, s_e)`, and the observation-restricted LSTD(λ)
critic on `(1, v, v^2)`,

    ghat(k; λ) = 2h [ r k Var(v) − (q + r k²) · Φ_k(βλ) · Λ_k(λ) / s_e ]          (I)

    Φ_k(μ)     = Σ_{j≥1} μ^j x_j y_j                  truncated kick-response sensitivity
    Λ_k(λ)     = 1 / (1 − (1−λ) Ñ_k(λ))               critic aliasing amplification
    Ñ_k(λ)     = Σ_{j≥0} (βλ)^j ρ_j²,   ρ_j = x_j/x_0  (observation autocorrelation)
    x_j = E[v_t v_{t+j}],  y_j = E[eta_t v_{t+j}] = s_e e1ᵀ F_k^{j−1} G.

Derivation. Write the λ-advantage as
`A_t = Σ_j (βλ)^j c_{t+j} − V(v_t) + β(1−λ) Σ_{j≥1} (βλ)^{j−1} V(v_{t+j})`.
With score `s_t = −v_t eta_t / s_e`: `E[s_t V(v_t)] = 0` because `eta_t ⊥ v_t`;
`E[s_t c_t] = 2 h r k Var(v)`; for `j ≥ 1` Isserlis gives
`E[s_t c_{t+j}] = −(2h(q+rk²)/s_e) x_j y_j` and `E[s_t V(v_{t+j})] = −(2θ₂/s_e) x_j y_j`.
Summing, `ghat = 2hrk Var(v) − (2Φ(βλ)/s_e)[h(q+rk²) + θ₂(1−λ)/λ]`. The LSTD(λ) fixed
point is `θ₂ = h(q+rk²) λ Ñ / (1 − (1−λ)Ñ)` — the denominator
`Σ_j (βλ)^j (x_j² − β x_{j+1}²)` telescopes to `(x_0² − (1−λ)N)/λ` — so the bracket is
`h(q+rk²) Λ`. ∎  Verified against the resolvent implementation to 6e-13 on 3 kernels ×
6 gains × 8 lambdas.

Reading: everything the aliased critic does to the policy gradient is one scalar,
`Λ_k(λ) ≥ 1`. The remaining λ-dependence, `Φ_k(βλ) ≤ Φ_k(β)`, is the honest truncation of
the horizon and would, alone, make the agent *under*-estimate the benefit of control. The
paper's §3.2–3.3 (θ₂ and ĝ as separate resolvent expressions) can be replaced by (I).

## 2. The direction of the displacement is a theorem about timescales

Let `B_k(λ) = Φ_k(βλ)Λ_k(λ)/s_e` be the perceived benefit of control per unit `(q+rk²)`,
so the equilibrium condition is `L(k) := r k Var_k(v)/(q+rk²) = B_k(λ)`. To first order in
`(1−λ)`,

    B_k(λ) = B_k(1) [ 1 + (1−λ) C_k + O((1−λ)²) ],     C_k = Ñ_k(1) − T_k,
    T_k = Σ_j j β^j x_j y_j / Σ_j β^j x_j y_j            (mean lag of the kick response).

`Ñ_k(1)` is the discounted squared-autocorrelation time of the *observation*; `T_k` is the
lag over which a control kick is still felt. Whenever the hidden state outlives the control
response, `C_k > 0`: the critic inflates the perceived benefit of control for every λ < 1
and the equilibrium gain is displaced *upward* (over-control), toward the plateau. On every
instance below `C_k > 0`; at the paper's instance `Ñ = 11.2`, `T = 2.7`. At fixed gain the
two factors pull opposite ways and Λ wins by a wide margin (k = 6, H = 8: `Λ = 2.77`,
`Φ(βλ)/Φ(β) = 0.71`, product 1.97).

## 3. The knee: where λ½ sits and why it is far beyond the correlation time

At the gain-midpoint knee `k_AC(λ_K) = k_m := √(k₀ k₁)` by definition (`k₀` TD(0)
equilibrium, `k₁` λ=1 equilibrium), so `λ_K` is the scalar root of `B_{k_m}(λ) = L(k_m)`:
two gains and one 1-D root, no `k_AC(λ)` curve. Expanding `B` to first order,

    1 − λ½  ≈  (R_m − 1) / C_m,        R_m = L(k_m) / B_{k_m}(1),                     (II)
    H½ = 1/(1 − βλ½)  ≈  1 / [ (R_m − 1)/C_m + (1 − β) ].

`R_m − 1` is the normalised *true* policy gradient at the midpoint gain — how hard the
unbiased λ=1 objective pulls `k_m` back toward `k₁`. It is small (0.06–0.5 on every instance
tested) because the memoryless cost surface `J(k)` is flat between `k₁` and the plateau.
`C_m` is the critic's aliasing budget in steps, of order the hidden correlation time. Hence

> the knee horizon is the hidden correlation time *divided by the flatness of the memoryless
> cost landscape*.

That is why `H½` runs 1.5–6× the correlation time rather than tracking it, why λ = 0.95 is
unsafe for a 20-step hidden coordinate, and why λ½ is not ordered by kernel correlation time
across kernel shapes (`R_m` is not): the "2-mode slow-tail" kernel has `τ_K = 25` but a
smaller `Ñ(1)` (its slow tail carries little weight in the observation) and a larger `R_m`,
so an earlier knee.

Validation (`analysis/knee_formula.py --slices`; lamE = cost-midpoint knee, the paper's
definition; lamK = gain-midpoint knee; lam_asym = (II)):

| instance | τ (steps) | lamE | lamK | lam_asym | R_m−1 | Ñ(1) | T | C | H_E | H_E/τ |
|---|---|---|---|---|---|---|---|---|---|---|
| GLEAM_g1_h0.02 | 50 | 0.99655 | 0.99655 | 0.99645 | 0.079 | 26.7 | 4.3 | 22.4 | 225 | 4.5 |
| GLEAM_g1_h0.05 | 20 | 0.98540 | 0.98540 | 0.98552 | 0.122 | 11.2 | 2.7 | 8.5 | 59 | 2.9 |
| GLEAM_g2_h0.05 | 10 | 0.95800 | 0.95800 | 0.95778 | 0.188 | 7.4 | 2.9 | 4.5 | 22 | 2.3 |
| GLEAM_g4_h0.05 | 5 | 0.87117 | 0.87193 | 0.86712 | 0.324 | 5.4 | 2.9 | 2.4 | 7.6 | 1.5 |
| 2-mode balanced (dim 3) | 29 | 0.98368 | 0.98393 | 0.98520 | 0.144 | 12.2 | 2.5 | 9.7 | 53 | 1.8 |
| 2-mode slow-tail (dim 3) | 25 | 0.97316 | 0.97331 | 0.97593 | 0.163 | 9.5 | 2.7 | 6.8 | 34 | 1.4 |
| 3-mode broad (dim 4) | 31 | 0.98436 | 0.98461 | 0.98582 | 0.142 | 12.5 | 2.5 | 10.0 | 55 | 1.8 |

Across the 31 one-parameter slices (γ, c, h, ρ, s_e, Θ around the paper's instance), (II)
reproduces the exact fixed-gain knee within 0.003 wherever `1 − λ ≤ 0.05`, degrading to
0.04 at λ ≈ 0.63 (γ = 8) where the expansion parameter is no longer small. Against the
paper's cost-midpoint knee λ_E (which differs from the fixed-gain knee where the E(λ)
profile is asymmetric) the same formula is within 0.005 for every continuous instance with
`1 − λ ≤ 0.03` and within 0.007 up to `1 − λ = 0.05` (h = 0.1); the first-order instances
(Π > 100, section 4) are excluded from both statements because their λ_E is the jump of the
largest-root branch, not a smooth crossing (`analysis/figures_knee_formula.py`). Observations the
table makes exact: `s_e` and `Θ` enter only through `s_e/Θ` (identical knees at
`(s_e, Θ) = (0.3, 1)` and `(0.09, 0.3)`); the discount caps the horizon through `(1−β)`;
smaller `h` pushes the knee out as `H ∝ h^{−1.6}` because the TD(0) plateau gain scales as
`k₀ ≈ 1.2/h`.

## 4. The collapse is first-order for strongly coupled or weakly exploring instances

Scanning the roots of `ghat(k; λ)` in `k` (`analysis/bistability.py`): for some instances
the averaged update has **three** equilibria over a window of λ — a stable low-gain
branch near `k₁`, a stable high-gain branch near the plateau, and an unstable one between —
and the paper's `k_AC(λ)` ("largest root") jumps discontinuously at the window's upper edge.
At the paper's discount and cost weights (ρ = 0.05, q = r = 1) the order of the transition is
decided by one dimensionless number, the signal-to-exploration ratio `Π := Θc/(h s_e)`
(stationary variance of the hidden force over the per-step exploration variance injected into
`v`): **continuous for Π ≤ 100, first-order for Π ≥ 102**, with the boundary unchanged across
`h ∈ {0.02, 0.05, 0.1}` and `γ ∈ {0.4, 1, 2, 4}`. The boundary moves with the other two
dimensionless groups: a heavier discount (ρ = 0.2) or a larger control weight (r = 3) makes
Π = 90 first-order; a lighter discount (ρ = 0.005) or a smaller control weight (r = 0.3) keeps
Π = 120 continuous. So Π* is a function of (ρh, r/q) with Π*(0.0025, 1) ≈ 101.

| instance | Π | collapse | bistable window | branches at centre (S/U) |
|---|---|---|---|---|
| GLEAM_g1_h0.05, g2, g4 | 67 | continuous | — | — |
| GLEAM_g1_h0.02 | 167 | first-order | [0.99529, 0.99655] | 2.50 S, 6.51 U, 20.8 S |
| c = 0.5 | 111 | first-order | [0.98326, 0.98616] | |
| c = 1.0 | 222 | first-order | [0.97172, 0.98637] | 1.39 S, 4.77 U, 19.9 S |
| c = 3.0 | 667 | first-order | [0.90260, 0.98273] | 0.77 S, 3.80 U, 30.1 S |
| s_e = 0.03 (≡ Θ = 3) | 200 | first-order | [0.98807, 0.99030] | |
| s_e = 0.01 | 600 | first-order | [0.98876, 0.99309] | |
| h = 0.01 | 333 | first-order | [0.99778, 0.99886] | |
| s_e = 0.06, 0.07, 0.3; c = 0.1, 0.4; h = 0.1 | ≤ 100 | continuous | — | — |

Consequences, stated for the record:
- One of the paper's four instances, `GLEAM_g1_h0.02`, is first-order. Its frozen λ½ =
  0.996552 is the *upper* edge of the window (the jump of the largest-root branch). The
  grid-matched estimator cannot see a jump between 0.995 and 0.999, so E7/E10's pass on this
  instance is unaffected and uninformative about the order.
- Inside a window the equilibrium a run reaches depends on where it starts: from a small
  gain (the mlp's zero-initialised head) the lower branch is reached first, so training from
  scratch should show the collapse at the *lower* edge; a run started at `k_AC` (E10's
  linear arm, `--linear-gain-init`) sits on the upper branch until its edge. Hysteresis and
  seed-bimodal costs near the knee are predicted for Π > 100 instances and should not be read
  as non-convergence.
- The paper's "fixed point, not limit" caveat (§3.3) now has a concrete form: for Π > 100
  the fixed point is not unique.

## 5. What is and is not claimed

Exact: (I) and the fixed-gain reformulation of λ_K. First-order asymptotic: (II), with the
validation table as its error budget. Computed, not derived: the Π ≈ 100 boundary (a
scan, reproducible from `analysis/bistability.py`). All of it is for the linear family with
a quadratic critic; E13 says the thresholds survive a 50% cubic perturbation, which is as
far as the evidence goes.
