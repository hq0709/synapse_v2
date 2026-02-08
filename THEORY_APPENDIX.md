# Theoretical Appendix: Constrained Epistemic Tool Policy

This appendix expands the derivation and proof sketches used by the epistemic tool policy.

## A. Setup

Let `(S_t, A_t, O_t)` define a belief-state control process.
State `S_t` includes hypothesis posterior `b_t`, uncertainty vector `u_t`, and evidence cache `E_t`.
Action `A_t` is tool selection.
Observation `O_t` is structured output of the selected tool.

Define per-step random variables:

- gain `g_t := Delta I_t`
- cost `c_t`
- risk `r_t`

Budgets: `C` for cost and `R` for risk.

## B. Primal constrained objective

`max_pi E_pi[ sum_{t=1}^T g_t ]`
subject to
`E_pi[ (1/T) sum_t c_t ] <= C`
`E_pi[ (1/T) sum_t r_t ] <= R`

## C. Lagrangian and dual

Lagrangian:

`L(pi, lambda, mu) = E_pi[sum_t g_t] - lambda (E_pi[sum_t c_t]-TC) - mu (E_pi[sum_t r_t]-TR)`

Dual function:

`D(lambda, mu) = max_pi L(pi, lambda, mu)`, with `lambda,mu >= 0`.

Online approximation uses per-step greedy maximization of local surrogate:

`a_t = argmax_a E[g_t | s_t,a] - lambda_t(E[c_t|s_t,a]-C) - mu_t(E[r_t|s_t,a]-R)`

Dual ascent:

`lambda_{t+1} = [lambda_t + eta_t (c_t - C)]_+`
`mu_{t+1}     = [mu_t     + eta_t (r_t - R)]_+`

## D. Belief update in log-odds space

For a hypothesis `h` with prior `p_t(h)`, define bounded evidence signal `delta_t(h)` and sensitivity `kappa`.

`logit(p_{t+1}(h)) = logit(p_t(h)) + kappa delta_t(h)`

Equivalent multiplicative form:

`odds_{t+1}(h) = odds_t(h) * exp(kappa delta_t(h))`

Then normalize over all hypotheses:

`b_{t+1}(h_i) = p_{t+1}(h_i) / sum_j p_{t+1}(h_j)`

This keeps beliefs valid and limits one-step drift when `delta_t` is clipped.

## E. Information gain quantities

### E.1 Entropy drop

`H(b) = - sum_i b_i log b_i / log n`

`Delta H_t = max(0, H(b_t)-H(b_{t+1}))`

### E.2 KL gain

`Delta KL_t = D_KL(b_{t+1} || b_t) = sum_i b_{t+1,i} log( b_{t+1,i} / b_{t,i} )`

`Delta KL_t >= 0` by Gibbs inequality.

KL can detect meaningful posterior shifts even if entropy decreases only slightly.

## F. Proof sketches

### Theorem 1 (Simplex invariance)
Assuming finite positive initialization and normalization after each update, `b_t` remains in the simplex for all `t`.

Sketch: each component is positive after clipped logistic update; normalization enforces unit sum.

### Theorem 2 (Bounded one-step posterior drift)
If `|delta_t(h)| <= d_max`, then log-odds increment magnitude is bounded by `kappa d_max`.

Sketch: direct from update equation; this bounds multiplicative odds ratio and prevents single-step collapse.

### Theorem 3 (Dual regret-style bound, sketch)
Under convex surrogate losses and bounded subgradients, projected dual ascent with step size `eta_t ~ 1/sqrt(t)` yields sublinear dual regret and vanishing average constraint violation.

Sketch: standard online convex optimization argument for projected subgradient methods.

### Theorem 4 (Anytime stopping validity)
If stop is triggered only when entropy is below threshold and at least one falsifiable experiment exists, output satisfies a minimum confidence-actionability criterion.

Sketch: low entropy gives concentrated posterior; experiment-existence enforces actionability constraint.

## G. Practical implications

1. `lambda` and `mu` are interpretable control knobs with automatic adaptation.
2. Gain/cost/risk tradeoff is explicit and auditable at every step.
3. Posterior trajectories are stable and replayable.
4. Stop condition is principled, not purely token-budget based.

## H. Suggested camera-ready theorem section

For manuscript text, report:

- constrained objective and dual updates
- bounded posterior drift statement
- KL non-negativity and entropy criterion
- assumptions used by dual-convergence argument

This keeps claims strong but defensible before full empirical benchmarking.
