# Epistemic Tool Policy for Scientific Exploration

This document gives a formal derivation of Synapse's tool-based scientific exploration policy.
The implementation lives in `/Users/henryjiang/Desktop/synapse_v2/core/synapse_brain.py` under `ScientificExplorer` in `epistemic_tools` mode.

## 1) Problem formulation: constrained epistemic control

Let `H={h_1,...,h_n}` be the active hypothesis set for a topic.
At decision step `t`, the agent state is:

- posterior proxy `b_t(h_i)` with `sum_i b_t(h_i)=1`
- uncertainty scores `u_t(h_i) in [0,1]`
- evidence context `E_t`
- tool action history `a_{1:t-1}`

The agent chooses a tool `a_t` from a finite tool set `A`.
Tool execution yields observation `o_t` (retrieved evidence, contradiction signals, calibration output, or experiment plans).

The theoretical objective is a constrained optimization:

maximize `sum_t E[Delta I_t]`
subject to `E[c_t] <= C` and `E[r_t] <= R`

where:

- `Delta I_t` is epistemic gain
- `c_t` is per-step computational/operational cost
- `r_t` is per-step epistemic risk (overconfidence, brittle inference)
- `C,R` are user-specified budgets

This is a constrained partially observed control problem over belief states.

## 2) Lagrangian relaxation and dual policy

We optimize the per-step relaxed objective:

`L_t(a; lambda_t, mu_t) = E[Delta I_t | a, s_t] - lambda_t (E[c_t|a,s_t]-C) - mu_t (E[r_t|a,s_t]-R)`

Tool choice:

`a_t = argmax_{a in A} L_t(a; lambda_t, mu_t)`

Dual updates (projected subgradient ascent):

`lambda_{t+1} = [lambda_t + eta (c_t - C)]_+`
`mu_{t+1}     = [mu_t     + eta (r_t - R)]_+`

with step size `eta > 0`, and `[x]_+ = max(0,x)`.

Interpretation:

- if observed cost exceeds budget, `lambda` increases and expensive tools are penalized more
- if observed risk exceeds budget, `mu` increases and risky tools are penalized more

This gives adaptive regularization rather than fixed heuristics.

## 3) Belief update model

For each hypothesis we maintain `b_t(h)`.
For non-absolute tool updates we use bounded log-odds Bayesian updates:

`logit(b_{t+1}(h)) = logit(b_t(h)) + kappa * delta_t(h)`

where:

- `delta_t(h)` is tool-provided signed evidence signal, clipped to a bounded interval
- `kappa` is a sensitivity constant

Then probabilities are normalized over hypotheses to enforce simplex constraints.

Uncertainty update uses tool-specific deltas plus conservative shrinkage under stronger evidence magnitude.

### Why this matters

It removes arbitrary additive belief jumps and enforces stable posterior dynamics in probability space.

## 4) Information gain definition

Two gains are tracked:

1. Entropy drop (normalized Shannon entropy)

`Delta H_t = max(0, H(b_t) - H(b_{t+1}))`

2. KL gain (posterior shift relative to prior)

`Delta KL_t = D_KL(b_{t+1} || b_t)`

with `D_KL(p||q) = sum_i p_i log(p_i/q_i)`.

`Delta KL_t >= 0` always, giving a non-negative progress signal even when entropy is flat.

## 5) Stopping rule

Exploration halts if one of the following holds:

1. low posterior entropy and at least one concrete experiment is produced
2. information-gain plateau for consecutive steps
3. step budget exhausted

This defines an anytime policy: it can stop early when the posterior is concentrated enough for action.

## 6) Theoretical properties (under standard assumptions)

Assume bounded per-step gain/cost/risk and bounded stochastic noise in tool observations.

### Proposition 1: Posterior validity
Belief normalization after each update ensures `b_t` remains in the probability simplex for all `t`.

### Proposition 2: KL non-negativity
For every step, `Delta KL_t >= 0` by Gibbs inequality.

### Proposition 3: Dual feasibility tendency
Under convex surrogate losses and standard diminishing step-size conditions, projected dual ascent drives average constraint violation toward zero in expectation.

### Proposition 4: Bounded-update stability
With clipped evidence signals and bounded sensitivity, single-step log-odds updates are bounded, preventing catastrophic posterior collapse in one action.

These are the core reasons the policy is mathematically constrained and auditable, not free-form chain-of-thought tool use.

## 7) Algorithm sketch

1. Initialize hypothesis set and normalized beliefs from retrieved evidence.
2. At each step:
   - estimate per-tool expected gain/cost/risk
   - choose `argmax` relaxed Lagrangian utility
   - execute tool, obtain structured observation
   - perform Bayesian-style posterior update
   - compute entropy/KL gains
   - update dual variables `(lambda, mu)`
3. stop by entropy or plateau condition
4. emit ranked hypotheses + experiment plan + audit trace

## 8) What is logged for reproducibility

Each step logs:

- selected tool
- expected utility and expected IG
- realized entropy gain and realized KL gain
- observed cost/risk
- dual variables `lambda, mu`
- entropy before/after

Run summary logs:

- total entropy gain
- total KL gain
- final entropy
- final dual variables
- stopping reason

These are sufficient to replay and diagnose policy behavior without hidden state.

## 9) Why this is stronger than generic tool-use

Generic tool-use frameworks decide actions heuristically.
This policy explicitly optimizes an epistemic objective with constraint handling and normalized posterior dynamics.
That gives:

- formally interpretable decision traces
- principled cost-risk tradeoffs
- mathematically controlled stopping
- clearer path to theorem-backed analysis and review-ready methodology sections
