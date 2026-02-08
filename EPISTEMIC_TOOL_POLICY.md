# Epistemic Tool Policy for Scientific Exploration

This document specifies the theoretical core of Synapse's tool-based scientific exploration mode.

## Core idea

Scientific discovery is formulated as a sequential decision process over a hypothesis posterior.
At each step, the agent selects the tool that maximizes expected epistemic value under cost and risk constraints.

## State

At step `t`, the agent tracks an epistemic state:

- `H_t = {h_i}`: hypothesis set
- `b_t(h_i)`: belief (posterior proxy) for hypothesis `h_i`
- `u_t(h_i)`: uncertainty for hypothesis `h_i`
- `E_t`: evidence memory context
- `A_{<t}`: action history

Normalized entropy over belief mass is used as a stopping/control signal:

`Entropy(b_t) = -sum_i p_i log p_i / log |H_t|` where `p_i = b_t(h_i) / sum_j b_t(h_j)`.

## Tool selection objective

The policy chooses tool `a_t` by:

`a_t = argmax_a [ E[Delta I(H; O | a, b_t)] - lambda * Cost(a) - mu * Risk(a) ]`

Implementation uses calibrated proxies:

- expected information gain (`expected_information_gain`)
- execution cost (`cost`)
- epistemic risk (`risk`, e.g., hallucination/overconfidence risk)
- utility = `expected_information_gain - lambda_cost * cost - mu_risk * risk`

## Tool set

Current tools:

1. `retrieve_evidence`
2. `propose_hypotheses`
3. `audit_contradictions`
4. `design_experiments`
5. `recalibrate_beliefs`

Each tool emits structured updates:

- belief deltas (`support_delta`)
- uncertainty deltas (`uncertainty_delta`)
- new hypotheses or experiment plans
- qualitative insight traces

## Stopping rules

Exploration stops when one of the following is met:

1. `Entropy(b_t) <= stop_entropy` and at least one experiment plan exists.
2. Realized information gain plateaus for consecutive steps.
3. Step budget exhausted.

## Logged quantities

Each step logs:

- selected tool
- expected utility and expected information gain
- realized information gain (entropy drop)
- entropy before/after update

Run-level summary logs:

- total information gain
- final entropy
- stopping reason

## Why this is not generic tool-use

This framework is not plain ReAct. Tool selection is constrained by a quantitative epistemic objective and explicit stopping criteria, enabling reproducible evaluation of:

- posterior concentration dynamics
- gain-per-step efficiency
- contradiction robustness
- experiment planning quality under uncertainty
