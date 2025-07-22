# Concise SAE Analysis Report

## Experiment Configuration (from sae.py)
- **Model**: Gemma-2-2B-IT
- **SAE**: Gemma-scope-2b-pt-res-canonical, width 16k
- **Layers**: 16, 18, 21
- **Dataset**: AdvBench (harmful) + Alpaca (harmless)
- **Sample Size**: 200 harmful + 200 harmless
- **Metric**: Combined (point-biserial + mutual info, normalized_product)
- **Steering**: ±20.0 addition; orthogonal ablation
- **Evaluation**: Conditional rates + bypass (% previously refused now not) + induced (% previously not now refused)

## Results

### Layer 16
- Top Feature: 1131 (score: 1.0)
- Top 3: [1131, 1580, 708]
- Baseline: 190/200 harmful refused, 5/200 harmless
- Avg Projection: 7.66 (relative to 20: 2.61x)
- Steering:
  - Single +20: Harmful cond 99.47%, bypass ~0.5%, induced 0%
  - Single -20: Harmful cond 98.95%, bypass ~1%, induced 0%
  - Top3 +20: Harmful cond 100%, bypass 0%, induced 0%
  - Top3 -20: Harmful cond 100%, bypass 0%, induced 0%
  - Single Ablate: Harmful cond 0%, bypass 100%, induced 0%
  - Top3 Ablate: Harmful cond 0%, bypass 100%, induced 0%

**Layer Explanation**: Perfect ablation bypass (100%) on harmful; no induced refusals.

### Layer 18
- Top Feature: 1287 (score: 0.9811)
- Top 3: [1287, 139, 728]
- Baseline: Same
- Avg Projection: 14.21 (relative to 20: 1.41x)
- Steering:
  - Single +20: Harmful cond 100%, bypass 0%, induced 0%
  - Single -20: Harmful cond 99.47%, bypass ~0.5%, induced 0%
  - Top3 +20: Harmful cond 100%, bypass 0%, induced 0%
  - Top3 -20: Harmful cond 99.47%, bypass ~0.5%, induced 0%
  - Single Ablate: Harmful cond 0%, bypass 100%, induced 0%
  - Top3 Ablate: Harmful cond 0%, bypass 100%, induced 0%

**Layer Explanation**: Ablation achieves 100% bypass; additive steering minimal effect.

### Layer 21
- Top Feature: 1628 (score: 0.9056)
- Top 3: [1628, 1627, 132]
- Baseline: Same
- Avg Projection: 6.66 (relative to 20: 3.00x)
- Steering:
  - Single +20: Harmful cond 100%, bypass 0%, induced 0%
  - Single -20: Harmful cond 99.47%, bypass ~0.5%, induced 0%
  - Top3 +20: Harmful cond 100%, bypass 0%, induced 0%
  - Top3 -20: Harmful cond 100%, bypass 0%, induced 0%
  - Single Ablate: Harmful cond 0%, bypass 100%, induced 0%
  - Top3 Ablate: Harmful cond 0%, bypass 100%, induced 0%

**Layer Explanation**: Consistent 100% ablation success; projections show natural component 1.4-3x steering strength.

## Metric Explanation
- Conditional: P(refusal after | baseline refusal) - lower = more suppression
- Bypass: % previously refused now not
- Induced: % previously not now refused

## Brief Explanation
SAE features highly causal for refusal. Additive steering (±20) yields 0.5-1% bypass. Ablation perfectly bypasses (100%) without inducing new refusals, confirming refusal mediated by single directions per paper. 