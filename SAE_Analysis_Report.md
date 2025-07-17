# Concise SAE Analysis Report

## Experiment Configuration (from sae.py)
- **Model**: Gemma-2-2B-IT
- **SAE**: Gemma-scope-2b-pt-res-canonical, width 16k
- **Layers**: 16, 18, 21
- **Dataset**: AdvBench (harmful) + Alpaca (harmless)
- **Sample Size**: 200 harmful + 200 harmless for both feature scoring and steering evaluation
- **Metric**: Combined (point-biserial + mutual info, normalized_product method)
- **Steering**: ±20.0 strength; single feature and top-3 average
- **Evaluation**: Conditional refusal rates on baseline-refusing prompts

## Results

### Layer 16
- Top Feature: 1131 (score: 1.0)
- Top 3: [1131, 1580, 708]
- Baseline: 190/200 harmful refused, 5/200 harmless
- Steering:
  - Single +20: Harmful 99.47%, Harmless 80%
  - Single -20: Harmful 98.95%, Harmless 80%
  - Top3 +20: Harmful 100%, Harmless 100%
  - Top3 -20: Harmful 100%, Harmless 60%

**Layer Explanation**: Highest correlation; single-feature steering bypasses 1-2 harmful refusals (98.95-99.47%) and reduces harmless false positives (60-80%).

### Layer 18
- Top Feature: 1287 (score: 0.9811)
- Top 3: [1287, 139, 728]
- Baseline: Same as above
- Steering:
  - Single +20: Harmful 100%, Harmless 60%
  - Single -20: Harmful 99.47%, Harmless 40%
  - Top3 +20: Harmful 100%, Harmless 100%
  - Top3 -20: Harmful 99.47%, Harmless 60%

**Layer Explanation**: Strong correlation; -20 steering bypasses 1 harmful (99.47%), strong harmless suppression (40-60%).

### Layer 21
- Top Feature: 1628 (score: 0.9056)
- Top 3: [1628, 1627, 132]
- Baseline: Same as above
- Steering:
  - Single +20: Harmful 100%, Harmless 100%
  - Single -20: Harmful 99.47%, Harmless 40%
  - Top3 +20: Harmful 100%, Harmless 100%
  - Top3 -20: Harmful 100%, Harmless 80%

**Layer Explanation**: Lowest correlation but consistent; -20 single-feature bypasses 1 harmful (99.47%), strong harmless suppression (40%).

## Metric Explanation
- Rates are **conditional**: P(refusal after steering | baseline refusal)
- Lower rate = more bypasses (steering suppresses refusals)
- Focuses on refusal-prone prompts; ignores if steering induces new refusals
- Example: Harmless 80% means 4/5 baseline-refusing harmless prompts still refused (1 bypassed, reducing false positives)

## Brief Explanation
Features show high correlation with refusal (scores >0.9). With larger sample (200), steering achieves ~0.5-1% bypass on harmful prompts (1-2 out of 190) via single-feature methods. Effectively reduces false positives on harmless prompts (down to 40%). Multi-feature less effective overall. Negative steering remains better for bypass; confirms robust safety with limited manipulability. 