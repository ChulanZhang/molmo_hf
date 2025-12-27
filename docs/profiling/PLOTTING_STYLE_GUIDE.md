# Plotting Style Guide

Unified plotting guidelines for profiling experiments.

## General
- Use consistent color palettes and line styles across plots.
- Label axes with units; include legends when multiple series exist.
- Prefer PDF/PNG outputs with vector-friendly options where possible.

## Fonts and sizes
- Title: 14–16 pt
- Axis labels: 12–14 pt
- Tick labels: 10–12 pt
- Legend: 10–12 pt

## Colors (suggested)
- Accuracy curves: blues
- Latency curves: oranges/reds
- Pareto frontiers: highlight with thicker lines and markers

## Layout
- Keep margins sufficient for labels; avoid clipping.
- For multi-panel figures, share axes when comparable.
- Use grids sparingly to improve readability.

## File naming
- `plot_{experiment}.py` for generators
- Output files: include experiment name, metric, and dataset if relevant, e.g., `exp5_accuracy_coco.png`

## Reproducibility
- Seed matplotlib/NumPy where randomness is used.
- Document input data source and command used to generate each plot.
