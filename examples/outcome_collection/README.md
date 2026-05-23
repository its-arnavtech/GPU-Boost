# Controlled Outcome Collection Templates

This folder is for controlled local GPUBoost experiments that produce real
optimization outcome labels from measured benchmark results.

Place baseline and optimized benchmark JSON files here, or reference their local
paths in `pairs.example.json`. Use real measured benchmark JSON from controlled
experiments. Do not use AI-generated data as production training data.

These files are templates only. Labels are derived from GPUBoost comparison
results, not manual guessing.

Current Phase 11.8 tooling compares existing benchmark JSON files only. It does
not execute benchmark commands, run user code, download data, scrape websites, or
call external APIs.
