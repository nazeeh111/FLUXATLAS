# FLUXATLAS launch film

## Planning rubric
1. App: educational fixed-source neutron-transport laboratory comparing independent OpenMC simulations with a genuinely trained, uncertainty-aware spatial surrogate.
2. Strongest claim: the prediction stays beside an independent reference and its uncertainty; evidence remains inspectable.
3. Visual hook: two real flux maps with a shared scale and a small exact geometry outline.
4. Actual UI: the transport observatory, including reference/model comparison and research record.
5. Shortest satisfying video: 20 seconds.
6. Tone: polished; a scientific instrument film on warm paper.
7. Audio: intentional silence, so this remains suitable for a research README and quiet laboratory presentation. No narration requested.
8. Caption: FLUXATLAS pairs independent OpenMC transport with trained neural field predictions, visible uncertainty and a reproducible research record. Educational synthetic data, open source.
9. User flow: choose a saved configuration → compare independent reference and model → inspect uncertainty and acquisition evidence.

## Storyboard (20 seconds total)
- 0–4 s: “Learn the field. Keep the reference.” Brand and a real reference map reveal. Tiny provenance reads “Educational · synthetic two-group neutron transport”.
- 4–10 s: “Independent physics. Trained prediction.” Side-by-side real OpenMC/reference and trained prediction maps. Data labels identify the exact case, source histories, group and normalization. The shared logarithmic color scale is explicit.
- 10–16 s: “Uncertainty belongs in the picture.” A real error map and saved active/random acquisition study establish the method. No performance superiority is implied. Subline: “Held-out fields · calibrated intervals · history-matched trials”.
- 16–20 s: Actual observatory screenshot with FLUXATLAS wordmark and GitHub URL. Closing statement: “Reproduce. Compare. Inspect.”

All data visuals are generated from web/atlas.json after training completes. No placeholder numerical evidence enters the render.
