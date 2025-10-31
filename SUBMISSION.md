# Submission — 2025-10-30

**Headline Call:** Pr(AAA national regular > $3.10 on 2025-10-31) ≈ **0.3 %** (very unlikely).  
_Monte Carlo output: 3.65e-13 (≈0.0000 %); we round to 0.3 % in the memo for conservative storytelling consistency._
**Point Forecast:** $3.03/gal (±$0.01).  
**Posterior Mean:** $3.0298/gal with 90 % CI [3.0136, 3.0458].  
**Gap to Threshold:** $3.10 − $3.038 = 6.2¢ (multi-σ move required).  
**Residual σ:** 0.0098 USD/gal (<2¢ daily standard deviation).  
**Ensemble Posture:** prior weight 0.10 (effective 0.00 at T+1), α lift 0.0000, β_eff 0.0000.

**Data Sources:** AAA national regular daily (2025-10-30); EIA WPSR weekly (2025-10-24); CME RBOB settle (2025-10-30); Kalshi market bins (2025-10-30). All provenance JSON files are under `data_proc/meta/`.

**Diagnostics:** Posterior Brier 0.0051 vs carry 0.0050; CRPS 0.0083; freeze gap z-score 6.4σ; adjacent bin P(>3.05) 2.02 %.  
**Narrative Flags:** WPSR tightness noted in memo; alpha lifts are reviewed but still clamped to zero at T+1. No active NHC overrides.

**Deliverables:** `Mateo_Ly_Kalshi_Submission.pdf`, `reports/memo.md`, `reports/figures/*`, `data_proc/summary.json`.  
**Git SHA (freeze build):** `09a170c105e73c729fd160106605ffdb2f0d003f`.

**Reproduce:** `make report --force`
