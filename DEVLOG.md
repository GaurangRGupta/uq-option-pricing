# Dev Log

Running, chronological session log for the `ivuq` project. Different purpose
from `claude-context.md` (which is a point-in-time snapshot of "what's built
and decided") — this file is the day-by-day record of what happened in each
work session, kept so any of the three of us (or a fresh assistant session)
can pick up context fast without re-reading everything.

Newest entry on top.

---

## 2026-09-20 — ML/DL baselines and the PINN comparison harness (American put first)

Motivation (user request): show why a PINN over ordinary ML/DL pricers, with
American options as the main case and implied vol included. The paper draft
(LaTeX, `paper/`, gitignored) and its title were also finished earlier this
stretch: "Physics-Informed Neural Networks for Distribution-Free Uncertainty
Quantification in American Option Pricing". It states plainly that conformal
UQ and real-data validation are not yet done.

### What was built (`src/ivuq/baselines/`, `scripts/`, `tests/test_baselines.py`)

- 10 baselines behind one `fit(X, u)` / `predict(X)` interface, inputs
  (m = S/K, tau), target u = V/K (same variables as the PINN).
  Classical ML: polynomial ridge, kernel ridge (CV-tuned), random forest,
  gradient boosting. Deep learning: tanh MLP (same 4x64 shape as the PINN),
  ReLU MLP, residual SiLU net, Fourier-feature net, partially input-convex
  net (convex in m by construction, Amos et al. 2017), and a BS-embedded
  implied-vol net (learns sigma(m, tau), price = Black-Scholes(sigma)).
  The last two were added after the user objected that a plain MLP is too
  weak a comparator.
- Labels: Black-Scholes closed form (European), 500-step CRR tree (American),
  optional Gaussian label noise.
- Scoring (`metrics.py`), all from one shared finite-difference treatment:
  price error, delta/gamma error, BS PDE residual (European), no-arbitrage
  violation rates (convexity, delta range, below-intrinsic, calendar), and
  for the American put the exercise-boundary error and the implied-vol error
  (price inverted through the 200-step tree, compared with the vol recovered
  from the reference price).
- Regimes (`experiments.py`): labels everywhere, labels only near the money
  and short maturity (extrapolation), noisy labels ($0.50). Label counts
  50/200/1000/3000, 2 seeds. PINNs get no labels, so appear as flat lines.
- Runner `scripts/run_baseline_comparison.py` (American first; American call
  skipped because with q = 0 it equals the European call), figure script
  `scripts/make_comparison_figures.py` (fig1-fig10).
- 16 tests pass. They check the label generators, that the metrics score the
  exact Black-Scholes price as perfect, that a broken price is flagged, that
  the input-convex net is convex even untrained, the exercise-boundary
  extraction, and American IV recovery.

### Results (American put, reporting-config PINN, in dollars on a $100 strike)

The honest reading: the PINN is NOT more accurate than models trained on
labels in this setup.
- Price error: PINN 2.31; kernel ridge 0.005, BS-embedded net 0.009, tanh MLP
  0.23 at 3000 labels. Nine of ten baselines beat the PINN.
- Exercise boundary error is 21 for the PINN vs 0.2 to 2.6 for baselines, and
  IV error about 8 vol points. Part of this is the boundary metric (price
  within $0.20 of intrinsic), which a price sitting ~$2 high never satisfies.
- Where the PINN looks reasonable: it needs zero labels; zero calendar
  violations; outside the labelled region its error (~$2.5) is far below
  kernel ridge ($28) and polynomial ridge ($164), though similar to the
  neural nets and well above the BS-embedded net ($0.11).
- It still breaks convexity on ~29% of test points, like most baselines.

Consequence: do not claim superiority in the paper. Defensible claims are no
labels needed, no calendar arbitrage, no extrapolation collapse of
kernel/polynomial models, and the PDE residual as the signal for conformal UQ.

### Known limits / next steps

- The PINN is likely undertrained (fixed weighting, 6000 epochs); improving
  it and rerunning is the recommended next step.
- American under Heston, the project's centerpiece, is NOT in this comparison
  yet: it needs 3-input baselines, slow LSMC labels and a reporting-scale
  N2-Heston run.
- IV is measured at one fixed vol (0.20). A real IV-surface comparison needs
  a PINN conditioned on vol.
- A first version of the figure script let `$` in axis labels render as math;
  fixed by escaping, figures regenerated.
- `results/` holds the CSVs and PNGs; `results_run.log` and `figs.log` are
  scratch logs and should not be committed.

---

## 2026-09-10 — Phase 3b step 2: N2-Heston, American put under Heston (the centerpiece), plus cleanup

Built the actual centerpiece of Phase 3b: the coupled solution+boundary
network for the American put under Heston, per the architectural call made
at the end of the 2026-08-30 session (dual network, own math/code, not an
extension of the GBM LCP). Also cleared the two long-outstanding loose
scratch files.

- Deleted `_scratch_nb1.ipynb` and `paper4_extracted.txt` outright (both
  untracked, so no git history impact). The notebook was leftover
  exploration code that directly imported a reference repo's own modules
  (`american_utils`, `model_types_am`) and reused its exact class names --
  fine as throwaway scratch work at the time, but the kind of thing that
  should never sit in the repo long-term given the project's own
  cite-ideas-never-copy-code rule, so removed rather than kept around. The
  text file was a stale plaintext dump of one of the 4 reviewed papers,
  already properly indexed (PDF + summary) in `planning/papers/`.

- `src/ivuq/pricing/heston_lsmc.py`: Longstaff-Schwartz Monte Carlo
  (Longstaff & Schwartz, 2001) American-Heston reference pricer -- the
  validation instrument N2-Heston needs, since no closed form exists for
  American options and Heston has no fast lattice method the way GBM has
  CRR. Full-truncation Euler for the variance path (Lord, Koekkoek & Van
  Dijk, 2010, so the CIR process can't go negative between steps), OLS
  regression on a small polynomial basis in (moneyness, variance) at each
  backward step. Classical numerical-finance machinery, unrelated to and
  predating every paper in `planning/papers/PAPER_TRAIL.md` -- same
  "independent reference instrument" role `heston.py` and `binomial.py`
  play elsewhere. `tests/test_heston_lsmc.py`: 6 tests, matches CRR when
  vol-of-vol -> 0, American >= European under Heston (both option types),
  deep-ITM early-exercise premium is positive and nontrivial, q=0 call
  stays close to European. All passing.

- `src/ivuq/pinn/heston_free_boundary.py`: the coupled solution+boundary
  network. Classical value-matching (u(b,v,tau)=h(b)) and smooth-pasting
  (du/dm(b,v,tau)=h'(b)) conditions (McKean 1965; Wilmott/Howison/Dewynne
  ch. 8) at a learned boundary surface b_phi(v,tau) (Heston has variance as
  a second state variable, so the free boundary is a surface, not a curve
  the way it is under GBM) -- `BoundaryNet` wraps a plain `PINN(in_dim=2)`
  with a sigmoid so its output always lies in (0,1), exploiting the
  classical fact that an American put's boundary always sits below the
  strike. Scoped to puts only: an American call with q=0 (this project's
  own default) is never exercised early, which sends that boundary to
  infinity -- unlearnable by a bounded network, and pointless anyway since
  American = European there. Paper 3 in `PAPER_TRAIL.md` makes the
  identical scoping choice for the identical reason, arrived at
  independently here, not copied from it.
  - Found during development, not before: with only the two classical
    conditions plus the terminal/far-field anchors, the trained solution
    network sagged well below intrinsic value in the mid-domain (several
    dollars negative on a $100 strike, well past any reasonable soft-
    training slack) while the PDE fit was still converging -- nothing in
    the dual-network formulation stops that the way the LCP's own obstacle
    term does for GBM/N2, since the continuation-region network has no
    obstacle term at all by construction. Added `obstacle_residual`
    (u >= h(m), restricted to the continuation-region collocation points)
    as an eighth loss term -- not one of the two classical free-boundary
    conditions, but a direct consequence of them (continuation region is
    defined as u > h(m)) and cheap to enforce; weighted at 5.0 (well above
    every other term) once weighting it at parity with the rest turned out
    not to be enough to stop the sag. Documented in both the module
    docstring and `AmericanHestonPINNConfig`'s own comments so the reason
    for that unusually high weight isn't lost.
- `src/ivuq/pinn/config.py`: added `AmericanHestonPINNConfig` (put-only, no
  `option_type` field, unlike every other config in this file).
- `src/ivuq/pinn/train_american_heston.py`: one training loop, eight loss
  terms (PDE + obstacle + value-matching + smooth-pasting + two terminal
  conditions + two far-field conditions), one shared optimizer over both
  networks so their coupling flows through ordinary gradient descent.
- `src/ivuq/pinn/american_heston.py`: `AmericanHestonPINN` wrapper --
  `.price()` is piecewise (intrinsic value below the learned boundary,
  solution network above it), `.boundary()` exposes the learned exercise
  boundary directly in real (S, tau) units, `.pde_residual()` mirrors the
  other wrappers.
- `src/ivuq/pinn/__init__.py`: exports `AmericanHestonPINNConfig`,
  `AmericanHestonPINN`.
- `tests/test_pinn_american_heston.py`: 6 tests, fast config (3 layers/32
  wide solution net, 2 layers/16 wide boundary net, 1500 epochs). Recovers
  LSMC within a loose tolerance; obeys the obstacle condition on average
  (same convention as N2/GBM's own obstacle test, not an exact per-point
  guarantee); boundary stays in (0,1); American >= European Heston; direct
  checks on the value-matching/smooth-pasting residuals themselves and on
  `BoundaryNet`'s sigmoid output range (untrained, math/code-consistency
  only). All passing.
- Full suite: 95 tests passing (83 + 6 LSMC + 6 N2-Heston).
- Not started: Track 2 (conformal/UQ harness), still deferred per user
  instruction. This closes out the base-PINN model ladder (N0/N1 GBM, N2
  GBM, N0/N1 Heston, N2 Heston) -- Section 8's Checkpoint B.

### Pending commits for this session's work

Items 22-28 in `planning/COMMIT_COMMANDS.md`. The two deleted scratch files
need no commit (both were untracked already).

## 2026-08-30 (cont'd) — Phase 3b step 1: N0/N1-Heston, European under Heston

Before attempting N2-Heston (American under Heston, the dual-network
free-boundary problem), went and read the actual source of three of the
four reference repos (forked research, not summarized here) to check we
weren't missing a standard trick or misreading a paper. Findings: our
self-adaptive weighting scheme (`train_american.py`) is faithful in
mechanism to `levimcclenny/SA-PINNs`; the Coupled-PINN Heston repo
(`Rohan-217/...`) uses a genuinely dual solution+boundary network with
classical value-matching/smooth-pasting, not an LCP penalty like ours —
legitimate, different route, but it means our LCP approach never produces
an explicit boundary function, which Phase 5 (exercise-boundary conformal
band) will need; Dhiman & Hu's public American PINN has a real bug (their
inequality-violation loss is computed but never added to the backpropagated
loss, and the dead code that would've added it can't ever fire) — our
`free_boundary.py` is strictly more faithful to the LCP than that published
reference. No copying risk anywhere; all three of our implementations are
structurally independent of the repos they were compared against.

Decision from this: build N2-Heston as a dual solution+boundary network
(own math, own code, independently derived), not an extension of the
single-network LCP approach. Leave N2-GBM exactly as-is — already tested,
and Phase 3 was always scoped as commodity reproduction, not the novel
contribution.

Before touching the dual-network American case at all, built the
validation instrument and the simpler stepping-stone first — same
discipline as Phase 2 (validate against a known closed form before adding
free-boundary complexity):

- `src/ivuq/pricing/heston.py`: Heston (1993) semi-closed-form European
  price via characteristic-function inversion, using the "Little Trap"
  sign convention (Albrecher et al. 2007) for numerical stability. No
  overlap with any of the 4 reviewed papers (none touch Heston's own
  closed form) — this is classical numerical-finance machinery, the same
  role `black_scholes.py` plays for N1 and `binomial.py` plays for N2.
  `tests/test_heston.py`: recovers Black-Scholes within a cent as
  vol-of-vol -> 0, respects no-arbitrage bounds, put-call parity holds
  exactly, price increases with initial variance. All 7 passing.
- `src/ivuq/pinn/heston_pde.py`: non-dimensionalized Heston PDE (same
  m=S/K, u=V/K convention as `black_scholes_pde.py`, v left alone since
  it's already dimensionless), derived from Heston's SDE pair via the usual
  Feynman-Kac argument. Boundary conditions: m=0/m=m_max reused directly
  from the BS case (unchanged by v); v=0 is a natural boundary (the PDE
  degenerates on its own, no separate condition needed or well-posed);
  v=v_max uses the standard artificial d2u/dv2=0 condition from Heston
  finite-difference literature (in't Hout & Foulon).
- `src/ivuq/pinn/network.py`: generalized `PINN` to take an `in_dim`
  parameter (2 for GBM's (m,tau), 3 for Heston's (m,v,tau)) instead of
  hardcoding 2 — one-line change, no behavior change for existing callers.
- `src/ivuq/pinn/heston_collocation.py`, `train_european_heston.py`,
  `european_heston.py`: same shape as the GBM European PINN's
  collocation/train/wrapper trio, extended to the extra v dimension and
  the extra v-boundary loss term.
- `src/ivuq/pinn/config.py`: added `HestonEuropeanPINNConfig`
  (`lambda_pde` is the N0/N1 switch here too).
- `tests/test_pinn_european_heston.py`: 5 tests. One is a pure math/code-
  consistency check with no training involved at all — confirms
  `heston_pde.pde_residual` collapses exactly onto `black_scholes_pde.
  pde_residual` (same network, same inputs) when xi=0 and v is held fixed
  at sigma^2, i.e. equation (4) really does reduce to equation (2) in the
  code, not just on paper. The other four mirror N1's tests: N1-Heston
  recovers the closed-form `heston_price` (loose tolerance, fast config),
  and N1-Heston beats N0-Heston (the PDE-loss ablation). All passing.
- Full suite: 83 tests passing (71 + 7 Heston-pricer + 5 Heston-PINN).
- Not started: N2-Heston (the dual-network American free-boundary PINN
  under Heston — the actual Phase 3b centerpiece), and Track 2, still
  deferred per user instruction.

### Pending commits for this segment (Phase 3b step 1)

Items 14-21 in `planning/COMMIT_COMMANDS.md`. Also revised items 9 and 12
from the Phase 3 batch below, since `config.py` and `__init__.py` each got
touched again by this segment's work before Phase 3's commits were made —
see the notes on those items for why the `git add` lines changed.

## 2026-08-30 — Phase 3 built: N2, the American free-boundary PINN under GBM

- `src/ivuq/pinn/free_boundary.py`: the American option as a linear
  complementarity problem (LCP). Reuses the exact same Black-Scholes
  operator L[u] from `black_scholes_pde.py` (no re-derivation, no
  duplicated autograd machinery) via `pde_residual`. The LCP:
  `L[u] >= 0, u >= h(m), L[u]*(u-h(m)) = 0`, soft-penalized as three
  per-point residuals (`ineq_violation`, `obstacle_violation`,
  `complementarity_violation`) that `train_american.py` squares and
  weights. Terminal condition and both spatial boundary conditions are
  unchanged from the European case (American and European agree at expiry
  and at the domain edges) — reused directly, not redefined.
- `src/ivuq/pinn/config.py`: added `AmericanPINNConfig`, same shape as
  `EuropeanPINNConfig` plus `weighting_scheme: "fixed" | "curriculum" |
  "self_adaptive"` — the Phase 3 ablation from Section 5/6 of the roadmap
  (self-adaptive weighting folded in as a training-config option, not a
  standalone phase).
- `src/ivuq/pinn/collocation.py`: added `sample_interior_near_boundary` —
  rejection-samples interior points where the model's current prediction is
  close to the intrinsic value, i.e. near its current estimate of the
  exercise boundary. Falls back to uniform sampling early in training when
  nothing qualifies yet.
- `src/ivuq/pinn/train_american.py`: one training loop, three schemes:
  - `fixed` — constant LCP loss weights, fresh uniform interior sampling
    every epoch. The baseline.
  - `curriculum` — the three LCP weights ramp linearly from 10% to 100% of
    target over `curriculum_ramp_epochs`; once the ramp completes, interior
    sampling shifts to half uniform / half boundary-concentrated.
  - `self_adaptive` — a simplified, points-only version of McClenny &
    Braga-Neto (arXiv:2009.04544): interior points sampled once and held
    fixed for the run, each of the three LCP terms gets a per-point
    trainable weight (`softplus(a_i)`), updated by gradient ascent (flip
    the gradient sign on those params before the shared optimizer step).
    Deliberately scoped down from the paper's full per-point scheme applied
    everywhere — here it's only the free-boundary terms, since that's the
    one place spatial adaptivity actually matters (concentrating weight
    where the a-priori-unknown exercise boundary sits); terminal/boundary
    weights stay plain scalars, no free boundary to resolve there.
- `src/ivuq/pinn/american.py`: `AmericanPINN` wrapper, same interface as
  `EuropeanPINN` (`.fit()`, `.price()`, `.pde_residual()`).
- `tests/test_pinn_american.py`: fast config (3 layers, 32 wide, 1200
  epochs), 8 tests, all passing. No closed form exists for American options,
  so the reference is the CRR binomial tree (`ivuq.pricing.binomial.
  crr_price`) — comparing against a tree at all is new relative to the 4
  reviewed papers (none do it). Checks (a) all three weighting schemes
  converge to within a loose tolerance of CRR, for both calls and puts, and
  (b) the obstacle condition `u >= intrinsic` roughly holds after training.
  Full suite: 71 tests passing (63 existing + 8 new).
- At the fast/test config, mean abs error vs. CRR across all six
  (scheme, option_type) combinations ranged ~$3.26–$5.32 on a $100-strike
  option — meaningfully better than an untrained/broken model (10s of
  dollars off), but this is the fast config; no scheme was singled out as
  clearly best at this scale, and that's expected — the ablation is meant to
  show all three are viable, not to crown a winner yet. At the real config
  (4 layers, 64 wide, 6000 epochs, matching N1's reporting config,
  `fixed` weighting): mean abs error vs. CRR ≈ $2.31 (put) / $2.66 (call) on
  a $100-strike option — tighter than the fast-config numbers above, as
  expected, and in the same ballpark as N1's own real-config BS-recovery
  error ($0.21 put / $0.74 call), just a bit looser since the LCP is a
  harder target than plain BS. `curriculum`/`self_adaptive` real-config
  numbers not run yet (4 more full runs at this scale) — can be run on
  request if a head-to-head at full scale is wanted.
- Not started: extending N2 to Heston (Phase 3b), and Track 2
  (conformal/UQ harness) — both explicitly deferred this session per user
  instruction.

### Pending commits for this session's work (Phase 3, N2)

1. `feat(pinn): American free-boundary PDE as a linear complementarity problem` — `src/ivuq/pinn/free_boundary.py`
2. `feat(pinn): American PINN training configuration (weighting-scheme ablation)` — `src/ivuq/pinn/config.py`
3. `feat(pinn): boundary-concentrated collocation resampling` — `src/ivuq/pinn/collocation.py`
4. `feat(pinn): American PINN training loop (fixed/curriculum/self-adaptive weighting)` — `src/ivuq/pinn/train_american.py`
5. `feat(pinn): AmericanPINN wrapper and package exports` — `src/ivuq/pinn/american.py`, `src/ivuq/pinn/__init__.py`
6. `test(pinn): N2 vs. CRR binomial tree, all three weighting schemes` — `tests/test_pinn_american.py`

Commands to follow, same convention as `planning/COMMIT_COMMANDS.md` (git add/commit/push origin dev/push myfork dev per item) — to be added there once this DEVLOG entry is confirmed accurate.

## 2026-08-25 — Cleanup pass: deleted superseded planning docs and claude-context.md

- Deleted from `planning/` (all superseded per the 2026-08-24 "roadmap is the
  only authoritative doc" decision below; none were git-tracked since
  `planning/` is gitignored, so nothing lost on GitHub): `PROJECT_PLAN.md`,
  `ARCHITECTURE.md`, `BUILD_PLAN.md`, `ADDON.md`, `PROJECT_FLOW.md`,
  `flow.md`, `README-mine.md` (all last touched Aug 3-11, predate the
  roadmap), plus `2605.06688v1.txt` (redundant text extract of the PDF
  already sitting next to it) and `New Text Document.txt` (stray junk).
  `planning/` now only holds `RESEARCH_GAP_AND_ROADMAP.md`,
  `papers/PAPER_TRAIL.md` + the 4 source PDFs, and `COMMIT_COMMANDS.md`.
- Deleted `claude-context.md` from disk entirely (not just untracked —
  gone). It was dated 2026-08-13 (session 2) and had drifted into actively
  wrong territory: referenced the five planning docs just deleted above as
  live specs, claimed "no `.gitignore`, no git repo yet," and claimed
  PINN work was halted — all stale. `DEVLOG.md` fully supersedes it as the
  "pick up context fast" doc. Its debugging war-stories (BAW 35x bug, Yahoo
  dividend-yield bug, LSE field-mapping findings) aren't duplicated here,
  but the file is still recoverable from git history on both remotes
  (`origin`/`myfork`) if ever needed — it was tracked and pushed before
  commit `6d05d51` untracked it.
- `resume_extracted.txt` (the PII-flagged scratch file, unresolved across
  several prior sessions) turned out to already be gone — resolved, no
  action needed.
- Updated `.gitignore` to drop the now-dead `claude-context.md` line (file
  doesn't exist anymore, nothing to ignore) and `planning/COMMIT_COMMANDS.md`
  to drop commit 0 (already done as `6d05d51`, and its subject file is now
  deleted outright rather than just untracked).
- Still sitting on disk, not yet decided: `paper4_extracted.txt` (root,
  old scratch file) and the build-artifact caches (`__pycache__/`,
  `.pytest_cache/`, `src/ivuq.egg-info/`) — all gitignored already, safe to
  clear anytime, just not done yet.

## 2026-08-24 — Roadmap adopted as single source of truth; papers indexed

- Decided: `planning/RESEARCH_GAP_AND_ROADMAP.md` is now the **only**
  planning doc the team follows for direction. `PROJECT_PLAN.md`,
  `ARCHITECTURE.md`, `BUILD_PLAN.md`, `ADDON.md`, `PROJECT_FLOW.md`,
  `flow.md`, `README-mine.md` are left on disk (some early scoping ideas in
  them — e.g. the curriculum-learning / adaptive-resampling recipe for the
  American PINN — turned out to independently match published work, which is
  how the roadmap file came about) but are no longer authoritative.
- Added `planning/papers/PAPER_TRAIL.md` — quick-access index (arXiv link,
  local PDF path, GitHub repo) for the 4 papers the roadmap's gap analysis is
  based on. Full per-paper writeup stays in `RESEARCH_GAP_AND_ROADMAP.md`
  Section 2.
- Ground rule reaffirmed: the 3 papers with public repos
  (`pinn_option_pricing`, `PI-ConvTF`,
  `American_Options_Pricing_using_Coupled_PINNs`) are for reading and
  benchmarking only. All of our code is our own implementation — cite ideas,
  never copy code.
- Coding-style contract for this project going forward: plain, readable,
  entry-level-quant-appropriate code (no over-engineered abstractions, no
  defensive handling for things that can't happen) — production-quality in
  the sense of being tested and packaged correctly, not in the sense of being
  over-abstracted. Comments are fine on most lines, but every comment has to
  earn its place: meaningful, human-sounding, simple, says something the code
  itself doesn't already say. No filler restating what a line obviously does.
- Timeline decision: compress the roadmap's week-based phases into a fast
  10-15 working day build (see `RESEARCH_GAP_AND_ROADMAP.md` Section 8),
  instead of the original ~28-week pace. Paper-writing timing (concurrent
  with the build vs. after) is explicitly deferred — not decided yet.
- Added Section 8 to `RESEARCH_GAP_AND_ROADMAP.md`: the compressed day-by-day
  execution plan, with an explicit checkpoint for when the
  conformal/UQ teammate can start work in parallel with the base-PINN track
  (Day 4, once N1/European PINN is emitting point predictions + PDE
  residuals — doesn't need to wait for the American/Heston models), plus the
  git commit/push convention to use once coding starts.
- No code written yet — still pre-Phase-1. Coding starts when explicitly told
  to start.

### Phase 1 + Phase 2 built (same day)

- `paper/related_work.md`: Deep Optimal Stopping (Becker/Cheridito/Jentzen,
  arXiv:1804.05394), Neural Optimal Stopping Boundary (Reppen/Soner/
  Tissot-Daguette, arXiv:2205.04595), self-adaptive PINN (McClenny/
  Braga-Neto, arXiv:2009.04544), Merton (1976), Heston (1993), and the
  style-vs-dynamics citation (Gaudenzi/Stucchi/Spangaro, arXiv:1712.08137),
  plus the 4-paper recap (full version stays in
  `RESEARCH_GAP_AND_ROADMAP.md`).
- `src/ivuq/pinn/`: European PINN under GBM (N0/N1). PDE derivation
  (non-dimensionalized by moneyness m=S/K) lives as a docstring in
  `black_scholes_pde.py`, next to the code that implements each term of it.
  `EuropeanPINNConfig.lambda_pde` is the N0/N1 switch (0.0 = N0, no interior
  physics constraint; >0 = N1).
  - Tried and reverted: hard-enforcing the terminal condition via an ansatz
    (`u = payoff(m) + tau*correction(m,tau)`) so there'd be no terminal loss
    term to weight. Backfired — the payoff's kink at m=1 has to be exactly
    cancelled by `correction` for every tau, which a smooth tanh network
    can't do, and it showed up as a visible error band around m=1 at every
    maturity. Went back to the standard soft-constraint terminal loss
    (same as all 4 reference papers), and instead improved convergence with
    kink-concentrated terminal-point sampling + a learning-rate step decay.
    Left a note in `network.py` explaining why, so nobody reintroduces it
    without re-hitting the same wall.
  - Result at the "real" config (4 layers, 64 wide, 6000 epochs): mean
    abs price error over an interior (S, tau) grid ≈ $0.21 (put) / $0.74
    (call) on a $100-strike option. Not state-of-the-art, but this is
    explicitly commodity-reproduction work (Section 7 of the roadmap), not
    a claimed contribution.
  - `tests/test_pinn_european.py`: fast config (3 layers, 24 wide, 1500
    epochs, ~10s/model), 4 tests, all passing. Checks (a) N1 recovers
    closed-form BS within a loose tolerance, and (b) N1's interior error is
    meaningfully below N0's at identical settings — this is the actual
    ablation evidence that the PDE loss is doing something, not just a
    theoretical claim.
  - Full existing test suite (59 tests) still passes; `pyproject.toml` now
    depends on `torch>=2.4` (CPU build currently installed — machine has an
    unused RTX 3050, switching to a CUDA build deferred until it's actually
    needed, likely Phase 3b/Heston).
- **Checkpoint A reached**: N1 exists and exposes both `.price(S, K, tau)`
  and `.pde_residual(S, K, tau)`. Whoever's doing the conformal/UQ track can
  start now against this model, per Section 8 of the roadmap.

### Repo visibility / commit decisions (same day)

- `planning/` was already gitignored before this session (pre-existing,
  intentional — "kept locally, not pushed to GitHub").
- Decided (reversing the earlier call below): `paper/` is now gitignored too,
  same as `planning/`. Not pushed to GitHub at all for now.
- `DEVLOG.md` (this file) is going in `.gitignore` too, same reasoning as
  `planning/` — it narrates AI-assistant session work and references
  gitignored files, so it'd read oddly to an outside repo viewer. It stays
  as a local-only continuity file.
- `tests/` stays tracked and pushed as normal, always — it's part of the
  actual deliverable (already true for the pre-existing 59 tests), not
  process notes. Never gitignore tests.
- User feedback: don't propose 1-2 "major" lumped commits for a coding
  session's code — split even a single feature into several small commits
  by sub-component (math/PDE, network, config, training loop, wrapper),
  same granularity as docs/tests already get. Saved to memory
  (`feedback-commit-granularity`) so this is the default from now on.
- Revised: `DEVLOG.md` is **not** in `.gitignore` after all — instead it's
  just never `git add`ed, so it stays untracked and local without needing a
  gitignore rule. Same practical effect, simpler.
  Also added `paper4_extracted.txt` (old scratch file) and
  `claude-context.md` (frozen prior-session context file) to `.gitignore` —
  `resume_extracted.txt` (contains PII) still sitting on disk, untouched,
  still unaddressed.
- `claude-context.md` was previously tracked/pushed to GitHub — being
  removed from tracking (kept on disk, added to `.gitignore`) since it's the
  same kind of local session-continuity file as `DEVLOG.md`.

### The 7 commits for this session's work (Phase 1 + Phase 2), to make when ready

Revised from an original 9-commit plan: user decided `paper/` is gitignored
(not pushed to GitHub at all, file stays on disk) and merged the
torch-dependency commit into the PDE commit since they're small and land
together anyway.

1. `feat(pinn): non-dimensionalized Black-Scholes PDE and boundary conditions` — `pyproject.toml`, `src/ivuq/pinn/black_scholes_pde.py`
2. `feat(pinn): network architecture` — `src/ivuq/pinn/network.py`
3. `feat(pinn): collocation point sampling` — `src/ivuq/pinn/collocation.py`
4. `feat(pinn): training configuration (N0/N1 switch via lambda_pde)` — `src/ivuq/pinn/config.py`
5. `feat(pinn): Adam training loop` — `src/ivuq/pinn/train.py`
6. `feat(pinn): EuropeanPINN wrapper and package exports` — `src/ivuq/pinn/european.py`, `src/ivuq/pinn/__init__.py`
7. `test(pinn): N0-vs-N1 ablation and closed-form BS recovery check` — `tests/test_pinn_european.py`

Not part of any commit: `DEVLOG.md` (pending the open question above),
`paper/` (now gitignored, per above), anything under `planning/` (already
gitignored), `paper4_extracted.txt` and `resume_extracted.txt` (old scratch
files, unrelated to this work, still sitting on disk — never asked to
delete them, so still there).

Note: session ended 2026-08-24, this is being read back on 2026-08-25 or
later (user is restarting their PC) — if these commits get made today
instead of the 24th, that's fine and actually helps the "genuine multi-day
pacing" goal in `project-commit-pacing` memory, not against it.
- Standing principle for the whole build, refined: the ideal is still
  **math first, then code** — derive the equation, then implement it. That
  stays the default. But it's not a hard requirement every single time;
  occasionally an architecture/loss choice gets tried empirically first and
  gets formalized into an equation afterward, and that's acceptable too.
  What's non-negotiable is that **the two always match** — no code sitting
  in the repo whose corresponding equation isn't written down somewhere
  (docstring/paper), and no equation in the paper that the code doesn't
  actually implement. Every loss term, every architecture choice, every
  nonconformity score has to trace back to an equation we can point to and
  justify (the PDE, the free-boundary conditions, the conformal guarantee
  proof) — not "this is what worked empirically" with no derivation ever
  written down. This matters for two reasons: (1) it's what makes the code
  reviewable by someone who knows the math rather than a black box, and
  (2) the paper's math section and the code have to describe the same
  model, regardless of which was written first.
