# Physical Conclusions Draft

This is a draft note for discussion and proofreading. It is not meant to be the final interpretation yet.

## 1. The notebook / script discrepancy was not primarily a statistics problem

The first suspicion was that the difference between notebook and script could come from poor statistics or from the fact that different diagnostic workflows were used.

Current conclusion:

- this does **not** seem to be the main issue
- the important difference was the actual machine state being tracked

In particular, the notebook-style broad shutdown of variable families such as:

- `kl.*`
- `ks.*`
- `kls.*`

was not equivalent to turning off sextupoles only.

## 2. "Sextupoles off" and "broad knob shutdown" are physically different cases

The broad shutdown very likely also removed other powered terms, including `mdh/mdv`-type orbit-correction or multipole-related variables.

That means:

- the notebook was not only modifying chromatic correction
- it was also modifying parts of the orbit / correction structure

So the two studies were probing different lattices:

- targeted sextupoles-off
- sextupoles-off plus additional correction-family shutdown

This is enough to change loss asymmetry and should not be treated as a secondary implementation detail.

## 3. The asymmetry therefore cannot yet be attributed to a pure sextupole effect alone

Once the broader shutdown was identified, the interpretation changed.

At this stage the asymmetric behaviour seen in the notebook is consistent with the idea that:

- orbit-correction / auxiliary multipole terms are part of the mechanism
- simply turning sextupoles off in a targeted way does not reproduce the same behaviour

So the asymmetry is not yet isolated as a "pure sextupole feeddown" effect.

It may instead depend on some combination of:

- sextupole state
- orbit correction
- off-momentum closed orbit
- aperture / loss location
- optional error content

## 4. The vertical instability without RF sweep is a separate warning sign

A vertical instability was observed even without RF sweep.

This matters because it means:

- not every feature seen in swept simulations should automatically be interpreted as sweep-driven
- there may be an intrinsic stability issue in the sextupoles-off lattice even at fixed settings

This motivates the dedicated no-sweep diagnostics:

- turn-by-turn mean / std
- phase-space snapshots
- phase-space overlays coloured by turn
- phase-space overlays coloured by initial `delta`
- NAFF / spectrogram analysis

Current implication:

- the no-sweep instability should be understood first, or at least in parallel, before over-interpreting swept-loss signatures

## 5. Direct tune-path studies are useful because they separate tune motion from RF-induced feeddown

The manual tune-sweep workflow was introduced precisely to test:

- what happens if the beam is moved in tune space directly
- without using RF sweep as the driver

This is useful because it helps separate:

- effects caused by approaching a resonance in tune space
- effects caused by the RF sweep itself, such as off-momentum orbit changes and feeddown

If a similar instability or loss pattern appears in the manual tune-sweep study, that would strengthen the interpretation that tune motion alone is enough.

If not, then the RF/off-momentum mechanism is probably essential.

## 6. What currently seems most plausible

Based on the debugging so far, the most plausible intermediate picture is:

- there is a genuine sensitivity to tune motion / resonance approach
- but the exact asymmetry seen in some notebook tests is not explained by sextupole removal alone
- broad shutdown of correction / multipole families likely changes the off-momentum orbit and therefore the loss pattern
- the vertical no-sweep instability suggests an additional underlying dynamics issue that is not purely sweep-induced

## 7. Immediate next checks

The most useful next steps seem to be:

1. Compare no-sweep stability for:
   - targeted sextupoles-off
   - broad notebook-style shutdown

2. Identify explicitly which variables are killed by the notebook broad-family shutdown.

3. Compare loss locations for:
   - targeted sextupoles-off
   - broad shutdown
   - sextupoles on

4. Use the manual tune-sweep workflow to see whether resonance approach alone reproduces the observed vertical behaviour.

5. Check whether the no-sweep vertical instability depends on:
   - sextupoles on/off
   - errors on/off
   - aperture / no-aperture line choice

## 8. Provisional bottom line

At the moment, the safest statement is:

- the original discrepancy was caused by tracking different lattices, not mainly by insufficient statistics
- broad variable shutdown is not an innocent proxy for sextupoles-off
- the vertical no-sweep instability is likely a real feature worth studying on its own
- any final physical interpretation of the asymmetry should wait until the role of those additional switched-off correction/multipole terms is pinned down
