# Multi-Fidelity Greedy Coordinate Search for MOT Parameter Optimization

## High-level idea

The goal is to design a cheap, greedy-like, single-objective optimization algorithm for tuning multi-object tracking parameters.

The algorithm is similar to coordinate descent: instead of optimizing all parameters jointly, it optimizes one parameter at a time while keeping the remaining parameters fixed.

The proposed name is:

**Multi-Fidelity Greedy Coordinate Search**

This name captures the main ideas:

- **Multi-fidelity**: candidate values are first tested cheaply on a small subset of scenes, then validated on the full dataset.
- **Greedy**: only changes that improve the objective are accepted.
- **Coordinate search**: one parameter is optimized at a time.

---

## Search space

The search space is constrained.

For numeric parameters, each parameter has a bounded interval:

\[
[A, B]
\]

For discrete parameters, each parameter has a finite set of possible values:

\[
V = \{V_1, V_2, \dots, V_n\}
\]

The objective is a single MOT validation score, for example:

- HOTA
- IDF1
- MOTA
- a weighted combination of MOT metrics

The algorithm can be written as maximization of a score or minimization of a loss. For MOT tuning, maximization is usually more natural.

---

## Main assumptions

The algorithm relies on the following assumptions:

1. Many tracker parameters have useful local structure when optimized one at a time.

2. For a fixed configuration of all other parameters, the objective is often approximately quasi-unimodal with respect to a single parameter.

3. The global MOT objective is not assumed to be truly unimodal.

4. MOT metrics are noisy, non-smooth, and often piecewise constant.

5. Parameters can interact strongly, so coordinate-wise optimization may miss improvements that require changing multiple parameters jointly.

6. A small sample of scenes can provide a useful but noisy estimate of whether a candidate parameter value is promising.

7. Full-dataset validation is required before accepting a candidate update.

8. Random coordinate search is a useful baseline because it is simple and less biased than a structured search strategy.

---

## Components

### SceneSampler

`SceneSampler` selects a small subset of scenes from the full dataset `D`.

The baseline sampler can be:

- `RandomSceneSampler`

For example, it may randomly select `M` scenes from the dataset.

A stronger future variant could be:

- `StratifiedSceneSampler`

This could sample scenes from different difficulty or domain groups, such as:

- crowded scenes
- sparse scenes
- long sequences
- short sequences
- camera-motion scenes
- static-camera scenes
- high-error scenes
- different dataset domains

The initial implementation can use random scene sampling.

---

### CoordinateOptimizer

The coordinate-level optimizer is called:

`CoordinateOptimizer`

It optimizes one parameter while keeping all other parameters fixed.

Possible implementations:

- `TernaryCoordinateOptimizer`
- `GridCoordinateOptimizer`
- `RandomCoordinateOptimizer`
- `GridCoordinateOptimizer`

The main version can use ternary search, while coarse-to-fine grid search and random coordinate search can be used as alternatives or baselines.

---

## Main algorithm

At each outer iteration, the algorithm performs a sweep over the parameters.

For each parameter:

1. Sample a small set of scenes using `SceneSampler`.

2. Run `CoordinateOptimizer` on this parameter using only the sampled scenes.

3. Inside `CoordinateOptimizer`, perform `M` optimization steps on the sampled scenes.

4. The optimizer returns the best candidate value for the selected parameter.

5. Evaluate the candidate configuration on the full dataset.

6. If the candidate improves the full-dataset score, accept the update.

7. If the candidate does not improve the full-dataset score, reject the update.

8. Continue with the next parameter.

The algorithm has three stopping criteria, whichever fires first: (a) a full sweep over all parameters produces no accepted move (greedy-coordinate convergence; can be disabled via `early_stop=False`); (b) `max_sweeps` reached; (c) `max_trials` full-fidelity evaluations completed (bootstrap counts as 1). `max_trials` is a hard cap useful when the search space is large enough that `max_sweeps` × `n_params` would exceed the desired budget.

---

## Evaluation flow

The algorithm uses two evaluation fidelities:

1. **Low-fidelity evaluation** on sampled scenes.
2. **High-fidelity evaluation** on the full dataset.

The evaluation flow is:

sampled scenes -> full dataset

There is no middle-stage dataset evaluation.

This keeps the algorithm simpler and cheaper.

---

## Ternary coordinate optimization

For numeric parameters, `TernaryCoordinateOptimizer` can be used.

For a selected parameter, the optimizer performs `M` ternary-search steps on the sampled scenes.

At each ternary-search step:

1. Choose two candidate values inside the current interval.
2. Evaluate both candidates on the same sampled scenes.
3. Keep the better part of the interval.
4. Repeat for `M` steps.
5. Return the best candidate value found on the sampled scenes.

The final candidate is then evaluated once on the full dataset.

This is cheaper than evaluating every ternary-search step on the full dataset.

Important detail:

> Candidate values compared inside the coordinate optimizer should be evaluated on the same sampled scenes.

This reduces noise from scene sampling and makes the comparison more reliable.

---

## Coarse-to-fine coordinate optimization

A more robust alternative to ternary search is coarse-to-fine grid search.

For a selected parameter:

1. Evaluate several grid points in the current interval using the sampled scenes.
2. Select the best value.
3. Shrink the interval around the best value.
4. Repeat for a fixed number of rounds or until the improvement becomes small.
5. Return the best candidate found on the sampled scenes.
6. Validate the candidate on the full dataset.

Example:

Initial interval: [0.1, 0.9]

Evaluate:

- 0.1
- 0.3
- 0.5
- 0.7
- 0.9

If the best value is 0.5, shrink the interval around it:

New interval: [0.3, 0.7]

Then repeat.

This may be more robust for MOT than pure ternary search because MOT metrics can be noisy and piecewise constant.

---

## Random coordinate optimization baseline

`RandomCoordinateOptimizer` is a useful baseline.

For a selected parameter:

1. Randomly sample `M` candidate values from its search space.
2. Evaluate them on the sampled scenes.
3. Select the best candidate.
4. Validate the candidate on the full dataset.
5. Accept the candidate only if it improves the full-dataset score.

This is a good baseline because it is simple and less biased than ternary search or grid search.

It helps test whether the structured greedy coordinate strategy actually improves over random exploration.

---

## Search-space constraint / shrinking

After accepting a new value for a parameter, the search space can optionally be narrowed around the accepted value.

For a numeric parameter:

old interval: [A, B]

accepted value: v

new interval:

\[
[v - r, v + r]
\]

clipped to the original valid range.

For ordered discrete parameters, the new search space can keep values near the accepted value.

For unordered categorical parameters, shrinking should be used carefully or avoided, because there is no natural notion of distance between values.

---

## Computational cost

Let:

- `P` be the number of parameters
- `M` be the number of coordinate-optimization steps per parameter
- `S` be the number of sampled scenes
- `|D|` be the number of scenes in the full dataset
- `C_full` be the cost of evaluating the full dataset
- `A` be the number of candidates evaluated on the full dataset

The approximate cost of one sweep is:

\[
P \cdot M \cdot \frac{S}{|D|} \cdot C_{\text{full}} + A \cdot C_{\text{full}}
\]

The method is efficient when:

- `S << |D|`
- `M` is small
- `A` is much smaller than `P * M`

Compared with TPE or full random search, the method can be much cheaper because most candidate evaluations happen only on a small scene sample.

---

## Benefits

- Cheap to iterate over parameters.
- Easy to implement.
- Easy to interpret.
- Works naturally with bounded tracker hyperparameters.
- Uses full-dataset validation before accepting updates.
- Avoids evaluating every candidate on the full dataset.
- Random coordinate optimization provides a fair unbiased baseline.
- The `CoordinateOptimizer` interface allows multiple search strategies to be compared.

---

## Limitations

- The MOT objective is not guaranteed to be unimodal.
- Ternary search can fail if the objective is noisy or multi-modal along a parameter.
- Parameters can interact strongly.
- Scene subsampling introduces noise.
- A candidate may improve sampled scenes but fail on the full dataset.
- Greedy coordinate-wise updates may miss improvements that require changing multiple parameters together.
- Search-space shrinking can remove useful values if applied too aggressively.

---

## Recommended initial implementation

Use the following initial components:

Algorithm name:

**Multi-Fidelity Greedy Coordinate Search**

Scene sampler:

- `RandomSceneSampler`

Coordinate optimizer variants:

- `TernaryCoordinateOptimizer`
- `GridCoordinateOptimizer`
- `RandomCoordinateOptimizer`

Validation:

sampled scenes -> full dataset

Stopping rule:

> Stop after a full sweep with no accepted improvement, when `max_sweeps` is reached, or when `max_trials` full-fidelity evals have been spent — whichever fires first.

The most robust first version is probably:

**RandomSceneSampler + GridCoordinateOptimizer + full-dataset acceptance gate**

The random coordinate optimizer should be included as a baseline because it provides a simple, less biased comparison.
