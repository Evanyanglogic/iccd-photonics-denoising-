# Theoretical Gate Derivation

For each 512 x 512 Gaussian residual, `SE(mean) = sigma / sqrt(262144) = sigma / 512` and `z_mean = residual_mean / SE(mean)`.

Across 2151 preregistered pair realizations, the approximate two-sided family probabilities under independent standard-normal means are:

- at least one `|z| > 4.0`: 0.127379
- at least one `|z| > 4.5`: 0.014511
- at least one `|z| > 5.0`: 0.001232

The frozen pair gate is `|z_mean| <= 4.5`, with `|residual_mean| < 2 DN` as an implementation safety bound. It is combined with experiment/seed, experiment/condition/seed, all-pair, and clipping gates. It does not center or resample residuals.
