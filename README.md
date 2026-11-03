# quant-lab / Numerical methods under review

Live: https://aditya2375.github.io/quant-lab/

Educational Python implementations of Monte Carlo risk, pairs trading and moving-average crossover. Not investment advice. Historical metrics and charts are withdrawn pending corrected market-data reruns. Old JSON/output artifacts remain for provenance, not current or validated performance.

## Corrections

Monte Carlo estimates annual log-return means, so simulation now uses that mean directly instead of subtracting half the variance a second time. Covariance is checked positive semidefinite and singular matrices supported. Simulation dimensions and total allocation are bounded.

Pairs now tests and fits log prices, trades the matching log spread, separates formation/trading dates, rejects failed screens, normalizes illustrative leg returns by gross exposure and charges on the lagged executed position changes. It is not necessarily dollar-neutral and still omits borrow, financing and realistic execution costs. Historical pair p-values and performance must not be reused.

Performance drawdown now includes the initial unit-capital peak. Invalid return paths/prices and insufficient histories are rejected. Price caches are keyed by ticker plus requested period with validated symbols; old filename caches are retained but ignored. Prices must be finite/positive with sorted unique dates. Yahoo downloads have a bounded timeout.

## Reproduction status

Original adjusted-price snapshots and package versions were not archived. A fixed random seed does not reproduce changing market data or provider revisions. Corrected code passes deterministic synthetic tests, not a rerun against the original market sample. Before publishing results, rerun all methods, archive exact inputs with dates/hashes and dependency versions, and inspect generated charts and metrics. Parameter changes/selection bias remain meaningful risks.

Install requirements.txt and run each Python script as documented in its folder. No paid data or brokerage action. Tests: python -m unittest discover -s tests.

Support: adityakulkarnicoding@gmail.com
