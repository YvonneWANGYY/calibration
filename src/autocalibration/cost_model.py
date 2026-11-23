"""Predict the upper (80th percentile) log-cost ratio for candidate inputs."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .lhs import to_unit


def coordinates(units, parameters):
    # Trees see log physical inputs on positive domains, linear inputs otherwise.
    # Affine rescaling to the unit interval leaves split order unchanged.
    values = np.array(units, dtype=np.float64, copy=True)
    for j, p in enumerate(parameters):
        if p.lower > 0 and p.scale == "linear":
            log_span = np.log(p.upper) - np.log(p.lower)
            if log_span > 0:
                physical = p.lower + values[:, j] * (p.upper - p.lower)
                values[:, j] = (np.log(physical) - np.log(p.lower)) / log_span
    return values


def predict_log_ratios(session, candidate_units, metric, seed):
    rows = session.observations
    costs = np.array([row.metrics[metric] for row in rows])
    reference = float(costs[np.argmin(session.losses())])
    train = [[to_unit(p, obs.parameters[p.name]) for p in session.parameters] for obs in rows]
    forest = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1)
    forest.fit(coordinates(train, session.parameters), np.log(costs) - np.log(reference))
    candidates = coordinates(candidate_units, session.parameters)
    prediction = np.empty(len(candidates))
    for start in range(0, len(candidates), 2048):
        chunk = candidates[start:start + 2048]
        predictions = [tree.predict(chunk) for tree in forest.estimators_]
        prediction[start:start + len(chunk)] = np.quantile(predictions, 0.8, axis=0)
    return prediction, reference
