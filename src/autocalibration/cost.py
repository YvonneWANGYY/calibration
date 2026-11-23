"""Computational cost metadata for the optional flow acquisition penalty."""

import math


def cost_settings(problem, session):
    if "cost" not in problem:
        return None
    cost = problem["cost"]
    if not isinstance(cost, dict) or set(cost) != {"metric", "beta"}:
        raise ValueError("cost requires metric and beta")
    if problem.get("method") != "neural_spline_flow":
        raise ValueError("cost requires method neural_spline_flow")
    metric = cost["metric"]
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("cost.metric must be a nonempty output name")
    if metric in session.target_profile or metric in {p.name for p in session.parameters}:
        raise ValueError("cost.metric must be separate from target and parameter names")
    beta = cost["beta"]
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0:
        raise ValueError("cost.beta must be a finite nonnegative number")
    return cost


def check_cost(metrics, cost):
    if cost and (cost["metric"] not in metrics or metrics[cost["metric"]] <= 0):
        raise ValueError(f"expected a positive measured cost in {cost['metric']!r}")
