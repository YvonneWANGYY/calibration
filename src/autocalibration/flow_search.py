"""Cross-validated Optuna selection and target-conditioned flow proposals."""

import json
import math

import optuna
import torch

from .lhs import _from_unit, to_unit
from .paper_sage.flow.nsf_core import ConditionalNeuralSplineFlow, FlowConfig

DEFAULT_SPACE = {
    "layers": [2, 4], "width": [16, 32], "bins": [4, 8],
    "learning_rate": [0.001, 0.003],
}


def settings(problem, initial_count):
    config = {"schedule": "initial", "trials": 8, "folds": 3, "epochs": 30, "members": 3}
    supplied = problem.get("tuning", {})
    if not isinstance(supplied, dict) or supplied.keys() - (config.keys() | {"search_space"}):
        raise ValueError("invalid tuning object or unknown tuning setting")
    config.update(supplied)
    if config["schedule"] not in ("initial", "every_iteration"):
        raise ValueError("tuning.schedule must be initial or every_iteration")
    for key in ("trials", "folds", "epochs", "members"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"tuning.{key} must be a positive integer")
    if not 2 <= config["folds"] <= initial_count:
        raise ValueError("tuning.folds must be between 2 and the number of initial evaluations")
    space = config.get("search_space", {})
    if not isinstance(space, dict) or space.keys() - DEFAULT_SPACE.keys():
        raise ValueError("unknown tuning.search_space field")
    config["search_space"] = DEFAULT_SPACE | space
    for key, choices in config["search_space"].items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"tuning.search_space.{key} must be a nonempty list")
        for value in choices:
            if key == "learning_rate":
                valid = type(value) in (float, int) and math.isfinite(value) and value > 0
            else:
                valid = type(value) is int and value >= (2 if key == "bins" else 1)
            if not valid:
                raise ValueError(f"invalid {key} choice: {value}")
    return config


def fit(x, c, params, epochs, seed):
    torch.manual_seed(seed)
    model = ConditionalNeuralSplineFlow(FlowConfig(
        features=x.shape[1], context_features=c.shape[1], num_layers=params["layers"],
        num_bins=params["bins"], hidden_dims=(params["width"], params["width"]),
    ))
    mean = c.mean(0)
    scale = c.std(0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    context = (c - mean) / scale
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = -model.log_prob(x, context).mean()
        loss.backward()
        optimizer.step()
    return model.eval(), mean, scale


def tune(x, c, config, seed):
    generator = torch.Generator().manual_seed(seed)
    folds = torch.randperm(len(x), generator=generator).tensor_split(config["folds"])

    def objective(trial):
        params = {k: trial.suggest_categorical(k, values)
                  for k, values in config["search_space"].items()}
        total = 0.0
        for i, valid in enumerate(folds):
            train = torch.cat(folds[:i] + folds[i + 1:])
            model, mean, scale = fit(x[train], c[train], params, config["epochs"], seed + i)
            with torch.no_grad():
                total -= model.log_prob(x[valid], (c[valid] - mean) / scale).sum().item()
        if not math.isfinite(total):
            raise FloatingPointError("non-finite validation loss")
        return total / len(x)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed % (2**32)))
    study.optimize(objective, n_trials=config["trials"], catch=(FloatingPointError,))
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError("no tuning trial completed with a finite validation loss")
    records = [{"trial": t.number, "parameters": t.params, "validation_nll": t.value,
                "state": t.state.name} for t in study.trials]
    return study.best_params, records


def datasets(session):
    units = [[to_unit(p, obs.parameters[p.name]) for p in session.parameters]
             for obs in session.observations]
    # Logit coordinates give the flow a bounded physical domain at sampling time.
    x = torch.logit(torch.tensor(units), eps=1e-6)
    c = torch.tensor([[obs.metrics[k] for k in session.target_profile]
                      for obs in session.observations])
    return x, c


def select(models, target, session, pool_size, weight, final):
    contexts = [(target - mean) / scale for _, mean, scale in models]
    with torch.no_grad():
        parts = []
        for i, ((model, _, _), context) in enumerate(zip(models, contexts)):
            count = pool_size // len(models) + int(i < pool_size % len(models))
            if count:
                parts.append(model.sample(count, context))
        candidates = torch.cat(parts)
        scores = torch.stack([model.log_prob(candidates, ctx.expand(pool_size, -1))
                              for (model, _, _), ctx in zip(models, contexts)])
        density = scores.logsumexp(0) - math.log(len(models))
        if final:
            # Convert log density to physical coordinates (additive constants cancel).
            density += (torch.nn.functional.softplus(candidates)
                        + torch.nn.functional.softplus(-candidates)).sum(1)
            for j, p in enumerate(session.parameters):
                if p.scale == "log":
                    density -= candidates[:, j].sigmoid() * (math.log(p.upper) - math.log(p.lower))
            score = density
        else:
            def standardize(v):
                return (v - v.mean()) / v.std(unbiased=False).clamp_min(1e-8)
            score = standardize(density) - weight * standardize(scores.std(0, unbiased=False))
        units = candidates[score.argmax()].sigmoid().tolist()
    return {p.name: _from_unit(p, u) for p, u in zip(session.parameters, units)}


def flow_rounds(session, problem, output, config):
    target = torch.tensor([list(session.target_profile.values())], dtype=torch.float32)
    rounds = problem["adaptive_steps"]
    params = None
    with (output / "flow_history.jsonl").open("x") as log:
        for step in range(rounds + 1):
            final = step == rounds
            seed = (problem["seed"] + step * 1009) % (2**32)
            x, c = datasets(session)
            trials = []
            if params is None or (not final and config["schedule"] == "every_iteration"):
                params, trials = tune(x, c, config, seed)
            models = [fit(x, c, params, config["epochs"], seed + k)
                      for k in range(config["members"])]
            progress = step / max(1, rounds - 1) if not final else 0
            weight = -1 - (2 / 3) * math.log1p(math.expm1(-3) * progress)
            theta = select(models, target, session, problem["candidate_pool_size"], weight, final)
            kind = "final_recommendation" if final else "adaptive"
            entry = dict(round=step + 1, kind=kind, seed=seed, observations=len(x), parameters=params,
                         trials=trials, disagreement_weight=None if final else weight)
            log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
            yield {"kind": kind, "parameters": theta, "rank": 1}
