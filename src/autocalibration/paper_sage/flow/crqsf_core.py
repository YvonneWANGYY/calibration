"""Compatibility imports for the archived SAGE training scripts."""

if __package__:
    from . import nsf_core as core
else:
    import nsf_core as core

ConditionalRQSplineFlow = core.ConditionalNeuralSplineFlow
ConditionalSplineCoupling = core.ConditionalSplineCoupling
FlowConfig = core.FlowConfig
nll_loss = core.nll_loss
parse_hidden_dims = core.parse_hidden_dims
rational_quadratic_spline = core.rational_quadratic_spline
