"""
MLflow experiment tracking wrapper.
Provides a thin, opinionated interface over mlflow.start_run().
"""

from typing import Any

import mlflow


def start_run(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    tags: dict[str, str] | None = None,
):
    """
    Start a tracked MLflow run. Logs all params immediately.
    Returns the active run context manager.
    """
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name, tags=tags or {})
    mlflow.log_params(params)
    return run
