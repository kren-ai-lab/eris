from __future__ import annotations

import json
from typing import Any


def apply_params(enc: Any, params_json: str, allow_unknown: bool = False) -> None:
    """
    Apply JSON parameters to an encoder instance via setattr.
    """
    if not params_json or params_json.strip() in ("{}", ""):
        return

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --params: {e}") from e

    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object/dict, e.g. '{\"cutoff\": 8.0}'")

    for k, v in params.items():
        if hasattr(enc, k):
            setattr(enc, k, v)
        else:
            if allow_unknown:
                continue
            raise ValueError(
                f"Encoder '{getattr(enc, 'name', enc.__class__.__name__)}' has no parameter '{k}'"
            )
