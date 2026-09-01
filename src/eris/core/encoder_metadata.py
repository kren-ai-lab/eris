from eris.utils.dict_utils import filter_none


def check_params(**kwargs) -> dict:
    """
    Build encoder params metadata (drop None values).
    """
    return filter_none(kwargs)
