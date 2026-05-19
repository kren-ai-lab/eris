from eris.utils.dict_utils import filter_none


NODE_ATTR_KEY_MAP = {
    "onehot": ["aa_onehot"],
    "physchem": ["physchem"],
    "plddt": ["plddt"],
    "bfactor": ["bfactor"],
}

EDGE_ATTR_KEY_MAP = {
    "dist": ["dist"],
    "rbf": ["rbf"],
    "seq": ["seq_sep"],
    "local": ["local_contact"],
}


def check_params(**kwargs) -> dict:
    """
    Build encoder params metadata (drop None values).
    """
    return filter_none(kwargs)


def parse_mode(mode: str) -> list:
    """
    'onehot+physchem+plddt' -> ['onehot', 'physchem', 'plddt']
    """
    return [m.strip() for m in mode.split("+") if m.strip()]


def build_attr_keys(mode: str, key_map: dict[str, list[str]]) -> list:
    keys = []
    for token in parse_mode(mode):
        try:
            keys.extend(key_map[token])
        except KeyError:
            raise ValueError(f"Unknown feature token: {token}")
    return keys


