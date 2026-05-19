# Import submodules so their register_encoder(...) calls run at import time.
from eris.encoders import graphs as _graphs  # noqa: F401
from eris.encoders import grids as _grids  # noqa: F401
from eris.encoders import geometry as _geometry  # noqa: F401
from eris.encoders import clouds as _clouds  # noqa: F401
