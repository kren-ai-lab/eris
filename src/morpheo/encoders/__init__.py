# Import submodules so their register_encoder(...) calls run at import time.
from morpheo.encoders import graphs as _graphs  # noqa: F401
from morpheo.encoders import grids as _grids  # noqa: F401
from morpheo.encoders import geometry as _geometry  # noqa: F401
from morpheo.encoders import clouds as _clouds  # noqa: F401
