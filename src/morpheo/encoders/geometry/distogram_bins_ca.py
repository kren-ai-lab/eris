from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, ClassVar
from typing import TYPE_CHECKING
from typing import Any, Dict, Optional, Sequence

import numpy as np

from morpheo.core.base_encoder import BaseResidueEncoder, EncodingContext
from morpheo.core.representations import DistogramBinsRepresentation
from morpheo.features.utils import validate_bin_edges

if TYPE_CHECKING:
    from morpheo.core.representations.base_rep import BaseRepresentation


@dataclass
class DistogramBinsCA(BaseResidueEncoder):
    #Identity
    name: ClassVar[str] = "distogram_bins_ca"
    representation: ClassVar[BaseRepresentation] = DistogramBinsRepresentation()

    #Instance
    edges: Optional[Sequence[float]] = None

    dmin: float = 2.0
    dmax: float = 20.0
    num_bins: int = 36

    include_self: bool = True

    # If True: class 0 = > last edge, class 1..K = in-range bins
    add_no_contact_class: bool = True

    dtype: str = "uint8"

    def __post_init__(self):
        super().__post_init__()

        if self.edges is not None:
            self.edges = tuple(float(x) for x in self.edges)
            self.dmin = None
            self.dmax = None
            self.num_bins = None
        else:
            self.edges = None
            self.dmin = float(self.dmin) if self.dmin is not None else None
            self.dmax = float(self.dmax) if self.dmax is not None else None
            self.num_bins = int(self.num_bins) if self.num_bins is not None else None


    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        n = int(pos_out.shape[0])

        diff = pos_out[:, None, :] - pos_out[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

        if not self.include_self:
            np.fill_diagonal(dist, np.nan)

        # Build edges
        if self.edges is not None:
            edges = validate_bin_edges(self.edges)
        else:
            edges = np.linspace(float(self.dmin), float(self.dmax), int(self.num_bins) + 1, dtype=np.float32)

        K = int(edges.size - 1)

        raw_bins = np.digitize(dist, edges, right=False) - 1
        raw_bins = np.clip(raw_bins, 0, K - 1)

        if self.add_no_contact_class:
            out = (raw_bins + 1).astype(np.int32)
            out[dist > float(edges[-1])] = 0
            out[np.isnan(dist)] = 0
        else:
            out = raw_bins.astype(np.int32)
            out[np.isnan(dist)] = 0

        if self.dtype == "uint16":
            dist_bins = out.astype(np.uint16)
        elif self.dtype == "int16":
            dist_bins = out.astype(np.int16)
        else:
            dist_bins = out.astype(np.uint8)

        # Update ctx
        ctx.runtime["num_bins"] = int(K)
        ctx.runtime["edges"] = edges.tolist(),

        data: Dict[str, Any] = {
            "dist_bins": dist_bins,
            "bin_edges": edges.astype(np.float32),
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
        }

        return data


    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        runtime = ctx.runtime
        
        meta["params"].update({
                "include_self": bool(self.include_self),
                "add_no_contact_class": bool(self.add_no_contact_class),
                "dtype": str(self.dtype),
                "dmin": None if self.edges is not None else float(self.dmin),
                "dmax": None if self.edges is not None else float(self.dmax),
                "num_bins": int(runtime["K"]),
                "edges": runtime["edges"].tolist(),
            })

        return meta
