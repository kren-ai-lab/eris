from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseAtomEncoder, EncodingContext
from eris.core.config import AtomPointCloudConfig
from eris.core.representations import PointCloudRepresentation
from eris.core.representations.output_declarations import ATOM_POINT_CLOUD_OUTPUTS
from eris.features.node.atom import build_node_attr

if TYPE_CHECKING:
    from eris.core.cache import DiskCache
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass(init=False)
class AtomPointCloud(BaseAtomEncoder):
    """
    Atom-level point cloud.

    Stores:
      - pos (N,3)
      - elem_index (N,)
      - atom_role (N,)
      - bfactor (N,)
      - occ (N,)
      - residue_ids (N,)
      - resname (N,)
      - atom_name (N,)
      - seq_pos_residue (N,)
      - aa_index_residue (N,)

    Optional:
      - node_attr (N,F) if node_attr_mode != "none"
    """

    # Identity
    name: ClassVar[str] = "atom_point_cloud"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar[BaseRepresentation] = PointCloudRepresentation()
    output_declarations_v2: ClassVar = ATOM_POINT_CLOUD_OUTPUTS

    config: AtomPointCloudConfig = field(init=False)

    def __init__(
        self,
        *,
        config: AtomPointCloudConfig | None = None,
        use_cache: bool = True,
        cache: DiskCache | None = None,
        **representation_kwargs: Any,
    ) -> None:
        if config is not None:
            if not isinstance(config, AtomPointCloudConfig):
                raise TypeError("config must be an AtomPointCloudConfig")
            if representation_kwargs:
                names = ", ".join(sorted(representation_kwargs))
                raise TypeError(f"config cannot be combined with representation kwargs: {names}")
            resolved_config = config
        else:
            resolved_config = AtomPointCloudConfig(**representation_kwargs)

        self.config = resolved_config
        self.use_cache = use_cache
        self.cache = cache
        super().__post_init__()

    @property
    def center(self) -> bool:
        return self.config.center

    @property
    def scale(self) -> float | None:
        return self.config.scale

    @property
    def include_hetero(self) -> bool:
        return self.config.include_hetero

    @property
    def include_h(self) -> bool:
        return self.config.include_h

    @property
    def drop_water(self) -> bool:
        return self.config.drop_water

    @property
    def keep_ions(self) -> bool:
        return self.config.keep_ions

    @property
    def altloc(self) -> str:
        return self.config.altloc

    @property
    def amino_acids_only(self) -> bool:
        return self.config.amino_acids_only

    @property
    def node_attr_mode(self) -> str:
        return "+".join(token.name for token in self.config.node_attr_mode.tokens)

    def get_params(self) -> dict[str, Any]:
        return self.config.to_dict()

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        node_attr, node_attr_keys = build_node_attr(
            elem_index=strc["elem_index"],
            atom_role=strc["atom_role"],
            bfactor=strc["bfactor"],
            occ=strc["occ"],
            mode=self.config.node_attr_mode.to_list(),
            return_keys=True,
        )

        ctx.runtime["node_attr_keys"] = node_attr_keys

        # Base data always stored
        data: dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "elem_index": strc["elem_index"].astype(np.int16),
            "chain_id": strc["chain_id"].astype(np.str_),
            "atom_role": strc["atom_role"].astype(np.int8),
            "bfactor": strc["bfactor"].astype(np.float32),
            "occ": strc["occ"].astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "atom_name": strc["atom_name"].astype(np.str_),
            "seq_pos_residue": strc["seq_pos_residue"].astype(np.int32),
            "aa_index_residue": strc["aa_index_residue"].astype(np.int16),
        }

        return data
