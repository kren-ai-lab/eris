"""Atom-level point-cloud encoder."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np

from eris.core.base_encoder import BaseAtomEncoder, EncodingContext
from eris.core.config import AtomPointCloudConfig
from eris.core.representations import PointCloudRepresentation
from eris.core.representations.output_declarations import ATOM_POINT_CLOUD_OUTPUTS
from eris.encoders._context import prepared_arrays
from eris.features.node.atom import build_node_attr


class AtomPointCloud(BaseAtomEncoder):
    """Atom-level point cloud.

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
    representation: ClassVar = PointCloudRepresentation()
    output_declarations: ClassVar = ATOM_POINT_CLOUD_OUTPUTS

    config_type: ClassVar = AtomPointCloudConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        config = cast("AtomPointCloudConfig", self.config)
        strc, centroid, pos_out = prepared_arrays(ctx)

        node_attr, node_attr_keys = build_node_attr(
            elem_index=strc["elem_index"],
            atom_role=strc["atom_role"],
            bfactor=strc["bfactor"],
            occ=strc["occ"],
            mode=config.node_attr_mode.to_list(),
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
