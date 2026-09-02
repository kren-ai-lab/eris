from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueContactMapConfig
from eris.core.io import load_and_extract_residue
from eris.core.representations import ContactMapRepresentation
from eris.core.representations.output_declarations import RESIDUE_CONTACT_MAP_OUTPUTS

if TYPE_CHECKING:
    from eris.core.cache import DiskCache
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass(init=False)
class ContactMapCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_contact_map"
    version: ClassVar[str] = "0.1.1"
    representation: ClassVar[BaseRepresentation] = ContactMapRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_CONTACT_MAP_OUTPUTS

    config: ResidueContactMapConfig = field(init=False)

    def __init__(
        self,
        *,
        config: ResidueContactMapConfig | None = None,
        use_cache: bool = True,
        cache: DiskCache | None = None,
        **representation_kwargs: Any,
    ) -> None:
        if config is not None:
            if not isinstance(config, ResidueContactMapConfig):
                raise TypeError("config must be a ResidueContactMapConfig")
            if representation_kwargs:
                names = ", ".join(sorted(representation_kwargs))
                raise TypeError(f"config cannot be combined with representation kwargs: {names}")
            resolved_config = config
        else:
            resolved_config = ResidueContactMapConfig(**representation_kwargs)

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
    def residue_center(self) -> str:
        return self.config.residue_center

    @property
    def amino_acids_only(self) -> bool:
        return self.config.amino_acids_only

    @property
    def altloc(self) -> str:
        return self.config.altloc

    @property
    def missing_residue_center(self) -> str:
        return self.config.missing_residue_center

    @property
    def cutoff(self) -> float:
        return self.config.cutoff

    @property
    def include_self(self) -> bool:
        return self.config.include_self

    @property
    def dtype(self) -> str:
        return self.config.dtype

    def get_params(self) -> dict[str, Any]:
        return self.config.to_dict()

    def _load_structure(self, ctx: EncodingContext) -> dict[str, Any]:
        return load_and_extract_residue(
            ctx.inp,
            amino_acids_only=self.config.amino_acids_only,
            mode=self.config.residue_center,
            altloc=self.config.altloc,
            missing_ca=self.config.missing_residue_center,
        )

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        diff = pos_out[:, None, :] - pos_out[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)

        contact = dist <= self.config.cutoff

        if not self.config.include_self:
            np.fill_diagonal(contact, False)

        if self.config.dtype == "bool":
            contact_out = contact
        else:
            contact_out = contact.astype(np.uint8)

        return {
            "contact": contact_out,
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "omitted_residue_ids": np.asarray(strc.get("omitted_residue_ids", []), dtype=np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
        }
