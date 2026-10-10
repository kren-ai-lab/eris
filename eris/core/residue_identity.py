from __future__ import annotations


def format_residue_id(
    *,
    chain_id: str,
    residue_number: int,
    insertion_code: str | None,
    residue_name: str,
) -> str:
    """Format Eris' internal canonical biological residue identifier."""

    normalized_insertion_code = (insertion_code or "").strip()
    return f"{chain_id}:{residue_number}:{normalized_insertion_code}:{residue_name}"


__all__ = ["format_residue_id"]
