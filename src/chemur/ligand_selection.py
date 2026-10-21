"""Resolve chemical ligands without changing deposited residue identities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _value(record, name: str, default=None):
    return record.get(name, default) if isinstance(record, dict) else getattr(record, name, default)


@dataclass(frozen=True, slots=True)
class LigandSelection:
    ligand_id: str
    component_ids: tuple[str, ...]
    atom_ids: tuple[int, ...]
    owner_component_id: str
    chemistry: dict[str, Any]

    def as_component(self, components):
        """Return a private chemistry view; never add it to result.components."""
        from .models import ComponentRecord

        owner = next(component for component in components
                     if _value(component, "component_id") == self.owner_component_id)
        metadata = dict(_value(owner, "metadata", {}) or {})
        metadata.update(self.chemistry)
        return ComponentRecord(
            self.ligand_id, _value(owner, "name", ""), _value(owner, "chain_id", ""),
            _value(owner, "residue_id", ""), "ligand", self.atom_ids, metadata,
        )

    def owns_feature(self, feature) -> bool:
        """Boundary groups containing receptor atoms are not ligand features."""
        atom_ids = set(_value(feature, "atom_ids", ()))
        return bool(atom_ids) and atom_ids.issubset(self.atom_ids)

    def touches(self, interaction, *, external: bool = False) -> bool:
        atom_ids = set(_value(interaction, "atom_ids", ()))
        ligand_atoms = set(self.atom_ids)
        if not atom_ids.intersection(ligand_atoms):
            return False
        return not external or bool(atom_ids - ligand_atoms)

    def has_boundary_feature(self, interaction, features_by_id) -> bool:
        ligand_atoms = set(self.atom_ids)
        for feature_id in _value(interaction, "feature_ids", ()):
            feature = features_by_id.get(feature_id)
            if feature is None:
                continue
            atom_ids = set(_value(feature, "atom_ids", ()))
            if atom_ids & ligand_atoms and atom_ids - ligand_atoms:
                return True
        return False


def ligand_selections(source) -> list[LigandSelection]:
    """List intact ligand assemblies, with a fallback for legacy results."""
    components = list(_value(source, "components", ()))
    metadata = _value(source, "metadata", {}) or {}
    registry = dict(metadata.get("ligand_assemblies") or {})
    for component in components:
        record = (_value(component, "metadata", {}) or {}).get("ligand_assembly")
        if isinstance(record, dict) and record.get("assembly_id"):
            registry.setdefault(str(record["assembly_id"]), record)
    by_component = {_value(component, "component_id"): component for component in components}
    assembly_by_component = {
        str(component_id): str(assembly_id)
        for assembly_id, record in registry.items()
        for component_id in record.get("component_ids", ())
    }
    groups: dict[str, list[str]] = {}
    for component in components:
        component_meta = _value(component, "metadata", {}) or {}
        if _value(component, "molecule_type") != "ligand" or component_meta.get("ignored_by_template"):
            continue
        component_id = str(_value(component, "component_id"))
        assembly_id = str(component_meta.get("ligand_assembly_id")
                          or assembly_by_component.get(component_id) or component_id)
        groups.setdefault(assembly_id, []).append(component_id)

    selections = []
    for assembly_id, members in sorted(groups.items()):
        component_ids = tuple(sorted(members))
        owner_id = component_ids[0]
        atom_ids = tuple(sorted({
            int(atom_id) for member in component_ids
            for atom_id in _value(by_component[member], "atom_ids", ())
        }))
        chemistry = dict(registry.get(assembly_id) or
                         _value(by_component[owner_id], "metadata", {}) or {})
        selections.append(LigandSelection(assembly_id, component_ids, atom_ids, owner_id, chemistry))
    return selections


def resolve_ligand(source, ligand_id: str) -> LigandSelection:
    """Resolve an assembly ID or any member's original component ID."""
    for selection in ligand_selections(source):
        if ligand_id == selection.ligand_id or ligand_id in selection.component_ids:
            return selection
    raise ValueError(f"Ligand not found: {ligand_id}")
