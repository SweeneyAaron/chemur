from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
import warnings

from .errors import DependencyMissingError, LigandTemplateError
from .models import AtomRecord, ComponentRecord


@dataclass(frozen=True)
class AtomAnnotation:
    donor_capacity: int = 0
    acceptor_capacity: int = 0
    halogen_donor_capacity: int = 0
    hydrogen_count: int = 0
    formal_charge: int = 0
    is_aromatic: bool = False
    is_hydrophobe: bool = False


def apply_ligand_smiles_templates(
    atoms: list[AtomRecord],
    components: list[ComponentRecord],
    ligand_smiles: dict[str, str] | None,
    *,
    strict: bool = False,
) -> None:
    """Transfer SMILES-derived chemistry onto matching ligand components.

    A ligand whose SMILES cannot be parsed or mapped (unparseable, atom-count or
    element mismatch, no isomorphism, or an ambiguous mapping) is, by default,
    skipped: it is marked ignored (so it contributes no interactions) and a
    warning is emitted, leaving the rest of the structure to be analyzed. Pass
    ``strict=True`` to re-raise :class:`LigandTemplateError` instead.
    """
    if not ligand_smiles:
        return

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyMissingError(
            "RDKit is required when ligand SMILES templates are supplied. "
            "Install this project with its runtime dependencies or install rdkit."
        ) from exc

    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    templates = {key.upper(): value for key, value in ligand_smiles.items()}

    for component in components:
        if component.molecule_type != "ligand":
            continue
        smiles = templates.get(component.component_id.upper()) or templates.get(component.name.upper())
        if smiles is None:
            continue

        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise LigandTemplateError(
                    f"Could not parse SMILES for ligand {component.name}: {smiles!r}"
                )

            component_atoms = [atoms_by_id[atom_id] for atom_id in component.atom_ids]
            mapping = _map_template_to_component(molecule, component, component_atoms)
            apply_rdkit_ligand_template(
                atoms,
                component,
                molecule,
                mapping,
                smiles=smiles,
                source="smiles",
            )
        except LigandTemplateError as exc:
            if strict:
                raise
            _ignore_component(component, atoms_by_id, reason=str(exc))
            warnings.warn(
                f"Skipping ligand {component.component_id}; it will not contribute "
                f"interactions. Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def apply_rdkit_ligand_template(
    atoms: list[AtomRecord],
    component: ComponentRecord,
    molecule,
    mapping: dict[int, int],
    *,
    smiles: str,
    source: str,
) -> None:
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    annotations = _template_annotations(molecule)

    for template_idx, atom_id in mapping.items():
        if template_idx not in annotations:
            continue
        annotation = annotations[template_idx]
        atom = atoms_by_id[atom_id]
        atom.donor_capacity = annotation.donor_capacity
        atom.acceptor_capacity = annotation.acceptor_capacity
        atom.halogen_donor_capacity = annotation.halogen_donor_capacity
        atom.hydrogen_count = annotation.hydrogen_count
        atom.formal_charge = annotation.formal_charge
        atom.is_aromatic = annotation.is_aromatic
        atom.is_hydrophobe = annotation.is_hydrophobe
        atom.metadata["chemistry_source"] = source
        _apply_template_hydrogen_policy(atom, annotation, atoms_by_id)

    component.metadata["smiles"] = smiles
    component.metadata["template_source"] = source
    component.metadata["template_mapping"] = {
        str(template_idx): atom_id for template_idx, atom_id in mapping.items()
    }
    component.metadata["template_rings"] = [
        tuple(mapping[idx] for idx in ring if idx in mapping)
        for ring in molecule.GetRingInfo().AtomRings()
        if all(idx in mapping for idx in ring)
    ]


def mark_untemplated_ligands_ignored(
    atoms: list[AtomRecord],
    components: list[ComponentRecord],
    ligand_smiles: dict[str, str],
    *,
    reason: str,
) -> None:
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    templated_names = {name.upper() for name in ligand_smiles}
    for component in components:
        if component.molecule_type != "ligand":
            continue
        if component.name.upper() in templated_names:
            continue
        _ignore_component(component, atoms_by_id, reason=reason)


def _ignore_component(
    component: ComponentRecord,
    atoms_by_id: dict[int, AtomRecord],
    *,
    reason: str,
) -> None:
    """Mark a ligand component (and its atoms) as ignored, clearing chemistry."""
    component.metadata["ignored_by_template"] = True
    component.metadata["ignored_reason"] = reason
    for atom_id in component.atom_ids:
        atom = atoms_by_id.get(atom_id)
        if atom is None:
            continue
        atom.metadata["ignored_by_template"] = True
        atom.metadata["ignored_reason"] = reason
        atom.donor_capacity = 0
        atom.acceptor_capacity = 0
        atom.halogen_donor_capacity = 0
        atom.formal_charge = 0
        atom.is_aromatic = False
        atom.is_hydrophobe = False


def _apply_template_hydrogen_policy(
    atom: AtomRecord,
    annotation: AtomAnnotation,
    atoms_by_id: dict[int, AtomRecord],
) -> None:
    bonded_hydrogens = [
        atoms_by_id[neighbor_id]
        for neighbor_id in atom.bonds
        if neighbor_id in atoms_by_id and atoms_by_id[neighbor_id].element == "H"
    ]
    atom.metadata["structure_hydrogen_count"] = len(bonded_hydrogens)
    atom.metadata["template_hydrogen_count"] = annotation.hydrogen_count

    if len(bonded_hydrogens) <= annotation.hydrogen_count:
        return

    extra_hydrogens = bonded_hydrogens[annotation.hydrogen_count :]
    atom.metadata["ignored_hydrogen_atom_ids"] = [
        hydrogen.atom_id for hydrogen in extra_hydrogens
    ]
    for hydrogen in extra_hydrogens:
        hydrogen.metadata["ignored_by_template"] = True
        hydrogen.metadata["ignored_reason"] = (
            f"SMILES template for {atom.component_id}:{atom.name} "
            f"has {annotation.hydrogen_count} hydrogen(s)"
        )
        hydrogen.donor_capacity = 0
        hydrogen.acceptor_capacity = 0
        hydrogen.halogen_donor_capacity = 0


def _map_template_to_component(
    molecule,
    component: ComponentRecord,
    component_atoms: list[AtomRecord],
) -> dict[int, int]:
    heavy_template_indices = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() != "H"
    ]
    heavy_structure_atoms = [atom for atom in component_atoms if atom.element != "H"]

    if len(heavy_template_indices) != len(heavy_structure_atoms):
        raise LigandTemplateError(
            f"Ligand {component.component_id} has {len(heavy_structure_atoms)} heavy "
            f"atoms in the structure but {len(heavy_template_indices)} in the SMILES template"
        )

    template_elements = sorted(
        _rdkit_symbol(molecule.GetAtomWithIdx(idx)) for idx in heavy_template_indices
    )
    structure_elements = sorted(atom.element for atom in heavy_structure_atoms)
    if template_elements != structure_elements:
        raise LigandTemplateError(
            f"Ligand {component.component_id} element mismatch between structure "
            f"{structure_elements} and SMILES template {template_elements}"
        )

    mappings = _graph_isomorphisms(molecule, heavy_template_indices, heavy_structure_atoms)
    if not mappings:
        raise LigandTemplateError(
            f"Could not map SMILES template onto ligand {component.component_id}"
        )

    # More than one isomorphism can exist, but only because resonance-equivalent
    # terminal atoms are interchangeable: a phosphate/carboxylate/sulfonate/nitro group
    # has topologically identical terminal oxygens (each a degree-1 O on the same
    # parent) that the heavy-atom graph can map either way. Any two valid isomorphisms
    # of the same template onto the same structure differ only by a structure
    # automorphism (element matching forbids swapping atoms of different elements), so
    # they assign chemically equivalent labelings -- the choice only relabels
    # interchangeable atoms, leaving net charge, donor/acceptor counts and ring
    # perception unchanged. Picking the first mapping is therefore always correct;
    # rejecting these as "ambiguous" (the previous behaviour) was a false positive that
    # dropped every phosphate-bearing ligand (ADP, ATP, GTP, ...) from the analysis.
    return mappings[0]


def _graph_isomorphisms(
    molecule,
    heavy_template_indices: list[int],
    heavy_structure_atoms: list[AtomRecord],
    limit: int = 3,
) -> list[dict[int, int]]:
    template_neighbors = _template_heavy_neighbors(molecule, heavy_template_indices)
    structure_neighbors = _structure_heavy_neighbors(heavy_structure_atoms)

    candidate_atoms: dict[int, list[AtomRecord]] = {}
    for template_idx in heavy_template_indices:
        template_atom = molecule.GetAtomWithIdx(template_idx)
        symbol = _rdkit_symbol(template_atom)
        template_degree = len(template_neighbors[template_idx])
        candidate_atoms[template_idx] = [
            atom
            for atom in heavy_structure_atoms
            if atom.element == symbol
            and len(structure_neighbors.get(atom.atom_id, set())) == template_degree
        ]

    order = sorted(heavy_template_indices, key=lambda idx: len(candidate_atoms[idx]))
    mappings: list[dict[int, int]] = []

    def backtrack(mapping: dict[int, int], used_structure_atoms: set[int]) -> None:
        if len(mappings) >= limit:
            return
        if len(mapping) == len(order):
            mappings.append(dict(mapping))
            return
        template_idx = order[len(mapping)]
        for structure_atom in candidate_atoms[template_idx]:
            if structure_atom.atom_id in used_structure_atoms:
                continue
            if _compatible(
                template_idx,
                structure_atom.atom_id,
                mapping,
                template_neighbors,
                structure_neighbors,
            ):
                mapping[template_idx] = structure_atom.atom_id
                used_structure_atoms.add(structure_atom.atom_id)
                backtrack(mapping, used_structure_atoms)
                used_structure_atoms.remove(structure_atom.atom_id)
                del mapping[template_idx]

    backtrack({}, set())
    return mappings


def _compatible(
    template_idx: int,
    structure_atom_id: int,
    mapping: dict[int, int],
    template_neighbors: dict[int, set[int]],
    structure_neighbors: dict[int, set[int]],
) -> bool:
    for mapped_template_idx, mapped_structure_id in mapping.items():
        template_has_edge = mapped_template_idx in template_neighbors[template_idx]
        structure_has_edge = mapped_structure_id in structure_neighbors.get(
            structure_atom_id, set()
        )
        if template_has_edge != structure_has_edge:
            return False
    return True


def _template_heavy_neighbors(
    molecule,
    heavy_template_indices: Iterable[int],
) -> dict[int, set[int]]:
    heavy = set(heavy_template_indices)
    neighbors: dict[int, set[int]] = {idx: set() for idx in heavy}
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in heavy and end in heavy:
            neighbors[begin].add(end)
            neighbors[end].add(begin)
    return neighbors


def _structure_heavy_neighbors(
    atoms: Iterable[AtomRecord],
) -> dict[int, set[int]]:
    atom_ids = {atom.atom_id for atom in atoms if atom.element != "H"}
    neighbors: dict[int, set[int]] = defaultdict(set)
    for atom in atoms:
        if atom.atom_id not in atom_ids:
            continue
        for neighbor_id in atom.bonds:
            if neighbor_id in atom_ids:
                neighbors[atom.atom_id].add(neighbor_id)
    return neighbors


def _template_annotations(molecule) -> dict[int, AtomAnnotation]:
    annotations: dict[int, AtomAnnotation] = {}
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        symbol = _rdkit_symbol(atom)
        hydrogen_count = _total_hydrogen_count(atom)
        formal_charge = int(atom.GetFormalCharge())
        annotations[atom.GetIdx()] = AtomAnnotation(
            donor_capacity=1 if _is_donor(atom) else 0,
            acceptor_capacity=_acceptor_capacity(atom),
            halogen_donor_capacity=1 if symbol in {"Cl", "Br", "I"} else 0,
            hydrogen_count=hydrogen_count,
            formal_charge=formal_charge,
            is_aromatic=bool(atom.GetIsAromatic()),
            is_hydrophobe=symbol in {"C", "S", "Cl", "Br", "I"},
        )
    return annotations


def rdkit_atom_annotations(molecule) -> dict[int, AtomAnnotation]:
    """Return ChemeleonX's RDKit-derived per-heavy-atom chemistry annotations."""
    return _template_annotations(molecule)


def _rdkit_symbol(atom) -> str:
    symbol = atom.GetSymbol()
    if len(symbol) > 1:
        return symbol[0] + symbol[1:].lower()
    return symbol


def _is_donor(atom) -> bool:
    if atom.GetSymbol() not in {"N", "O", "S"}:
        return False
    if atom.GetFormalCharge() < 0:
        return False
    return _total_hydrogen_count(atom) > 0


def _acceptor_capacity(atom) -> int:
    symbol = atom.GetSymbol()
    formal_charge = atom.GetFormalCharge()
    if symbol == "O":
        return 2 if formal_charge <= 0 else 0
    if symbol == "S":
        return 2 if formal_charge <= 0 else 0
    if symbol == "N":
        if formal_charge > 0:
            return 0
        if _total_hydrogen_count(atom) > 0 and atom.GetDegree() >= 3:
            return 0
        return 1
    return 0


def _total_hydrogen_count(atom) -> int:
    try:
        return int(atom.GetTotalNumHs(True))
    except TypeError:
        return int(atom.GetTotalNumHs())
