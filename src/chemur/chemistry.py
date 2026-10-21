from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Iterable
import warnings

from .errors import DependencyMissingError, LigandTemplateError
from .models import AtomRecord, ComponentRecord


# Substituents electronegative enough to switch on a bonded chalcogen's sigma-hole.
# Sulfur is deliberately absent: whether a disulfide counts is not settled in the
# literature, and cystine is common enough that including it re-admits a large number
# of protein-protein contacts.
SIGMA_HOLE_ACTIVATING_ELEMENTS = {"N", "O", "F", "Cl", "Br", "I"}

# This is deliberately independent of the chalcogen sigma-hole rule: weak C-H
# hydrogen bonds require an electron-withdrawing environment on the carbon.
WEAK_HBOND_ACTIVATING_ELEMENTS = {"N", "O", "F", "Cl", "Br", "I"}

MAX_TEMPLATE_MAPPINGS = 4096
MAX_TEMPLATE_SEARCH_STATES = 100_000


def _chalcogen_sigma_hole_count(formal_charge: int, heavy_degree: int) -> int:
    """How many sigma-holes a chalcogen carries, one per R-Ch bond extension.

    Neutral divalent chalcogens (thioether, thiol, thiophene, thioamide) have one or
    two. A sulfonium is pyramidal with three substituents and three sigma-holes -- it
    is also the most strongly activated sulfur in biology, so excluding charged
    chalcogens outright would drop S-adenosylmethionine entirely.

    A thiolate is excluded: a negatively charged sulfur is a nucleophile, and donates
    nothing. Hypervalent sulfur (sulfoxide, sulfone, sulfonamide) is excluded too --
    there it is the oxygens that accept, and the geometry below does not describe it.
    """
    if formal_charge < 0 or formal_charge > 1:
        return 0
    maximum_degree = 3 if formal_charge == 1 else 2
    if not 1 <= heavy_degree <= maximum_degree:
        return 0
    return heavy_degree


@dataclass(frozen=True)
class AtomAnnotation:
    donor_capacity: int = 0
    acceptor_capacity: int = 0
    halogen_donor_capacity: int = 0
    chalcogen_donor_capacity: int = 0
    tetrel_donor_capacity: int = 0
    carbon_donor_capacity: int = 0
    hydrogen_count: int = 0
    formal_charge: int = 0
    is_aromatic: bool = False
    is_hydrophobe: bool = False
    sigma_hole_activated: bool = False
    weak_hbond_activated: bool = False


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
            # Canonical traversal makes equivalent representatives independent of
            # SMILES spelling. Translate back before applying: downstream consumers
            # reconstruct the original SMILES and need its original atom indices.
            original_molecule = molecule
            ranks = Chem.CanonicalRankAtoms(original_molecule, breakTies=True)
            original_indices = sorted(range(original_molecule.GetNumAtoms()), key=lambda idx: ranks[idx])
            molecule = Chem.RenumberAtoms(original_molecule, original_indices)

            component_atoms = [atoms_by_id[atom_id] for atom_id in component.atom_ids]
            mapping = _map_template_to_component(molecule, component, component_atoms)
            mapping = {original_indices[idx]: atom_id for idx, atom_id in mapping.items()}
            apply_rdkit_ligand_template(
                atoms,
                component,
                original_molecule,
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
    normalized = normalize_sulfoxide_resonance(molecule)
    if normalized is not molecule:
        from .assemblies import _serialize_molecule
        molecule = normalized
        smiles, serialized_mapping = _serialize_molecule(molecule, mapping)
    else:
        serialized_mapping = None
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    annotations = _template_annotations(molecule)
    apply_interaction_typing_metadata(molecule, mapping, atoms_by_id)

    # A new authoritative template may restore a previously excluded hydrogen or
    # component. Clear only state created by an earlier template application.
    for record in [component, *(atoms_by_id[atom_id] for atom_id in component.atom_ids)]:
        for key in ("ignored_by_template", "ignored_reason", "ignored_hydrogen_atom_ids",
                    "structure_hydrogen_count", "template_hydrogen_count"):
            record.metadata.pop(key, None)

    for template_idx, atom_id in mapping.items():
        if template_idx not in annotations:
            continue
        annotation = annotations[template_idx]
        atom = atoms_by_id[atom_id]
        atom.donor_capacity = annotation.donor_capacity
        atom.acceptor_capacity = annotation.acceptor_capacity
        atom.halogen_donor_capacity = annotation.halogen_donor_capacity
        atom.chalcogen_donor_capacity = annotation.chalcogen_donor_capacity
        atom.sigma_hole_activated = annotation.sigma_hole_activated
        atom.tetrel_donor_capacity = annotation.tetrel_donor_capacity
        atom.carbon_donor_capacity = annotation.carbon_donor_capacity
        atom.weak_hbond_activated = annotation.weak_hbond_activated
        atom.hydrogen_count = annotation.hydrogen_count
        atom.formal_charge = annotation.formal_charge
        atom.is_aromatic = annotation.is_aromatic
        atom.is_hydrophobe = annotation.is_hydrophobe
        atom.metadata["chemistry_source"] = source
        atom.metadata["acceptor_source"] = "rdkit_lipinski"
        atom.metadata.pop("acceptor_unresolved_reason", None)
        _apply_template_hydrogen_policy(atom, annotation, atoms_by_id)

    component.metadata["smiles"] = smiles
    component.metadata["template_source"] = source
    component.metadata["template_mapping"] = serialized_mapping or {
        str(template_idx): atom_id for template_idx, atom_id in mapping.items()
    }
    component.metadata["template_rings"] = [
        tuple(mapping[idx] for idx in ring if idx in mapping)
        for ring in molecule.GetRingInfo().AtomRings()
        if all(idx in mapping for idx in ring)
    ]
    component.metadata["template_carbonyls"] = _template_carbonyls(molecule, mapping)
    component.metadata["template_amides"] = _template_amides(molecule, mapping)


def _carbonyl_pairs(molecule) -> list[tuple[int, int]]:
    """(carbon, oxygen) template indices for every C=O, including carboxylates.

    A structure-only path cannot see bond orders, which is why this is captured
    here while the RDKit molecule is still in hand.
    """
    from rdkit import Chem

    pairs: list[tuple[int, int]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        symbols = {begin.GetSymbol(), end.GetSymbol()}
        if symbols != {"C", "O"}:
            continue
        carbon, oxygen = (begin, end) if begin.GetSymbol() == "C" else (end, begin)
        # A delocalised carboxylate/amide is stored as two 1.5-order bonds, so
        # accept aromatic and one-and-a-half bonds alongside a plain double bond.
        if bond.GetBondType() in {
            Chem.BondType.DOUBLE,
            Chem.BondType.AROMATIC,
            Chem.BondType.ONEANDAHALF,
        }:
            pairs.append((carbon.GetIdx(), oxygen.GetIdx()))
        elif oxygen.GetFormalCharge() < 0 and oxygen.GetDegree() == 1:
            pairs.append((carbon.GetIdx(), oxygen.GetIdx()))
    return pairs


def _template_carbonyls(molecule, mapping: dict[int, int]) -> list[tuple[int, int]]:
    return [
        (mapping[carbon], mapping[oxygen])
        for carbon, oxygen in _carbonyl_pairs(molecule)
        if carbon in mapping and oxygen in mapping
    ]


def _template_amides(molecule, mapping: dict[int, int]) -> list[tuple[int, int, int]]:
    """(carbon, oxygen, nitrogen) template indices for every amide C(=O)-N."""
    amides: list[tuple[int, int, int]] = []
    for carbon, oxygen in _carbonyl_pairs(molecule):
        carbon_atom = molecule.GetAtomWithIdx(carbon)
        for neighbor in carbon_atom.GetNeighbors():
            if neighbor.GetSymbol() != "N":
                continue
            nitrogen = neighbor.GetIdx()
            if carbon in mapping and oxygen in mapping and nitrogen in mapping:
                amides.append((mapping[carbon], mapping[oxygen], mapping[nitrogen]))
    return amides


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
        atom.chalcogen_donor_capacity = 0
        atom.sigma_hole_activated = False
        atom.tetrel_donor_capacity = 0
        atom.carbon_donor_capacity = 0
        atom.weak_hbond_activated = False
        atom.formal_charge = 0
        atom.is_aromatic = False
        atom.is_hydrophobe = False


def _apply_template_hydrogen_policy(
    atom: AtomRecord,
    annotation: AtomAnnotation,
    atoms_by_id: dict[int, AtomRecord],
) -> None:
    bonded_hydrogens = sorted((
        atoms_by_id[neighbor_id]
        for neighbor_id in atom.bonds
        if neighbor_id in atoms_by_id and atoms_by_id[neighbor_id].element == "H"
    ), key=_physical_atom_key)
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

    try:
        mappings = _graph_isomorphisms(molecule, heavy_template_indices, heavy_structure_atoms)
    except LigandTemplateError:
        component.metadata["template_mapping_status"] = "search_limit"
        raise
    if not mappings:
        raise LigandTemplateError(
            f"Could not map SMILES template onto ligand {component.component_id}"
        )

    annotations = _template_annotations(molecule)
    atoms_by_id = {atom.atom_id: atom for atom in component_atoms}
    ranked = [(mapping, _mapping_evidence(molecule, mapping, annotations, atoms_by_id, component))
              for mapping in mappings]
    best_score = min(score for _, score in ranked)
    best = [mapping for mapping, score in ranked if score == best_score]
    resonance_oxygens = _resonance_oxygen_indices(molecule, annotations)
    signatures = {
        _mapping_chemistry_signature(molecule, mapping, annotations, resonance_oxygens)
        for mapping in best
    }
    component.metadata["template_mapping_evidence"] = {
        "hydrogen_excess": best_score[0],
        "bond_order_mismatches": best_score[1],
        "charge_mismatches": best_score[2],
        "equally_supported_mappings": len(best),
    }
    if len(signatures) != 1:
        component.metadata["template_mapping_status"] = "ambiguous"
        raise LigandTemplateError(
            f"Ambiguous chemistry mapping for ligand {component.component_id}: "
            "equally supported mappings assign different chemistry to its atoms"
        )
    component.metadata["template_mapping_status"] = "unique" if len(best) == 1 else "equivalent"
    # Only chemically equivalent representations reach this tie-break. Atom IDs
    # and input order must never choose which physical site receives a proton.
    return min(best, key=lambda mapping: tuple(
        _physical_atom_key(atoms_by_id[mapping[idx]]) for idx in sorted(mapping)
    ))


def _physical_atom_key(atom: AtomRecord) -> tuple:
    return atom.name, tuple(atom.coord)


def _mapping_evidence(molecule, mapping, annotations, atoms_by_id, component) -> tuple[int, int, int]:
    observed_hydrogens = {
        atom_id: sum(atoms_by_id[idx].element == "H" for idx in atoms_by_id[atom_id].bonds
                     if idx in atoms_by_id)
        for atom_id in mapping.values()
    }
    excess = sum(max(observed_hydrogens[atom_id] - annotations[idx].hydrogen_count, 0)
                 for idx, atom_id in mapping.items())
    inverse = {atom_id: idx for idx, atom_id in mapping.items()}
    bond_mismatches = 0
    for bond in component.metadata.get("bond_orders", ()):
        first, second = int(bond["atom_id_1"]), int(bond["atom_id_2"])
        if first not in inverse or second not in inverse:
            continue
        template_bond = molecule.GetBondBetweenAtoms(inverse[first], inverse[second])
        if template_bond is None or abs(template_bond.GetBondTypeAsDouble() - float(bond["order"])) > 1e-6:
            bond_mismatches += 1
    charge_mismatches = 0
    for idx, atom_id in mapping.items():
        atom = atoms_by_id[atom_id]
        # Missing PDB charge is unknown, rather than evidence of neutrality.
        deposited = atom.metadata.get("structure_formal_charge")
        if deposited is not None and int(deposited) != 0 and int(deposited) != annotations[idx].formal_charge:
            charge_mismatches += 1
    return excess, bond_mismatches, charge_mismatches


def _resonance_oxygen_indices(molecule, annotations) -> set[int]:
    """A narrow resonance exception, never exchanging an OH with a bare O.

    RDKit's resonance supplier does not enumerate phosphate/sulfonate terminal
    oxygen permutations. Recognise their ordinary O=parent/O− forms explicitly.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for atom in molecule.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetSymbol() != "O" or annotations[idx].hydrogen_count:
            continue
        neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
        if len(neighbors) != 1 or neighbors[0].GetSymbol() not in {"C", "N", "P", "S"}:
            continue
        order = molecule.GetBondBetweenAtoms(idx, neighbors[0].GetIdx()).GetBondTypeAsDouble()
        if (atom.GetFormalCharge(), order) in {(0, 2.0), (-1, 1.0)}:
            groups[neighbors[0].GetIdx()].append(idx)
    equivalent: set[int] = set()
    for indices in groups.values():
        if {annotations[idx].formal_charge for idx in indices} != {0, -1}:
            continue
        labels = {tuple(getattr(annotations[idx], field.name) for field in fields(AtomAnnotation)
                        if field.name != "formal_charge") for idx in indices}
        if len(labels) == 1:
            equivalent.update(indices)
    return equivalent


def _mapping_chemistry_signature(molecule, mapping, annotations, resonance_oxygens) -> tuple:
    atom_labels = tuple(sorted(
        (atom_id, tuple(getattr(annotations[idx], field.name)
                        if field.name != "formal_charge" or idx not in resonance_oxygens else "resonance"
                        for field in fields(AtomAnnotation)))
        for idx, atom_id in mapping.items()
    ))
    bonds = []
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if first not in mapping or second not in mapping:
            continue
        order = "resonance" if first in resonance_oxygens or second in resonance_oxygens else bond.GetBondTypeAsDouble()
        bonds.append((tuple(sorted((mapping[first], mapping[second]))), order))
    rings = tuple(sorted(tuple(sorted(mapping[idx] for idx in ring))
                         for ring in molecule.GetRingInfo().AtomRings()))
    carbonyls = tuple(sorted(_template_carbonyls(molecule, mapping)))
    amides = tuple(sorted(_template_amides(molecule, mapping)))
    return atom_labels, tuple(sorted(bonds)), rings, carbonyls, amides


def _graph_isomorphisms(
    molecule,
    heavy_template_indices: list[int],
    heavy_structure_atoms: list[AtomRecord],
    limit: int | None = None,
    state_limit: int | None = None,
) -> list[dict[int, int]]:
    limit = MAX_TEMPLATE_MAPPINGS if limit is None else limit
    state_limit = MAX_TEMPLATE_SEARCH_STATES if state_limit is None else state_limit
    template_neighbors = _template_heavy_neighbors(molecule, heavy_template_indices)
    structure_neighbors = _structure_heavy_neighbors(heavy_structure_atoms)

    candidate_atoms: dict[int, list[AtomRecord]] = {}
    for template_idx in heavy_template_indices:
        template_atom = molecule.GetAtomWithIdx(template_idx)
        symbol = _rdkit_symbol(template_atom)
        template_degree = len(template_neighbors[template_idx])
        candidate_atoms[template_idx] = [
            atom
            for atom in sorted(heavy_structure_atoms, key=_physical_atom_key)
            if atom.element == symbol
            and len(structure_neighbors.get(atom.atom_id, set())) == template_degree
        ]

    order = sorted(heavy_template_indices, key=lambda idx: (len(candidate_atoms[idx]), idx))
    mappings: list[dict[int, int]] = []
    states = 0

    def backtrack(mapping: dict[int, int], used_structure_atoms: set[int]) -> None:
        nonlocal states
        states += 1
        if states > state_limit:
            raise LigandTemplateError(f"Template mapping search_limit: exceeded {state_limit} search states")
        if len(mapping) == len(order):
            if len(mappings) >= limit:
                raise LigandTemplateError(f"Template mapping search_limit: exceeded {limit} complete mappings")
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
    molecule = normalize_sulfoxide_resonance(molecule)
    annotations: dict[int, AtomAnnotation] = {}
    acceptor_indices = _acceptor_indices(molecule)
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        symbol = _rdkit_symbol(atom)
        hydrogen_count = _total_hydrogen_count(atom)
        formal_charge = int(atom.GetFormalCharge())
        annotations[atom.GetIdx()] = AtomAnnotation(
            donor_capacity=1 if _is_donor(atom) else 0,
            acceptor_capacity=(
                {"N": 1, "O": 2, "S": 2}.get(symbol, 0)
                if atom.GetIdx() in acceptor_indices else 0
            ),
            halogen_donor_capacity=1 if symbol in {"Cl", "Br", "I"} else 0,
            chalcogen_donor_capacity=_chalcogen_donor_capacity(atom),
            sigma_hole_activated=symbol in {"S", "Se"} and _sigma_hole_activated(atom),
            tetrel_donor_capacity=_tetrel_donor_capacity(atom),
            # Every carbon can donate a C-H to a pi system. This is deliberately
            # separate from donor_capacity, which _is_donor restricts to N/O/S --
            # so before this a templated ligand's carbons could never reach ch_pi.
            carbon_donor_capacity=max(hydrogen_count, 1) if symbol == "C" else 0,
            weak_hbond_activated=_weak_hbond_activated(atom),
            hydrogen_count=hydrogen_count,
            formal_charge=formal_charge,
            is_aromatic=bool(atom.GetIsAromatic()),
            is_hydrophobe=symbol in {"C", "S", "Cl", "Br", "I"},
        )
    return annotations


def _weak_hbond_activated(atom) -> bool:
    """Recognise supported activating environments, independently of hydrogen geometry.

    Aromaticity alone is insufficient. Charge exclusions are local: a remote
    anionic group does not disqualify an otherwise activated carbon.
    """
    from rdkit import Chem

    if atom.GetSymbol() != "C" or atom.GetFormalCharge() < 0:
        return False
    heavy_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() != "H"]
    carbonyl_bonds = {Chem.BondType.DOUBLE, Chem.BondType.AROMATIC, Chem.BondType.ONEANDAHALF}
    for neighbor in heavy_neighbors:
        if neighbor.GetFormalCharge() < 0:
            continue
        symbol = _rdkit_symbol(neighbor)
        if symbol in WEAK_HBOND_ACTIVATING_ELEMENTS:
            return True
        if symbol == "C":
            # Do not restrict the intervening C-C bond: base-ring carbons may
            # be aromatic in one representation and Kekulised in another.
            for bond in neighbor.GetBonds():
                other = bond.GetOtherAtom(neighbor)
                if (
                    other.GetSymbol() == "O"
                    and bond.GetBondType() in carbonyl_bonds
                    and sum(neighbor.GetSymbol() != "H" for neighbor in other.GetNeighbors()) == 1
                ):
                    return True
                if other.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.TRIPLE:
                    return True
            if len(heavy_neighbors) == 1:
                bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    return True
        elif symbol == "S":
            oxo_count = 0
            for bond in neighbor.GetBonds():
                oxygen = bond.GetOtherAtom(neighbor)
                if oxygen.GetSymbol() != "O":
                    continue
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    oxo_count += 1
                elif (
                    bond.GetBondType() == Chem.BondType.SINGLE
                    and neighbor.GetFormalCharge() > 0
                    and oxygen.GetFormalCharge() < 0
                    and oxygen.GetDegree() == 1
                ):
                    # S=O and [S+]-[O-] encode the same oxo group.
                    oxo_count += 1
            if oxo_count >= 2:
                return True
    return False


def _chalcogen_donor_capacity(atom) -> int:
    """Sigma-hole count for a chalcogen; see parser._chalcogen_donor_capacity."""
    if _rdkit_symbol(atom) not in {"S", "Se"}:
        return 0
    heavy_degree = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() != "H")
    return _chalcogen_sigma_hole_count(int(atom.GetFormalCharge()), heavy_degree)


def _sigma_hole_activated(atom) -> bool:
    """Whether this chalcogen plausibly carries a *positive* sigma-hole.

    Geometry alone over-calls chalcogen bonds. MEP calculations show the potential at
    the C-S extension of an ordinary protein Cys or Met is negative -- there is no
    positive cap to donate into, because nothing withdraws from that sulfur. A
    sigma-hole has to be switched on by an aromatic ring, an electronegative
    substituent, a positive charge, or conjugation into a C=S.
    """
    from rdkit import Chem

    if atom.GetFormalCharge() > 0:
        return True  # sulfonium, e.g. S-adenosylmethionine
    if atom.GetIsAromatic():
        return True  # thiophene, thiazole, thiadiazole, isothiazole
    if any(
        _rdkit_symbol(neighbor) in SIGMA_HOLE_ACTIVATING_ELEMENTS
        for neighbor in atom.GetNeighbors()
    ):
        return True  # S-nitroso, sulfenyl halide

    # Thioamide and thiourea: the withdrawing nitrogen sits two bonds away, across a
    # conjugated C=S. A thioketone (C-C(=S)-C) has no such substituent and fails here.
    # The bond-type set matches _carbonyl_pairs so delocalised thioureas still count.
    conjugated = {Chem.BondType.DOUBLE, Chem.BondType.AROMATIC, Chem.BondType.ONEANDAHALF}
    for bond in atom.GetBonds():
        carbon = bond.GetOtherAtom(atom)
        if carbon.GetSymbol() != "C" or bond.GetBondType() not in conjugated:
            continue
        if any(
            neighbor.GetIdx() != atom.GetIdx()
            and _rdkit_symbol(neighbor) in {"N", "O"}
            for neighbor in carbon.GetNeighbors()
        ):
            return True
    return False


def _tetrel_donor_capacity(atom) -> int:
    """Number of supported C/N/O/S roots on a validated sp3 carbon."""
    from rdkit import Chem

    if (atom.GetSymbol() != "C" or atom.GetIsAromatic()
            or atom.GetHybridization() != Chem.HybridizationType.SP3):
        return 0
    return sum(
        1
        for neighbor in atom.GetNeighbors()
        if _rdkit_symbol(neighbor) in {"C", "N", "O", "S"}
    )


def normalize_sulfoxide_resonance(molecule):
    """Canonicalize R2[S+]–[O-] to R2S=O without changing atom indices.

    The local resonance representation must not create a cation, an anion or
    sulfonium sigma-hole capacity. Sulfones and other S/O motifs are unchanged.
    """
    from rdkit import Chem
    result = Chem.RWMol(molecule)
    changed = False
    for sulfur in result.GetAtoms():
        heavy = [n for n in sulfur.GetNeighbors() if n.GetAtomicNum() != 1]
        if (sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 1
                or sulfur.GetIsAromatic() or len(heavy) != 3):
            continue
        oxygens = [n for n in heavy if n.GetSymbol() == "O" and n.GetFormalCharge() == -1
                   and n.GetDegree() == 1]
        carbons = [n for n in heavy if n.GetSymbol() == "C"]
        if len(oxygens) != 1 or len(carbons) != 2:
            continue
        oxygen = oxygens[0]
        bond = result.GetBondBetweenAtoms(sulfur.GetIdx(), oxygen.GetIdx())
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        sulfur.SetFormalCharge(0)
        oxygen.SetFormalCharge(0)
        bond.SetBondType(Chem.BondType.DOUBLE)
        changed = True
    if not changed:
        return molecule
    normalized = result.GetMol()
    Chem.SanitizeMol(normalized)
    return normalized


def apply_interaction_typing_metadata(molecule, mapping, atoms_by_id, *, atom_ids=None):
    """Attach motif eligibility separately from resource capacities."""
    from rdkit import Chem
    molecule = normalize_sulfoxide_resonance(molecule)
    for index, atom_id in mapping.items():
        if atom_ids is not None and atom_id not in atom_ids:
            continue
        source = molecule.GetAtomWithIdx(index)
        target = atoms_by_id[atom_id]
        target.metadata.pop("sulfur_pi_types", None)
        target.metadata.pop("tetrel_root_atom_ids", None)
        target.metadata.pop("tertiary_amine_cation", None)
        heavy = [n for n in source.GetNeighbors() if n.GetAtomicNum() > 1]
        if source.GetSymbol() == "N":
            target.metadata["tertiary_amine_cation"] = bool(
                source.GetFormalCharge() > 0 and source.GetHybridization() == Chem.HybridizationType.SP3 and len(heavy) == 3)
        if source.GetSymbol() == "S" and not source.GetIsAromatic() and source.GetFormalCharge() == 0:
            kinds = []
            singles = all(molecule.GetBondBetweenAtoms(index, n.GetIdx()).GetBondType() == Chem.BondType.SINGLE for n in heavy)
            if len(heavy) == 2 and singles:
                if all(n.GetSymbol() == "C" for n in heavy):
                    kinds.append("thioether")
                elif sorted(n.GetSymbol() for n in heavy) == ["C", "S"]:
                    kinds.append("disulfide")
            if (len(heavy) == 3 and sum(n.GetSymbol() == "C" for n in heavy) == 2
                    and sum(n.GetSymbol() == "O" and molecule.GetBondBetweenAtoms(index, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE for n in heavy) == 1):
                kinds.append("sulfoxide")
            if kinds:
                target.metadata["sulfur_pi_types"] = kinds
        if _tetrel_donor_capacity(source):
            target.metadata["tetrel_root_atom_ids"] = sorted(
                mapping[n.GetIdx()] for n in heavy if n.GetSymbol() in {"C", "N", "O", "S"} and n.GetIdx() in mapping)


def rdkit_atom_annotations(molecule) -> dict[int, AtomAnnotation]:
    """Return Chemur's RDKit-derived per-heavy-atom chemistry annotations."""
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


def _acceptor_indices(molecule) -> set[int]:
    """RDKit Lipinski acceptors, independent of explicit hydrogen representation.

    Matching on a heavy-atom copy is essential: explicit H neighbours can satisfy
    the recursive acid exclusion in HAcceptorSmarts differently. The original
    molecule (including coordinates and isotopes) is never changed.
    """
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        atom.SetIntProp("_chemur_original_index", atom.GetIdx())
    parameters = Chem.RemoveHsParameters()
    parameters.removeIsotopes = True
    copy = Chem.RemoveHs(copy, parameters)
    return {
        copy.GetAtomWithIdx(match[0]).GetIntProp("_chemur_original_index")
        for match in copy.GetSubstructMatches(Lipinski.HAcceptorSmarts)
        if copy.GetAtomWithIdx(match[0]).GetFormalCharge() <= 0
    }


def _total_hydrogen_count(atom) -> int:
    try:
        return int(atom.GetTotalNumHs(True))
    except TypeError:
        return int(atom.GetTotalNumHs())
