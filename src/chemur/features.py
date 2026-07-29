from __future__ import annotations

from .core import centroid, distance, plane_fit
from .models import AtomRecord, ComponentRecord, FeatureRecord

# Each residue maps to a tuple of rings, not a single ring: TRP is an indole, and
# its pyrrole ring stacks and donates NE1-H just as the benzo ring does.
PROTEIN_RING_ATOMS = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "TRP": (
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        ("CG", "CD1", "NE1", "CE2", "CD2"),
    ),
}

# Saturated rings. These are perceived as ring features so they can stack, but are
# tagged aliphatic so they never masquerade as pi systems.
PROTEIN_ALIPHATIC_RING_ATOMS = {
    "PRO": (("N", "CA", "CB", "CG", "CD"),),
    "HYP": (("N", "CA", "CB", "CG", "CD"),),
}

NUCLEOTIDE_SUGAR_RING_ATOMS = ("C1'", "C2'", "C3'", "C4'", "O4'")

# (carbonyl carbon, carbonyl oxygen) name pairs. The backbone C=O is shared by every
# residue; the rest are side-chain amides and carboxylates. Bond orders are not
# available on the structure-only path, so these have to be named explicitly.
PROTEIN_CARBONYL_ATOMS = {
    "ASN": (("CG", "OD1"),),
    "GLN": (("CD", "OE1"),),
    "ASP": (("CG", "OD1"), ("CG", "OD2")),
    "GLU": (("CD", "OE1"), ("CD", "OE2")),
}
BACKBONE_CARBONYL_ATOMS = ("C", "O")

# (carbon, oxygen, nitrogen) name triples for side-chain amides.
PROTEIN_AMIDE_ATOMS = {
    "ASN": (("CG", "OD1", "ND2"),),
    "GLN": (("CD", "OE1", "NE2"),),
}

# Longest credible peptide C-N bond; used to confirm two residues are really linked
# rather than trusting residue numbering.
PEPTIDE_BOND_MAX = 1.5

NUCLEOTIDE_RING_ATOMS = {
    "A": ("N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"),
    "G": ("N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"),
    "DA": ("N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"),
    "DG": ("N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"),
    "C": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "U": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "T": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "DC": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "DT": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "DU": ("N1", "C2", "N3", "C4", "C5", "C6"),
}

METAL_ELEMENTS = {"Al", "Ba", "Ca", "Cd", "Co", "Cu", "Fe", "K", "Li", "Mg", "Mn", "Na", "Ni", "Sr", "Zn"}

# Feature types that get a fitted mean plane (normal + planarity), not just a centroid.
PLANAR_FEATURE_TYPES = {"ring", "amide", "carbonyl"}


def perceive_features(
    atoms: list[AtomRecord],
    components: list[ComponentRecord],
) -> list[FeatureRecord]:
    features: list[FeatureRecord] = []
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    next_id = 0

    def add_feature(
        feature_type: str,
        component_id: str,
        atom_ids: tuple[int, ...],
        capacity: int = 0,
        metadata: dict | None = None,
    ) -> None:
        nonlocal next_id
        feature_atoms = [atoms_by_id[atom_id] for atom_id in atom_ids]
        center = centroid([atom.coord for atom in feature_atoms])
        metadata = dict(metadata or {})
        normal = None
        if feature_type in PLANAR_FEATURE_TYPES and len(feature_atoms) >= 3:
            normal, planarity = plane_fit([atom.coord for atom in feature_atoms])
            # The scorers read metadata["planarity"]; before the mean-plane fit
            # there was nothing to write it from.
            metadata["planarity"] = planarity
        features.append(
            FeatureRecord(
                feature_id=next_id,
                feature_type=feature_type,
                component_id=component_id,
                atom_ids=atom_ids,
                center=center,
                capacity=capacity,
                normal=normal,
                metadata=metadata,
            )
        )
        next_id += 1

    def add_ring(
        component_id: str,
        atom_ids: tuple[int, ...],
        *,
        aromatic: bool,
        source: str,
        minimum: int = 5,
    ) -> None:
        if len(atom_ids) < minimum:
            return
        add_feature(
            "ring",
            component_id,
            atom_ids,
            metadata={"source": source, "aromatic": aromatic},
        )

    for atom in atoms:
        if atom.metadata.get("ignored_by_template"):
            continue
        if atom.element == "H":
            continue
        if atom.donor_capacity > 0:
            add_feature("donor", atom.component_id, (atom.atom_id,), atom.donor_capacity)
        if atom.acceptor_capacity > 0:
            add_feature("acceptor", atom.component_id, (atom.atom_id,), atom.acceptor_capacity)
        if atom.halogen_donor_capacity > 0:
            add_feature(
                "halogen_donor",
                atom.component_id,
                (atom.atom_id,),
                atom.halogen_donor_capacity,
            )
        if atom.chalcogen_donor_capacity > 0:
            add_feature(
                "chalcogen_donor",
                atom.component_id,
                (atom.atom_id,),
                atom.chalcogen_donor_capacity,
            )
        if atom.tetrel_donor_capacity > 0:
            add_feature(
                "tetrel_donor",
                atom.component_id,
                (atom.atom_id,),
                atom.tetrel_donor_capacity,
            )
        if atom.carbon_donor_capacity > 0:
            add_feature(
                "carbon_donor",
                atom.component_id,
                (atom.atom_id,),
                atom.carbon_donor_capacity,
            )
        if atom.formal_charge > 0:
            add_feature("cation", atom.component_id, (atom.atom_id,), metadata={"charge": atom.formal_charge})
        if atom.formal_charge < 0:
            add_feature("anion", atom.component_id, (atom.atom_id,), metadata={"charge": atom.formal_charge})
        if atom.is_hydrophobe:
            add_feature("hydrophobe", atom.component_id, (atom.atom_id,))
        if atom.molecule_type == "ion" and atom.element in METAL_ELEMENTS:
            add_feature("metal", atom.component_id, (atom.atom_id,))

    for component in components:
        component_atoms = [atoms_by_id[atom_id] for atom_id in component.atom_ids]
        atom_ids_by_name = {atom.name.strip(): atom.atom_id for atom in component_atoms}

        def named_ring(names: tuple[str, ...]) -> tuple[int, ...]:
            return tuple(atom_ids_by_name[name] for name in names if name in atom_ids_by_name)

        if component.molecule_type == "protein":
            _add_protein_charge_features(component, atom_ids_by_name, add_feature)
            _add_protein_carbonyl_features(
                component, atom_ids_by_name, atoms_by_id, add_feature
            )
            for ring_names in PROTEIN_RING_ATOMS.get(component.name, ()):
                add_ring(
                    component.component_id,
                    named_ring(ring_names),
                    aromatic=True,
                    source="protein_template",
                )
            for ring_names in PROTEIN_ALIPHATIC_RING_ATOMS.get(component.name, ()):
                add_ring(
                    component.component_id,
                    named_ring(ring_names),
                    aromatic=False,
                    source="protein_aliphatic_template",
                )

        elif component.molecule_type == "nucleotide":
            _add_nucleotide_charge_features(component, component_atoms, add_feature)
            base_ring_names = NUCLEOTIDE_RING_ATOMS.get(component.name)
            if base_ring_names:
                add_ring(
                    component.component_id,
                    named_ring(base_ring_names),
                    aromatic=True,
                    source="nucleotide_template",
                )
            add_ring(
                component.component_id,
                named_ring(NUCLEOTIDE_SUGAR_RING_ATOMS),
                aromatic=False,
                source="nucleotide_sugar_template",
            )

        elif component.molecule_type == "ligand":
            for ring in component.metadata.get("template_rings", []):
                ring_ids = tuple(int(atom_id) for atom_id in ring)
                add_ring(
                    component.component_id,
                    ring_ids,
                    # RDKit's ring info is the full SSSR, saturated rings included.
                    # Without this flag a cyclohexyl was indistinguishable from a phenyl.
                    aromatic=all(
                        atoms_by_id[atom_id].is_aromatic
                        for atom_id in ring_ids
                        if atom_id in atoms_by_id
                    ),
                    source="smiles",
                    minimum=3,
                )
            for carbon_id, oxygen_id in component.metadata.get("template_carbonyls", []):
                _add_carbonyl(
                    component.component_id,
                    int(carbon_id),
                    int(oxygen_id),
                    atoms_by_id,
                    add_feature,
                    source="smiles",
                )
            for carbon_id, oxygen_id, nitrogen_id in component.metadata.get(
                "template_amides", []
            ):
                _add_amide(
                    component.component_id,
                    int(carbon_id),
                    int(oxygen_id),
                    int(nitrogen_id),
                    atoms_by_id,
                    add_feature,
                    source="smiles",
                )

    _add_protein_amide_features(components, atoms_by_id, add_feature)

    return features


def _add_carbonyl(
    component_id: str,
    carbon_id: int,
    oxygen_id: int,
    atoms_by_id: dict[int, AtomRecord],
    add_feature,
    *,
    source: str,
) -> None:
    """Add a carbonyl feature spanning C=O plus the carbon's other heavy substituents.

    The substituents are included so the feature has a real sp2 plane -- two atoms
    alone would give plane_fit nothing to fit, and the n->pi* approach criterion is
    measured against that plane.
    """
    carbon = atoms_by_id.get(carbon_id)
    oxygen = atoms_by_id.get(oxygen_id)
    if carbon is None or oxygen is None:
        return
    substituents = tuple(
        neighbor_id
        for neighbor_id in carbon.bonds
        if neighbor_id != oxygen_id
        and neighbor_id in atoms_by_id
        and atoms_by_id[neighbor_id].element != "H"
    )
    add_feature(
        "carbonyl",
        component_id,
        (carbon_id, oxygen_id) + substituents,
        metadata={
            "source": source,
            "carbon_atom_id": carbon_id,
            "oxygen_atom_id": oxygen_id,
        },
    )


def _add_amide(
    component_id: str,
    carbon_id: int,
    oxygen_id: int,
    nitrogen_id: int,
    atoms_by_id: dict[int, AtomRecord],
    add_feature,
    *,
    source: str,
) -> None:
    if any(atom_id not in atoms_by_id for atom_id in (carbon_id, oxygen_id, nitrogen_id)):
        return
    add_feature(
        "amide",
        component_id,
        (carbon_id, oxygen_id, nitrogen_id),
        metadata={
            "source": source,
            "carbon_atom_id": carbon_id,
            "oxygen_atom_id": oxygen_id,
            "nitrogen_atom_id": nitrogen_id,
        },
    )


def _add_protein_carbonyl_features(
    component: ComponentRecord,
    atom_ids_by_name: dict[str, int],
    atoms_by_id: dict[int, AtomRecord],
    add_feature,
) -> None:
    pairs = (BACKBONE_CARBONYL_ATOMS,) + PROTEIN_CARBONYL_ATOMS.get(component.name, ())
    for carbon_name, oxygen_name in pairs:
        if carbon_name in atom_ids_by_name and oxygen_name in atom_ids_by_name:
            _add_carbonyl(
                component.component_id,
                atom_ids_by_name[carbon_name],
                atom_ids_by_name[oxygen_name],
                atoms_by_id,
                add_feature,
                source="protein_template",
            )


def _add_protein_amide_features(
    components: list[ComponentRecord],
    atoms_by_id: dict[int, AtomRecord],
    add_feature,
) -> None:
    """Side-chain amides, plus backbone peptide amides spanning residue pairs.

    A backbone amide is C(=O) of residue i with N of residue i+1, so it cannot be
    built from a single component -- and the parser only infers bonds within a
    component, so the link is confirmed by the C...N distance rather than by
    residue numbering, which insertion codes and chain breaks make unreliable.
    """
    names_by_component: dict[str, dict[str, int]] = {}
    protein_components: list[ComponentRecord] = []
    for component in components:
        if component.molecule_type != "protein":
            continue
        protein_components.append(component)
        names_by_component[component.component_id] = {
            atoms_by_id[atom_id].name.strip(): atom_id
            for atom_id in component.atom_ids
            if atom_id in atoms_by_id
        }

    for component in protein_components:
        atom_ids_by_name = names_by_component[component.component_id]
        for carbon_name, oxygen_name, nitrogen_name in PROTEIN_AMIDE_ATOMS.get(
            component.name, ()
        ):
            if all(
                name in atom_ids_by_name
                for name in (carbon_name, oxygen_name, nitrogen_name)
            ):
                _add_amide(
                    component.component_id,
                    atom_ids_by_name[carbon_name],
                    atom_ids_by_name[oxygen_name],
                    atom_ids_by_name[nitrogen_name],
                    atoms_by_id,
                    add_feature,
                    source="protein_template",
                )

    by_chain: dict[str, list[ComponentRecord]] = {}
    for component in protein_components:
        by_chain.setdefault(component.chain_id, []).append(component)

    for chain_components in by_chain.values():
        chain_components.sort(key=lambda component: _residue_sort_key(component.residue_id))
        for current, following in zip(chain_components, chain_components[1:]):
            carbon_id = names_by_component[current.component_id].get("C")
            oxygen_id = names_by_component[current.component_id].get("O")
            nitrogen_id = names_by_component[following.component_id].get("N")
            if carbon_id is None or oxygen_id is None or nitrogen_id is None:
                continue
            if distance(atoms_by_id[carbon_id].coord, atoms_by_id[nitrogen_id].coord) > PEPTIDE_BOND_MAX:
                continue
            _add_amide(
                current.component_id,
                carbon_id,
                oxygen_id,
                nitrogen_id,
                atoms_by_id,
                add_feature,
                source="protein_backbone",
            )


def _residue_sort_key(residue_id: str) -> tuple[int, str]:
    digits = ""
    for character in residue_id:
        if character.isdigit() or (character == "-" and not digits):
            digits += character
        else:
            break
    return (int(digits) if digits not in {"", "-"} else 0, residue_id)


def _add_protein_charge_features(component, atom_ids_by_name, add_feature) -> None:
    if component.name == "ARG":
        atom_names = ("CZ", "NE", "NH1", "NH2")
        atom_ids = tuple(atom_ids_by_name[name] for name in atom_names if name in atom_ids_by_name)
        if atom_ids:
            add_feature("cation", component.component_id, atom_ids, metadata={"source": "protein_arg"})
    elif component.name == "LYS" and "NZ" in atom_ids_by_name:
        add_feature("cation", component.component_id, (atom_ids_by_name["NZ"],), metadata={"source": "protein_lys"})
    elif component.name in {"ASP", "GLU"}:
        oxygen_ids = tuple(
            atom_id
            for name, atom_id in atom_ids_by_name.items()
            if name.startswith(("OD", "OE"))
        )
        if oxygen_ids:
            add_feature("anion", component.component_id, oxygen_ids, metadata={"source": "protein_carboxylate"})


def _add_nucleotide_charge_features(component, component_atoms, add_feature) -> None:
    phosphate_ids = tuple(
        atom.atom_id
        for atom in component_atoms
        if atom.name.startswith(("P", "OP", "O1P", "O2P", "O3P"))
    )
    if phosphate_ids:
        add_feature("anion", component.component_id, phosphate_ids, metadata={"source": "nucleotide_phosphate"})
