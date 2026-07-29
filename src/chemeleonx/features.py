from __future__ import annotations

from .core import centroid, plane_normal
from .models import AtomRecord, ComponentRecord, FeatureRecord

PROTEIN_RING_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

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
        normal = None
        if feature_type == "ring" and len(feature_atoms) >= 3:
            normal = plane_normal([atom.coord for atom in feature_atoms])
        features.append(
            FeatureRecord(
                feature_id=next_id,
                feature_type=feature_type,
                component_id=component_id,
                atom_ids=atom_ids,
                center=center,
                capacity=capacity,
                normal=normal,
                metadata=metadata or {},
            )
        )
        next_id += 1

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

        if component.molecule_type == "protein":
            _add_protein_charge_features(component, atom_ids_by_name, add_feature)
            ring_names = PROTEIN_RING_ATOMS.get(component.name)
            if ring_names:
                ring_ids = tuple(
                    atom_ids_by_name[name] for name in ring_names if name in atom_ids_by_name
                )
                if len(ring_ids) >= 5:
                    add_feature("ring", component.component_id, ring_ids, metadata={"source": "protein_template"})

        elif component.molecule_type == "nucleotide":
            _add_nucleotide_charge_features(component, component_atoms, add_feature)
            base_name = component.name
            ring_names = NUCLEOTIDE_RING_ATOMS.get(base_name)
            if ring_names:
                ring_ids = tuple(
                    atom_ids_by_name[name] for name in ring_names if name in atom_ids_by_name
                )
                if len(ring_ids) >= 5:
                    add_feature("ring", component.component_id, ring_ids, metadata={"source": "nucleotide_template"})

        elif component.molecule_type == "ligand":
            for ring in component.metadata.get("template_rings", []):
                ring_ids = tuple(int(atom_id) for atom_id in ring)
                if len(ring_ids) >= 3:
                    add_feature("ring", component.component_id, ring_ids, metadata={"source": "smiles"})

    return features


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
