from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .chemistry import SIGMA_HOLE_ACTIVATING_ELEMENTS, _chalcogen_sigma_hole_count
from .core import distance
from .errors import DependencyMissingError, StructureParseError
from .models import AtomRecord, ComponentRecord

AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

NUCLEOTIDES = {
    "A",
    "C",
    "G",
    "U",
    "T",
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
    "ADE",
    "CYT",
    "GUA",
    "URA",
    "THY",
}

# Crystallographic (HOH/WAT/DOD) plus common MD water residue names
# (GROMACS SOL; CHARMM/AMBER TIP*/SPC*/T*P/WAT).
SOLVENT = {
    "HOH",
    "WAT",
    "DOD",
    "SOL",
    "TIP",
    "TIP2",
    "TIP3",
    "TIP4",
    "TIP5",
    "TIP3P",
    "TIP4P",
    "T3P",
    "T4P",
    "SPC",
    "SPCE",
    "SW",
}
IONS = {
    "AL",
    "BA",
    "CA",
    "CD",
    "CO",
    "CU",
    "FE",
    "K",
    "LI",
    "MG",
    "MN",
    "NA",
    "NI",
    "SR",
    "ZN",
    "CL",
    "BR",
    "IOD",
    # Common MD force-field ion residue names (GROMACS/CHARMM/AMBER).
    "SOD",
    "POT",
    "CLA",
    "CAL",
    "CES",
    "MG2",
    "ZN2",
    "NA+",
    "CL-",
    "K+",
    "ION",
}

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "P": 1.07,
    "S": 1.05,
    "SE": 1.20,
    "F": 0.57,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}

# Chalcogens that act as sigma-hole donors. Oxygen is excluded: its sigma-hole is
# negligible at biological polarisation.
CHALCOGEN_ELEMENTS = {"S", "Se"}

# Substituents electron-withdrawing enough to open a sigma-hole on a bonded carbon.
TETREL_WITHDRAWING_ELEMENTS = {"N", "O", "S", "Se", "P", "F", "Cl", "Br", "I"}

# Longest credible C=O bond. Used as an sp2 test on the structure-only path, where
# no bond orders are available.
CARBONYL_BOND_MAX = 1.30

# Deuterium and tritium are chemically hydrogen and are normalised to it. This is not
# cosmetic: neutron structures are the only ones that reliably model hydrogen positions,
# and they put D on every *exchangeable* site -- in 1LZN, all 241 N-H and all 237 O-H are
# deuterons while C-H stays protium. Keying on "H" alone therefore found zero hydrogen
# bonds in a neutron structure and reported it as missing_donor_hydrogen_geometry rather
# than as an error. The original label is kept in AtomRecord.metadata["isotope"].
HYDROGEN_ISOTOPES = {"D", "T"}


def _aromatic_ring_atom_names() -> dict[str, frozenset[str]]:
    """Aromatic ring atom names per residue, derived from the feature ring templates.

    Built from the same tables the ring features use so the two cannot drift.
    """
    from .features import NUCLEOTIDE_RING_ATOMS, PROTEIN_RING_ATOMS

    names: dict[str, set[str]] = {}
    for residue, rings in PROTEIN_RING_ATOMS.items():
        for ring in rings:
            names.setdefault(residue, set()).update(ring)
    for residue, ring in NUCLEOTIDE_RING_ATOMS.items():
        names.setdefault(residue, set()).update(ring)
    return {residue: frozenset(atom_names) for residue, atom_names in names.items()}


AROMATIC_RING_ATOM_NAMES = _aromatic_ring_atom_names()


def parse_structure(
    path: str | Path,
    *,
    include_ligands: bool = True,
) -> tuple[list[AtomRecord], list[ComponentRecord]]:
    try:
        import gemmi
    except ImportError as exc:
        raise DependencyMissingError(
            "gemmi is required to parse PDB/mmCIF files. Install the package with "
            "`pip install gemmi` or install this project with its runtime dependencies."
        ) from exc

    path = Path(path)
    if not path.exists():
        raise StructureParseError(f"Structure file does not exist: {path}")

    try:
        structure = gemmi.read_structure(str(path))
    except Exception as exc:
        raise StructureParseError(f"Could not parse structure file {path}: {exc}") from exc

    atoms: list[AtomRecord] = []
    components_by_id: dict[str, ComponentRecord] = {}

    atom_id = 0
    for model in structure:
        for chain in model:
            chain_id = chain.name or "_"
            for residue in chain:
                residue_name = residue.name.strip().upper()
                residue_id = _residue_id(residue)
                component_id = f"{chain_id}:{residue_name}:{residue_id}"
                molecule_type = classify_component(residue_name)
                if molecule_type == "ligand" and not include_ligands:
                    continue
                component_atom_ids: list[int] = []
                for atom in residue:
                    if atom.element.name == "":
                        continue
                    raw_element = atom.element.name.strip().upper()
                    element = normalize_element(raw_element)
                    coord = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                    record = AtomRecord(
                        atom_id=atom_id,
                        name=atom.name.strip(),
                        element=element,
                        coord=coord,
                        residue_name=residue_name,
                        residue_id=residue_id,
                        chain_id=chain_id,
                        component_id=component_id,
                        molecule_type=molecule_type,
                        serial=getattr(atom, "serial", None),
                        # element is normalised to H for D/T, so keep what was deposited.
                        metadata=(
                            {"isotope": raw_element}
                            if raw_element in HYDROGEN_ISOTOPES
                            else {}
                        ),
                    )
                    atoms.append(record)
                    component_atom_ids.append(atom_id)
                    atom_id += 1

                if component_atom_ids:
                    components_by_id[component_id] = ComponentRecord(
                        component_id=component_id,
                        name=residue_name,
                        chain_id=chain_id,
                        residue_id=residue_id,
                        molecule_type=molecule_type,
                        atom_ids=tuple(component_atom_ids),
                    )

    if not atoms:
        raise StructureParseError(f"No atoms were parsed from structure file {path}")

    _assign_inferred_bonds(atoms, components_by_id.values())
    _assign_default_atom_chemistry(atoms)
    return atoms, list(components_by_id.values())


def parse_embedded_ligand_smiles(path: str | Path) -> dict[str, str]:
    """SMILES embedded in an mmCIF's chem_comp records, keyed by component id (upper).

    ``gemmi.read_structure`` (used by :func:`parse_structure`) drops the chemistry
    dictionary, so this reads the raw CIF separately. AlphaFold3 writes the SMILES
    directly as ``_chem_comp.pdbx_smiles``; wwPDB files instead carry a
    ``_pdbx_chem_comp_descriptor`` loop (the same form CCD files use, so we reuse
    :func:`chemur.ccd._choose_smiles`). Returns ``{}`` for PDB inputs, when no SMILES
    is present, or on any parse error, so callers can fall back to a CCD lookup.
    """
    path = Path(path)
    if path.suffix.lower() not in {".cif", ".mmcif"}:
        return {}
    try:
        import gemmi

        from .ccd import _choose_smiles
    except ImportError:
        return {}

    try:
        doc = gemmi.cif.read(str(path))
    except Exception:
        return {}

    smiles: dict[str, str] = {}
    try:
        for block in doc:
            # AlphaFold3 form: a single SMILES per component.
            for row in block.find("_chem_comp.", ["id", "pdbx_smiles"]):
                comp_id = gemmi.cif.as_string(row[0]).strip().upper()
                value = gemmi.cif.as_string(row[1]).strip()
                if comp_id and value and value not in {"?", "."}:
                    smiles.setdefault(comp_id, value)

            # wwPDB form: a descriptor loop with several SMILES variants per component.
            descriptors_by_comp: dict[str, list[dict[str, str]]] = {}
            for row in block.find(
                "_pdbx_chem_comp_descriptor.",
                ["comp_id", "type", "program", "descriptor"],
            ):
                comp_id = gemmi.cif.as_string(row[0]).strip().upper()
                descriptors_by_comp.setdefault(comp_id, []).append(
                    {
                        "type": gemmi.cif.as_string(row[1]),
                        "program": gemmi.cif.as_string(row[2]),
                        "descriptor": gemmi.cif.as_string(row[3]),
                    }
                )
            for comp_id, descriptors in descriptors_by_comp.items():
                if comp_id in smiles:
                    continue
                chosen = _choose_smiles(descriptors)
                if chosen:
                    smiles[comp_id] = chosen
    except Exception:
        return {}

    return smiles


def classify_component(residue_name: str) -> str:
    if residue_name in AMINO_ACIDS:
        return "protein"
    if residue_name in NUCLEOTIDES:
        return "nucleotide"
    if residue_name in SOLVENT:
        return "solvent"
    if residue_name in IONS:
        return "ion"
    return "ligand"


def normalize_element(element: str) -> str:
    element = element.strip().upper()
    if element in HYDROGEN_ISOTOPES:
        return "H"
    if len(element) > 1:
        return element[0] + element[1:].lower()
    return element


def infer_bonds_for_component(atoms: list[AtomRecord]) -> dict[int, tuple[int, ...]]:
    bonds: dict[int, set[int]] = defaultdict(set)
    for i, atom_i in enumerate(atoms):
        for atom_j in atoms[i + 1 :]:
            cutoff = _bond_cutoff(atom_i.element, atom_j.element)
            dist = distance(atom_i.coord, atom_j.coord)
            if 0.35 <= dist <= cutoff:
                bonds[atom_i.atom_id].add(atom_j.atom_id)
                bonds[atom_j.atom_id].add(atom_i.atom_id)
    return {atom_id: tuple(sorted(neighbors)) for atom_id, neighbors in bonds.items()}


def _assign_inferred_bonds(
    atoms: list[AtomRecord],
    components: Iterable[ComponentRecord],
) -> None:
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    for component in components:
        component_atoms = [atoms_by_id[atom_id] for atom_id in component.atom_ids]
        inferred = infer_bonds_for_component(component_atoms)
        for atom_id, neighbors in inferred.items():
            atoms_by_id[atom_id].bonds = neighbors


def _assign_default_atom_chemistry(atoms: list[AtomRecord]) -> None:
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    for atom in atoms:
        # Residue templates are the only aromaticity signal available without bond
        # orders, and sp2 vs sp3 decides whether a carbon can carry a sigma-hole.
        atom.is_aromatic = atom.name.strip() in AROMATIC_RING_ATOM_NAMES.get(
            atom.residue_name.strip().upper(), ()
        )
        heavy_neighbors = [
            atoms_by_id[neighbor_id]
            for neighbor_id in atom.bonds
            if neighbor_id in atoms_by_id and atoms_by_id[neighbor_id].element != "H"
        ]
        hydrogens = sum(
            1 for neighbor_id in atom.bonds if atoms_by_id[neighbor_id].element == "H"
        )
        atom.hydrogen_count = hydrogens
        if atom.element in {"N", "O", "S", "Se"}:
            atom.acceptor_capacity = {"O": 2, "N": 1, "S": 2, "Se": 2}.get(atom.element, 0)
            atom.donor_capacity = max(hydrogens, 1 if atom.element == "N" else 0)
            if atom.element in CHALCOGEN_ELEMENTS:
                atom.chalcogen_donor_capacity = _chalcogen_donor_capacity(atom, heavy_neighbors)
                atom.sigma_hole_activated = _sigma_hole_activated(atom, heavy_neighbors)
        elif atom.element == "C":
            atom.donor_capacity = hydrogens
            atom.carbon_donor_capacity = max(hydrogens, 1)
            atom.tetrel_donor_capacity = _tetrel_donor_capacity(atom, heavy_neighbors)
            atom.is_hydrophobe = True
        elif atom.element in {"Cl", "Br", "I"}:
            atom.halogen_donor_capacity = 1
            atom.is_hydrophobe = True
        elif atom.element == "P":
            atom.acceptor_capacity = 0


def _chalcogen_donor_capacity(atom: AtomRecord, heavy_neighbors: list[AtomRecord]) -> int:
    """A chalcogen carries a sigma-hole on the extension of each R-Ch bond."""
    return _chalcogen_sigma_hole_count(atom.formal_charge, len(heavy_neighbors))


def _sigma_hole_activated(atom: AtomRecord, heavy_neighbors: list[AtomRecord]) -> bool:
    """Structure-path counterpart of chemistry._sigma_hole_activated.

    The conjugated-C=S clause used there needs bond orders, which a coordinate file
    does not carry. That costs nothing here: the only sulfur in the standard residues
    is MET SD, CYS SG and cystine, none of which is activated, so this correctly
    returns False for every protein sulfur. Ligands reach the RDKit predicate instead,
    where thioamides and thioureas are recognised properly.
    """
    if atom.formal_charge > 0:
        return True
    if atom.is_aromatic:
        return True
    return any(
        neighbor.element in SIGMA_HOLE_ACTIVATING_ELEMENTS for neighbor in heavy_neighbors
    )


def _tetrel_donor_capacity(atom: AtomRecord, heavy_neighbors: list[AtomRecord]) -> int:
    """An sp3 carbon polarised by an electron-withdrawing substituent.

    The sigma-hole sits trans to that substituent, so the count is the number of
    withdrawing neighbours; the R-C...A angle cutoff picks which one is in play.
    Carbonyl and aromatic carbons are excluded -- their sp2 lobes are not a
    sigma-hole, and a C=O bond length is the only sp2 signal available here.
    """
    if atom.is_aromatic:
        return 0
    if any(
        neighbor.element == "O" and distance(atom.coord, neighbor.coord) <= CARBONYL_BOND_MAX
        for neighbor in heavy_neighbors
    ):
        return 0
    return sum(
        1 for neighbor in heavy_neighbors if neighbor.element in TETREL_WITHDRAWING_ELEMENTS
    )


def _residue_id(residue) -> str:
    seqid = residue.seqid
    number = str(seqid.num) if seqid.num is not None else "?"
    insertion = seqid.icode.strip() if seqid.icode else ""
    return f"{number}{insertion}"


def _bond_cutoff(element_a: str, element_b: str) -> float:
    a = COVALENT_RADII.get(element_a.upper(), COVALENT_RADII.get(element_a, 0.77))
    b = COVALENT_RADII.get(element_b.upper(), COVALENT_RADII.get(element_b, 0.77))
    return a + b + 0.45
