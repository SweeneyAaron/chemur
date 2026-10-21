"""Conservative acceptor rules for standard coordinate-only residues."""
from __future__ import annotations

from .models import AtomRecord


PROTEIN_SIDECHAIN_ACCEPTORS = {
    "ASN": {"OD1"}, "GLN": {"OE1"},
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "CYS": {"SG"}, "MET": {"SD"},
}
PROTEIN_NONACCEPTOR_NITROGENS = {
    "ASN": {"ND2"}, "GLN": {"NE2"}, "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"}, "TRP": {"NE1"},
}
BASE_ACCEPTORS = {
    "A": {"N1", "N3", "N7"}, "C": {"N3", "O2"},
    "G": {"N3", "N7", "O6"}, "U": {"O2", "O4"},
    "T": {"O2", "O4"},
}
BASE_NONACCEPTORS = {
    "A": {"N6", "N9"}, "C": {"N1", "N4"}, "G": {"N1", "N2", "N9"},
    "U": {"N1", "N3"}, "T": {"N1", "N3"},
}
BASE_ALIASES = {
    "DA": "A", "ADE": "A", "DC": "C", "CYT": "C",
    "DG": "G", "GUA": "G", "DU": "U", "URA": "U",
    "DT": "T", "THY": "T",
}
SUGAR_PHOSPHATE_ACCEPTORS = {
    "O2'", "O3'", "O4'", "O5'", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
}


def _hydrogen_count(atom: AtomRecord, atoms_by_id: dict[int, AtomRecord]) -> int:
    return sum(atoms_by_id[idx].element == "H" for idx in atom.bonds if idx in atoms_by_id)


def residue_acceptor_capacity(
    atom: AtomRecord,
    atoms_by_id: dict[int, AtomRecord],
    component_atoms: list[AtomRecord],
) -> int:
    """Assign only supported sites; missing tautomer information is not a guess."""
    atom.metadata.pop("acceptor_unresolved_reason", None)
    atom.metadata["acceptor_source"] = "standard_residue"
    charge = int(atom.metadata.get("structure_formal_charge", atom.formal_charge))
    if charge > 0:
        return 0
    if atom.molecule_type == "solvent":
        return 2 if atom.element == "O" else 0
    name = atom.name.strip().upper().replace("*", "'")
    residue = atom.residue_name.strip().upper()
    if atom.molecule_type == "protein":
        if residue == "HIS" and name in {"ND1", "NE2"}:
            ring_nitrogens = [a for a in component_atoms if a.name.strip().upper() in {"ND1", "NE2"}]
            if any(int(a.metadata.get("structure_formal_charge", a.formal_charge)) > 0
                   for a in ring_nitrogens):
                return 0
            if _hydrogen_count(atom, atoms_by_id):
                return 0
            if len(ring_nitrogens) == 2 and any(_hydrogen_count(a, atoms_by_id) for a in ring_nitrogens):
                return 1
            atom.metadata["acceptor_unresolved_reason"] = "histidine_tautomer_unknown"
            return 0
        # Backbone and side-chain amide N, guanidinium, ammonium and indole N
        # never appear in this allowlist. Standard termini use the charged state.
        accepted = name in {"O", "OXT"} or name in PROTEIN_SIDECHAIN_ACCEPTORS.get(residue, ())
        if accepted and (name == "OXT" or residue in {"ASP", "GLU"}) and _hydrogen_count(atom, atoms_by_id):
            return 0
        if accepted:
            return {"O": 2, "S": 2}.get(atom.element, 0)
        known_nonacceptor = atom.element == "N" and (
            name == "N" or name in PROTEIN_NONACCEPTOR_NITROGENS.get(residue, ())
        )
        if not known_nonacceptor:
            atom.metadata["acceptor_unresolved_reason"] = "unsupported_residue_site"
        return 0
    if atom.molecule_type == "nucleotide":
        base = BASE_ALIASES.get(residue, residue)
        accepted = name in BASE_ACCEPTORS.get(base, ()) or name in SUGAR_PHOSPHATE_ACCEPTORS
        if accepted and (atom.element == "N" or name in {"OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}) and _hydrogen_count(atom, atoms_by_id):
            return 0
        if accepted:
            return {"N": 1, "O": 2}.get(atom.element, 0)
        if name not in BASE_NONACCEPTORS.get(base, ()):
            atom.metadata["acceptor_unresolved_reason"] = "unsupported_residue_site"
        return 0
    atom.metadata["acceptor_source"] = "unresolved"
    if atom.molecule_type == "ligand":
        atom.metadata["acceptor_unresolved_reason"] = "ligand_template_required"
    return 0
