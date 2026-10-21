"""Deposited polymer membership and sequence-backed linkage chemistry.

A component dictionary describes a monomer, not its molecular role or its final
linked chemical state. Neither role nor an unusual linkage is guessed by distance.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def source_polymer_metadata(path, structure):
    import gemmi
    from .connectivity import _rows, _value
    result = {"entities": {}, "components": {}, "parents": {}, "entry_id": None, "source_residues": [],
              "is_cif": Path(path).suffix.lower() in {".cif", ".mmcif"}}
    for entity in structure.entities:
        kind = str(entity.polymer_type).split(".")[-1].lower()
        role = "protein" if "peptide" in kind else "nucleotide" if any(x in kind for x in ("dna", "rna", "hybrid")) else None
        result["entities"][entity.name] = (role, kind)
    if Path(path).suffix.lower() in {".cif", ".mmcif"}:
        for block in gemmi.cif.read(str(path)):
            result["entry_id"] = _value(block.find_value("_entry.id")) or result["entry_id"]
            for row in _rows(block, "_chem_comp."):
                result["components"][str(row.get("id", "")).upper()] = {
                    "ccd_type": _value(row.get("type")),
                    "parent_component_id": _value(row.get("mon_nstd_parent_comp_id")),
                }
            for row in _rows(block, "_pdbx_struct_mod_residue."):
                key = (row.get("auth_asym_id"), str(row.get("auth_seq_id")), row.get("auth_comp_id"))
                result["parents"][key] = _value(row.get("parent_comp_id"))
            result["source_residues"].extend(_rows(block, "_chemur_source_residue."))
    else:
        for line in Path(path).read_text(errors="replace").splitlines():
            if line.startswith("HEADER"):
                result["entry_id"] = line[62:66].strip() or None
            if line.startswith("MODRES"):
                key = (line[16:17].strip(), line[18:22].strip() + line[22:23].strip(), line[12:15].strip())
                result["parents"][key] = line[24:27].strip() or None
    return result


def residue_polymer_role(source, residue, chain_id, residue_id, fallback):
    """Return role plus provenance before ligand filtering occurs."""
    name = residue.name.strip().upper()
    metadata = deepcopy(source["components"].get(name, {}))
    parent = source["parents"].get((chain_id, residue_id, name)) or metadata.get("parent_component_id")
    if parent:
        metadata["parent_component_id"] = parent
    role, kind = source["entities"].get(residue.entity_id, (None, None))
    is_polymer = str(residue.entity_type).endswith(".Polymer")
    if not role and source["parents"].get((chain_id, residue_id, name)) and not source["is_cif"]:
        from .parser import AMINO_ACIDS, NUCLEOTIDES
        role = "protein" if parent in AMINO_ACIDS else "nucleotide" if parent in NUCLEOTIDES else None
        is_polymer = bool(role)
        kind = "modified_residue_record"
    if role and is_polymer:
        metadata.update(source_polymer_type=kind, source_polymer_membership="deposited_entity_or_modres")
        if residue.label_seq is not None:
            metadata["source_polymer_sequence_index"] = int(residue.label_seq)
        from .parser import AMINO_ACIDS, NUCLEOTIDES
        standard = AMINO_ACIDS if role == "protein" else NUCLEOTIDES
        metadata["modified_polymer"] = name not in standard
    else:
        role = fallback
        if source["is_cif"] and str(residue.entity_type).endswith(".NonPolymer") and fallback in {"protein", "nucleotide"}:
            # An independently deposited amino acid or nucleoside is a ligand,
            # even when its residue name also occurs in a polymer template.
            role = "ligand"
            metadata["source_polymer_membership"] = "deposited_nonpolymer_entity"
    if source.get("entry_id"):
        metadata["source_entry_id"] = source["entry_id"].upper()
    _attach_source_residue_provenance(source, residue, chain_id, residue_id, metadata)
    return role, metadata


def _attach_source_residue_provenance(source, residue, chain_id, residue_id, metadata):
    """Resolve snapshot provenance without changing deposited molecular role.

    ChimeraX can combine several original sources in one CIF entry. Entry-level
    identity then belongs to the snapshot, while overrides must bind to the
    original source of each particular residue. This category grants no new
    chemistry or polymer membership and never guesses through ambiguous rows.
    """
    from .connectivity import _value
    expected = {"label_asym_id": _value(residue.subchain),
                "label_seq_id": str(residue.label_seq) if residue.label_seq is not None else None,
                "auth_asym_id": chain_id, "auth_seq_id": residue_id,
                "comp_id": residue.name.strip().upper()}
    matches = []
    for row in source.get("source_residues", ()):
        if not all(key in row for key in expected):
            continue
        candidate = {key: _value(row.get(key)) for key in expected}
        insertion = _value(row.get("pdbx_PDB_ins_code"))
        if insertion and candidate["auth_seq_id"] and not candidate["auth_seq_id"].endswith(insertion):
            candidate["auth_seq_id"] += insertion
        if candidate == expected:
            matches.append(row)
    identities = {(_value(row.get("source_entry_id")), _value(row.get("source_polymer_type")),
                   _value(row.get("source_polymer_sequence_index"))) for row in matches}
    if len(identities) > 1:
        metadata["source_identity_diagnostics"] = ["Conflicting source-residue provenance in coordinate snapshot"]
        metadata.pop("source_entry_id", None)
    elif identities:
        entry_id, polymer_type, sequence_index = next(iter(identities))
        if entry_id:
            metadata["source_entry_id"] = entry_id.upper()
        if polymer_type:
            metadata["original_source_polymer_type"] = polymer_type
        if sequence_index is not None and metadata.get("source_polymer_type"):
            try:
                index = int(sequence_index)
                if index < 1:
                    raise ValueError
            except ValueError:
                metadata.setdefault("source_identity_diagnostics", []).append("Invalid original polymer sequence index")
            else:
                metadata["source_polymer_sequence_index"] = index
        metadata["source_identity_origin"] = "chemur_source_residue"


def polymer_sequence_links(atoms, components):
    """Known peptide/phosphodiester link transformations for consecutive sites.

    A connection requires consecutive deposited sequence positions and present
    named endpoints. Missing residues never become a guessed cross-gap bond.
    """
    by_id = {a.atom_id: a for a in atoms}
    chains = {}
    for component in components:
        index = component.metadata.get("source_polymer_sequence_index")
        if index is None or component.molecule_type not in {"protein", "nucleotide"}:
            continue
        member = [by_id[aid] for aid in component.atom_ids]
        if not member:
            continue
        first = member[0]
        key = (first.metadata.get("source_model"), first.metadata.get("source_entity_id"),
               first.metadata.get("source_label_asym_id"), component.chain_id, component.molecule_type)
        chains.setdefault(key, {}).setdefault(index, []).append(component)
    links = []
    for chain in chains.values():
        for index, choices in sorted(chain.items()):
            following = chain.get(index + 1, [])
            if len(choices) != 1 or len(following) != 1:
                continue
            left, right = choices[0], following[0]
            names = ("C", "N") if left.molecule_type == "protein" else ("O3'", "P")
            ends = [[by_id[aid] for aid in c.atom_ids if by_id[aid].name == name] for c, name in zip((left, right), names)]
            if any(len(end) != 1 for end in ends):
                continue
            links.append({"component_ids": [left.component_id, right.component_id],
                          "atom_ids": [end[0].atom_id for end in ends], "order": 1.0,
                          "kind": "covalent", "source": "polymer_sequence",
                          "polymer_linkage": "peptide" if names == ("C", "N") else "phosphodiester",
                          "leaving_atoms": [], "provenance": [{"source": "deposited_polymer_sequence", "positions": [index, index + 1]}]})
    return links


def automatic_polymer_product(members, atoms, components, fetch_ccd, timeout, cache):
    """Build a bounded linked product from named monomers and known link rules.

    Three bonds of real context are retained around rewritten sites, expanded to
    whole source residues. Remaining cuts carry mapped boundary stubs rather
    than invented atom caps. Unusual links require an explicit product override.
    """
    from rdkit import Chem
    from .assemblies import _template, _named_molecule, _links, _bond_type, _validate_context, _source_bond_orders
    from .errors import LigandTemplateError
    by_id = {a.atom_id: a for a in atoms}
    seed_ids = {aid for cid in members for aid in components[cid].atom_ids if by_id[aid].element != "H"}
    updates = {aid for aid in seed_ids if by_id[aid].molecule_type in {"protein", "nucleotide"}}
    all_links = _links(list(components.values()))
    for link in all_links:
        if set(members) & set(link.get("component_ids", ())) and not link.get("polymer_linkage"):
            updates.update(aid for aid in link.get("atom_ids", ()) if aid in by_id and by_id[aid].molecule_type != "ligand")
    local = seed_ids | updates
    frontier = set(local)
    for _ in range(3):
        frontier = {n for aid in frontier for n in by_id[aid].bonds if n in by_id and by_id[n].element != "H"} - local
        local.update(frontier)
    source_orders = _source_bond_orders(list(components.values()), atoms)
    unsaturated = {aid for pair, order in source_orders.items() if order > 1 for aid in pair}
    changed = True
    while changed:
        previous = set(local)
        for aid in list(local):
            for neighbor in by_id[aid].bonds:
                if neighbor not in by_id or by_id[neighbor].element == "H":
                    continue
                pair = (by_id[aid], by_id[neighbor])
                if (source_orders.get(frozenset((aid, neighbor)), 0) > 1
                        or any(a.is_aromatic or a.element == "P" for a in pair)
                        or (aid in unsaturated and (neighbor in unsaturated or by_id[neighbor].element in {"N", "O", "S"}))
                        or (neighbor in unsaturated and by_id[aid].element in {"N", "O", "S"})):
                    local.add(neighbor)
        changed = local != previous
    local_components = sorted({by_id[aid].component_id for aid in local})
    links = [l for l in all_links if set(local_components) & set(l.get("component_ids", ()))]
    combined = None
    for cid in local_components:
        component = components[cid]
        template = _template(component, fetch_ccd, timeout, cache)
        # Also retain completed chemistry as source evidence for boundary checks.
        component.metadata["chemical_template"] = deepcopy(template)
        monomer = _named_molecule(template, component)
        combined = monomer if combined is None else Chem.CombineMols(combined, monomer)
    graph = Chem.RWMol(combined)
    named = {(a.GetProp("_chemur_component"), a.GetProp("_chemur_name")): a.GetIdx() for a in graph.GetAtoms()}
    deposited = {(a.component_id, a.name): a.atom_id for a in atoms if a.component_id in local_components}
    if len(deposited) != sum(a.component_id in local_components for a in atoms):
        raise LigandTemplateError("Ambiguous polymer atom identity (model or alternate location)")
    removed, boundaries = set(), {}

    def remove_atom(index):
        atom = graph.GetAtomWithIdx(index)
        key = (atom.GetProp("_chemur_component"), atom.GetProp("_chemur_name"))
        if key in deposited and atom.GetAtomicNum() != 1:
            raise LigandTemplateError(f"Polymer leaving group conflicts with deposited heavy atom {key}")
        removed.add(index)
        removed.update(n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)

    for link in links:
        if link.get("error"):
            raise LigandTemplateError(link["error"])
        ids = link.get("atom_ids", ())
        if len(ids) != 2 or link.get("order") is None:
            raise LigandTemplateError("Nonstandard polymer linkage requires an explicit product override")
        endpoints = [by_id[aid] for aid in ids]
        kind = link.get("polymer_linkage")
        for endpoint in endpoints:
            key = (endpoint.component_id, endpoint.name)
            if key not in named:
                continue
            index = named[key]
            atom = graph.GetAtomWithIdx(index)
            if not kind:
                # Named leaving groups (if any) and an explicit order constitute
                # a declared transformation, never a proximity-based guess.
                for ref in link.get("leaving_atoms", ()):
                    if ref["component_id"] == endpoint.component_id:
                        leaving = named.get((ref["component_id"], ref["atom_name"]))
                        if leaving is None:
                            raise LigandTemplateError("Declared polymer leaving atom is absent")
                        remove_atom(leaving)
                continue
            if (kind == "peptide" and endpoint.name == "C") or (kind == "phosphodiester" and endpoint.name == "P"):
                allowed = {"OXT"} if kind == "peptide" else {"OP3", "O3P"}
                candidates = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O" and n.GetProp("_chemur_name") in allowed]
                if len(candidates) != 1:
                    raise LigandTemplateError(f"Polymer linkage has no unique named leaving oxygen at {key}")
                remove_atom(candidates[0].GetIdx())
            else:
                hydrogens = sorted((n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1), key=lambda n: n.GetProp("_chemur_name"))
                if hydrogens:
                    remove_atom(hydrogens[0].GetIdx())
                elif atom.GetNumExplicitHs():
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
                elif atom.GetNumImplicitHs():
                    # Sanitization recomputes the one fewer implicit H after linking.
                    pass
                else:
                    raise LigandTemplateError(f"Polymer linkage has no removable donor hydrogen at {key}")
        present = [(endpoint, named.get((endpoint.component_id, endpoint.name))) for endpoint in endpoints]
        if all(index is not None for _, index in present):
            graph.AddBond(present[0][1], present[1][1], _bond_type(link["order"]))
        else:
            real, index = next(pair for pair in present if pair[1] is not None)
            omitted = next(endpoint for endpoint, idx in present if idx is None)
            stub = Chem.Atom(0)
            stub.SetIntProp("_chemur_boundary", omitted.atom_id)
            stub_id = graph.AddAtom(stub)
            graph.AddBond(index, stub_id, Chem.BondType.SINGLE)
    for key, index in named.items():
        atom = graph.GetAtomWithIdx(index)
        if atom.GetAtomicNum() != 1 and key not in deposited and index not in removed:
            raise LigandTemplateError(f"Unexplained missing template atom {key[0]}:{key[1]}")
    if any(by_id[aid].element != "H" and key not in named for key, aid in deposited.items()):
        raise LigandTemplateError("Deposited polymer atom is missing from named chemistry")
    for index in sorted(removed, reverse=True):
        graph.RemoveAtom(index)
    product = graph.GetMol()
    try:
        Chem.SanitizeMol(product)
    except (ValueError, RuntimeError) as exc:
        raise LigandTemplateError(f"Linked polymer product has invalid chemistry: {exc}") from exc
    if any(a.GetNumRadicalElectrons() for a in product.GetAtoms()):
        raise LigandTemplateError("Linked polymer product has unresolved radical chemistry")
    # Whole monomers establish chemistry; retain only the required real context
    # and complete rings/conjugated groups. Cutting at residue boundaries would
    # incorrectly cut peptide amides and force inclusion of an entire protein.
    retained = {a.GetIdx() for a in product.GetAtoms() if a.GetAtomicNum() > 1 and
                deposited[(a.GetProp("_chemur_component"), a.GetProp("_chemur_name"))] in local}
    changed = True
    while changed:
        previous = set(retained)
        for ring in product.GetRingInfo().AtomRings():
            if retained.intersection(ring):
                retained.update(ring)
        for bond in product.GetBonds():
            pair = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            if retained.intersection(pair) and (bond.GetIsConjugated() or bond.GetBondTypeAsDouble() > 1
                                                or any(product.GetAtomWithIdx(i).GetIsAromatic() for i in pair)
                                                or any(product.GetAtomWithIdx(i).GetSymbol() == "P" for i in pair)):
                retained.update(pair)
        changed = retained != previous
    trimmed = Chem.RWMol(product)
    for bond in product.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (first in retained) == (second in retained):
            continue
        kept, omitted = (first, second) if first in retained else (second, first)
        other = product.GetAtomWithIdx(omitted)
        if other.GetAtomicNum() <= 1:
            retained.add(omitted)
            continue
        aid = deposited[(other.GetProp("_chemur_component"), other.GetProp("_chemur_name"))]
        dummy = Chem.Atom(0)
        dummy.SetIntProp("_chemur_boundary", aid)
        stub = trimmed.AddAtom(dummy)
        trimmed.AddBond(kept, stub, bond.GetBondType())
    for index in sorted(set(range(product.GetNumAtoms())) - retained, reverse=True):
        trimmed.RemoveAtom(index)
    product = trimmed.GetMol()
    Chem.SanitizeMol(product)
    mapping = {a.GetIdx(): deposited[(a.GetProp("_chemur_component"), a.GetProp("_chemur_name"))]
               for a in product.GetAtoms() if a.GetAtomicNum() > 1}
    boundaries = {a.GetIdx(): a.GetIntProp("_chemur_boundary") for a in product.GetAtoms() if a.GetAtomicNum() == 0}
    _validate_context(product, mapping, updates, boundaries, atoms, list(components.values()))
    return product, mapping, updates, boundaries
