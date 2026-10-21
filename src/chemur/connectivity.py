"""Deposited chemical dictionaries and explicit covalent connectivity.

No cross-residue bond is inferred from coordinates here.  Source dictionaries
describe unlinked components; assembly preparation owns leaving-group chemistry.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


_ORDERS = {"sing": 1.0, "single": 1.0, "1": 1.0,
           "doub": 2.0, "double": 2.0, "2": 2.0,
           "trip": 3.0, "triple": 3.0, "3": 3.0,
           "arom": 1.5, "aromatic": 1.5, "delo": 1.5}

_COORDINATION_METALS = set("""
Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn
Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi
Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
""".split())


def _value(value):
    # Gemmi represents the CIF null marker '.' as False in category tables.
    return None if value is False or value in (None, "", ".", "?") else str(value)


def _rows(block, category):
    columns = block.get_mmcif_category(category)
    if not columns:
        return []
    return [dict(zip(columns, values)) for values in zip(*columns.values())]


def _order(value):
    return _ORDERS.get(str(value).lower())


def named_templates_from_cif(block, *, source: str = "embedded_cif") -> dict[str, dict]:
    """Extract named component graphs, retaining unknown charge as ``None``."""
    templates: dict[str, dict] = {}
    for row in _rows(block, "_chem_comp_atom."):
        comp = str(row.get("comp_id", "")).upper()
        name = _value(row.get("atom_id"))
        element = _value(row.get("type_symbol"))
        if not comp or not name or not element:
            continue
        template = templates.setdefault(comp, {"atoms": {}, "bonds": [], "source": source})
        raw_charge = _value(row.get("charge"))
        try:
            charge = int(raw_charge) if raw_charge is not None else None
        except ValueError:
            charge = None
            template["error"] = f"Invalid formal charge for {comp}:{name}"
        atom = {"element": "H" if element.upper() in {"D", "T"} else element.title(),
                "charge": charge,
                "aromatic": str(row.get("pdbx_aromatic_flag", "N")).upper() == "Y",
                "stereo": _value(row.get("pdbx_stereo_config")),
                "leaving": str(row.get("pdbx_leaving_atom_flag", "N")).upper() == "Y"}
        if name in template["atoms"] and template["atoms"][name] != atom:
            template["error"] = f"Conflicting atom dictionary entries for {comp}:{name}"
        template["atoms"][name] = atom
    for row in _rows(block, "_chem_comp_bond."):
        comp = str(row.get("comp_id", "")).upper()
        if comp not in templates:
            continue
        template = templates[comp]
        atom1, atom2 = _value(row.get("atom_id_1")), _value(row.get("atom_id_2"))
        order = _order(row.get("value_order"))
        if str(row.get("pdbx_aromatic_flag", "N")).upper() == "Y":
            order = 1.5
        if atom1 not in template["atoms"] or atom2 not in template["atoms"] or order is None:
            template["error"] = f"Unresolved bond dictionary entry for {comp}:{atom1}-{atom2}"
            continue
        bond = {"atom1": atom1, "atom2": atom2, "order": order}
        previous = next((item for item in template["bonds"]
                         if {item["atom1"], item["atom2"]} == {atom1, atom2}), None)
        if previous is not None:
            if previous["order"] != order:
                template["error"] = f"Conflicting bond dictionary entries for {comp}:{atom1}-{atom2}"
        else:
            template["bonds"].append(bond)
    from .ccd import _choose_smiles

    descriptors = defaultdict(list)
    for row in _rows(block, "_pdbx_chem_comp_descriptor."):
        descriptors[str(row.get("comp_id", "")).upper()].append(row)
    embedded = {str(row.get("id", "")).upper(): _value(row.get("pdbx_smiles"))
                for row in _rows(block, "_chem_comp.")}
    for comp, template in templates.items():
        smiles = embedded.get(comp) or _choose_smiles(descriptors[comp])
        if smiles:
            template["smiles"] = smiles
    return templates


def _endpoint(atoms, components, *, chain=None, residue=None, comp=None,
              name=None, altloc=None, label_asym=None, label_seq=None):
    """Resolve a deposited site, returning candidate components even if absent."""
    selected = []
    for component in components:
        if comp and component.name.upper() != str(comp).upper():
            continue
        if chain and component.chain_id != chain:
            continue
        if residue and component.residue_id != str(residue):
            continue
        member_atoms = [atom for atom in atoms if atom.component_id == component.component_id]
        if label_asym and not any(atom.metadata.get("source_label_asym_id") == label_asym
                                  for atom in member_atoms):
            continue
        if label_seq and not any(str(atom.metadata.get("source_label_seq_id")) == str(label_seq)
                                 for atom in member_atoms):
            continue
        selected.append(component.component_id)
    candidates = [atom for atom in atoms if atom.component_id in selected and atom.name == name
                  and (not altloc or atom.metadata.get("source_altloc") == altloc)]
    return selected, candidates


def _link_from_endpoints(endpoints, *, kind, order, source, record_id, error=None):
    component_ids = list(dict.fromkeys(cid for component_ids, _ in endpoints for cid in component_ids))
    link = {"component_ids": component_ids, "order": order, "leaving_atoms": [],
            "kind": kind, "source": source,
            "endpoints": [{"component_ids": cids,
                           "atom_names": sorted({atom.name for atom in candidates})}
                          for cids, candidates in endpoints],
            "provenance": [{"source": source, "record_id": str(record_id)}]}
    if all(len(candidates) == 1 for _, candidates in endpoints):
        link["atom_ids"] = [candidates[0].atom_id for _, candidates in endpoints]
        if len({candidates[0].metadata.get("source_model") for _, candidates in endpoints}) > 1:
            error = "Covalent endpoints belong to different coordinate models"
    else:
        error = error or "Missing or ambiguous covalent link endpoint (atom name, model or alternate location)"
    if error:
        link["error"] = error
    return link


def _cif_links(block, atoms, components):
    links = []
    for row in _rows(block, "_struct_conn."):
        kind = str(row.get("conn_type_id", "")).lower()
        if not (kind.startswith("covale") or kind == "disulf"):
            continue
        endpoints = []
        for n in (1, 2):
            p = f"ptnr{n}_"
            chain = _value(row.get(p + "auth_asym_id"))
            residue = _value(row.get(p + "auth_seq_id"))
            insertion = _value(row.get(f"pdbx_ptnr{n}_PDB_ins_code"))
            if residue and insertion:
                residue += insertion
            endpoints.append(_endpoint(
                atoms, components, chain=chain, residue=residue,
                comp=_value(row.get(p + "auth_comp_id")) or _value(row.get(p + "label_comp_id")),
                name=_value(row.get(p + "label_atom_id")) or _value(row.get(p + "auth_atom_id")),
                altloc=_value(row.get(f"pdbx_ptnr{n}_label_alt_id")),
                label_asym=None if chain else _value(row.get(p + "label_asym_id")),
                label_seq=None if residue else _value(row.get(p + "label_seq_id")),
            ))
        symmetry = [_value(row.get(f"ptnr{n}_symmetry")) for n in (1, 2)]
        error = ("Covalent link requires an unrepresented symmetry mate"
                 if any(value not in (None, "1_555", "1555") for value in symmetry) else None)
        link = _link_from_endpoints(
            endpoints, kind="disulfide" if kind == "disulf" else "covalent",
            order=_order(row.get("pdbx_value_order")) or (1.0 if kind == "disulf" else None),
            source="struct_conn", record_id=row.get("id", "?"), error=error,
        )
        for n, endpoint in enumerate(link["endpoints"], 1):
            endpoint["atom_names"] = [row.get(f"ptnr{n}_label_atom_id")
                                      or row.get(f"ptnr{n}_auth_atom_id")]
        links.append(link)
    # Branch numbering identifies monomers, not author residue sequence numbers.
    scheme = _rows(block, "_pdbx_branch_scheme.")
    for row in _rows(block, "_pdbx_entity_branch_link."):
        entity = _value(row.get("entity_id"))
        instances = sorted({r.get("asym_id") for r in scheme if r.get("entity_id") == entity
                            and r.get("asym_id")})
        if not instances:
            candidate_ids = list(dict.fromkeys(
                atom.component_id for atom in atoms
                if atom.metadata.get("source_entity_id") == entity
            ))
            links.append({
                "component_ids": candidate_ids, "order": _order(row.get("value_order")),
                "leaving_atoms": [], "kind": "covalent", "source": "entity_branch_link",
                "provenance": [{"source": "entity_branch_link", "record_id": str(row.get("link_id", "?"))}],
                "error": "Branch linkage lacks an unambiguous deposited branch scheme",
            })
        for asym in instances:
            endpoints, leaving = [], []
            for n in (1, 2):
                num = row.get(f"entity_branch_list_num_{n}")
                matches = [r for r in scheme if r.get("entity_id") == entity
                           and r.get("asym_id") == asym and r.get("num") == num]
                if len(matches) != 1:
                    endpoints.append(([], []))
                    continue
                item = matches[0]
                # pdb_seq_num/pdb_asym_id match the deposited atom_site auth ids;
                # historical auth_seq_num can differ (e.g. 1ANF: 371 vs 1).
                endpoint = _endpoint(
                    atoms, components,
                    chain=_value(item.get("pdb_asym_id")),
                    residue=_value(item.get("pdb_seq_num")),
                    comp=_value(row.get(f"comp_id_{n}")), name=_value(row.get(f"atom_id_{n}")),
                    label_asym=asym,
                )
                endpoints.append(endpoint)
                leaving_name = _value(row.get(f"leaving_atom_id_{n}"))
                if leaving_name and len(endpoint[0]) == 1:
                    leaving.append({"component_id": endpoint[0][0], "atom_name": leaving_name})
            link = _link_from_endpoints(
                endpoints, kind="covalent", order=_order(row.get("value_order")),
                source="entity_branch_link", record_id=f"{row.get('link_id', '?')}:{asym}",
            )
            link["leaving_atoms"] = leaving
            links.append(link)
    return links


def _gemmi_links(structure, atoms, components):
    import gemmi

    links = []
    for connection in structure.connections:
        if connection.type not in (gemmi.ConnectionType.Covale, gemmi.ConnectionType.Disulf):
            continue
        endpoints = []
        for address in (connection.partner1, connection.partner2):
            endpoints.append(_endpoint(
                atoms, components, chain=address.chain_name or "_",
                residue=str(address.res_id.seqid), comp=address.res_id.name,
                name=address.atom_name, altloc=address.altloc.strip("\x00 .?"),
            ))
        links.append(_link_from_endpoints(
            endpoints, kind="disulfide" if connection.type == gemmi.ConnectionType.Disulf else "covalent",
            order=1.0 if connection.type == gemmi.ConnectionType.Disulf else None,
            source="pdb_connection", record_id=connection.name,
            error="Covalent link requires an unrepresented symmetry mate"
            if connection.asu == gemmi.Asu.Different else None,
        ))
    # Repeated entries in a CONECT row encode multiplicity, but reciprocal rows
    # are the same bond and must not add to that multiplicity.
    counts = defaultdict(list)
    for serial, neighbors in structure.conect_map.items():
        for other, multiplicity in Counter(neighbors).items():
            counts[tuple(sorted((serial, other)))].append(multiplicity)
    for serials, multiplicities in sorted(counts.items()):
        candidates = [[atom for atom in atoms if atom.serial == serial] for serial in serials]
        if any(any(atom.molecule_type == "ion" or atom.element in _COORDINATION_METALS
                   for atom in side) for side in candidates):
            continue  # CONECT also encodes metal coordination, not covalent chemistry.
        endpoints = [(list(dict.fromkeys(atom.component_id for atom in side)), side) for side in candidates]
        error = "Conflicting reciprocal CONECT bond orders" if len(set(multiplicities)) > 1 else None
        order = float(max(multiplicities))
        if order > 3:
            error = "Unsupported CONECT bond order"
        links.append(_link_from_endpoints(
            endpoints, kind="covalent", order=order, source="pdb_conect",
            record_id=f"{serials[0]}-{serials[1]}", error=error,
        ))
    return links


def _merge_links(links):
    merged = []
    by_atoms = {}
    for link in links:
        key = tuple(sorted(link.get("atom_ids", ())))
        previous = by_atoms.get(key) if len(key) == 2 else None
        if previous is None:
            merged.append(link)
            if len(key) == 2:
                by_atoms[key] = link
            continue
        previous["provenance"].extend(link["provenance"])
        for cid in set(item["component_id"] for item in link["leaving_atoms"]):
            old = {item["atom_name"] for item in previous["leaving_atoms"] if item["component_id"] == cid}
            new = {item["atom_name"] for item in link["leaving_atoms"] if item["component_id"] == cid}
            if old and new and old != new:
                previous["error"] = "Conflicting deposited leaving atoms"
        for leaving in link["leaving_atoms"]:
            if leaving not in previous["leaving_atoms"]:
                previous["leaving_atoms"].append(leaving)
        if previous["order"] is not None and link["order"] is not None and previous["order"] != link["order"]:
            previous["error"] = "Conflicting deposited covalent bond orders"
        elif previous["order"] is None:
            previous["order"] = link["order"]
        if link.get("error"):
            previous["error"] = link["error"]
    return merged


def attach_source_chemistry(path, structure, atoms, components, *, include_ligands=True):
    """Attach source dictionaries/links and add validated explicit bond edges."""
    import gemmi

    templates, links = {}, []
    is_cif = Path(path).suffix.lower() in {".cif", ".mmcif"}
    if is_cif:
        try:
            document = gemmi.cif.read(str(path))
            for block in document:
                templates.update(named_templates_from_cif(block))
                links.extend(_cif_links(block, atoms, components))
        except (RuntimeError, ValueError) as exc:
            for component in components:
                component.metadata.setdefault("connectivity_diagnostics", []).append(str(exc))
    else:
        links = _gemmi_links(structure, atoms, components)
    by_id = {atom.atom_id: atom for atom in atoms}
    component_by_id = {component.component_id: component for component in components}
    for component in components:
        if component.name.upper() in templates:
            component.metadata["chemical_template"] = deepcopy(templates[component.name.upper()])
    from .polymer import polymer_sequence_links
    sequence_links = polymer_sequence_links(atoms, components)
    # Enrich explicit links before merging, preserving their source provenance.
    for sequence_link in sequence_links:
        for link in links:
            if set(link.get("atom_ids", ())) == set(sequence_link["atom_ids"]):
                link["polymer_linkage"] = sequence_link["polymer_linkage"]
    links.extend(sequence_links)
    for link in _merge_links(links):
        if not include_ligands and (not link.get("atom_ids") or len(link["component_ids"]) < 2):
            # SDF inputs replace original ligand coordinates; links into omitted
            # components must not attach an unrelated replacement by proximity.
            continue
        for cid in link["component_ids"]:
            if cid in component_by_id:
                component_by_id[cid].metadata.setdefault("structure_links", []).append(deepcopy(link))
        if link.get("error") or len(link.get("atom_ids", ())) != 2:
            continue
        first, second = (by_id[idx] for idx in link["atom_ids"])
        first.bonds = tuple(sorted(set(first.bonds) | {second.atom_id}))
        second.bonds = tuple(sorted(set(second.bonds) | {first.atom_id}))
        if first.component_id != second.component_id:
            for atom, neighbor in ((first, second), (second, first)):
                atom.metadata["assembly_link_neighbors"] = sorted(
                    set(atom.metadata.get("assembly_link_neighbors", ())) | {neighbor.atom_id}
                )
