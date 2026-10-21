"""Prepare connected ligand chemistry without changing deposited identities.

Source components remain a partition of the coordinates. An assembly is a
chemical view over those components, never an additional coordinate component.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys
import warnings

from .chemistry import (
    AtomAnnotation, _apply_template_hydrogen_policy, _map_template_to_component,
    _template_annotations, _template_amides, _template_carbonyls,
    normalize_sulfoxide_resonance, apply_interaction_typing_metadata,
)
from .errors import LigandTemplateError
from .models import AtomRecord, ComponentRecord
from .protonation import prepare_ligand_smiles


def assembly_registry(components):
    return {
        record["assembly_id"]: record
        for component in components
        if (record := component.metadata.get("ligand_assembly")) is not None
    }


def chemical_product_registry(components):
    return {record["assembly_id"]: record for component in components
            if (record := component.metadata.get("chemical_product")) is not None}


def load_assembly_overrides(value):
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise LigandTemplateError("Assembly overrides require a version: 1 document")
    products = value.get("products")
    if not isinstance(products, list):
        raise LigandTemplateError("Assembly overrides require a products list")
    seen = set()
    for product in products:
        if not isinstance(product, dict) or not isinstance(product.get("components"), list) or not product["components"]:
            raise LigandTemplateError("Each product must identify its seed components")
        if not all(isinstance(cid, str) for cid in product["components"]):
            raise LigandTemplateError("Product component identifiers must be strings")
        if len(set(product["components"])) != len(product["components"]):
            raise LigandTemplateError("A product contains duplicate components")
        overlap = seen.intersection(product["components"])
        if overlap:
            raise LigandTemplateError(f"Overlapping assembly overrides: {sorted(overlap)}")
        seen.update(product["components"])
    result = deepcopy(products)
    if value.get("source_entry_id"):
        for product in result:
            product.setdefault("source_entry_id", value["source_entry_id"])
    return result


def _links(components):
    unique = {}
    for component in components:
        for link in component.metadata.get("structure_links", []):
            key = json.dumps(link, sort_keys=True)
            unique[key] = link
    return list(unique.values())


def _groups(components, links, products):
    ligands = {c.component_id for c in components if c.molecule_type == "ligand" or c.metadata.get("modified_polymer")}
    adjacency = {cid: set() for cid in ligands}
    active = {c.component_id for c in components if c.metadata.get("modified_polymer") or
              (c.component_id in ligands and c.metadata.get("chemical_template", {}).get("atoms")
               and c.metadata.get("chemical_template", {}).get("bonds"))}
    for link in links:
        members = set(link.get("component_ids", ())) & ligands
        active.update(members)
        for cid in members:
            adjacency[cid].update(members - {cid})
    for product in products:
        members = set(product["components"])
        if not members <= ligands:
            raise LigandTemplateError(f"Override names unavailable ligand/modified-polymer components: {sorted(members - ligands)}")
        active.update(members)
        for cid in members:
            adjacency[cid].update(members - {cid})
    groups = []
    while active:
        pending = [min(active)]
        members = set()
        while pending:
            cid = pending.pop()
            if cid in members:
                continue
            members.add(cid)
            pending.extend(adjacency[cid] - members)
        active.difference_update(members)
        groups.append(sorted(members))
    return groups


def prepare_assemblies(
    atoms, components, *, ligand_smiles=None, assembly_overrides=None,
    fetch_ccd=True, ccd_timeout=10.0, protonate=False, ph_min=7.4, ph_max=7.4,
    precision=0.0, max_variants=1, debug=False, strict=False,
):
    """Type connected ligand/modified-polymer products and return handled IDs."""
    products = load_assembly_overrides(assembly_overrides)
    by_id = {a.atom_id: a for a in atoms}
    components_by_id = {c.component_id: c for c in components}
    links = _links(components)
    groups = _groups(components, links, products)
    handled = set()
    cache = {}
    for members in groups:
        handled.update(members)
        assembly_id = members[0] if len(members) == 1 else f"assembly:{members[0]}"
        group_links = [link for link in links if set(members) & set(link.get("component_ids", ()))]
        attached_components = sorted({cid for link in group_links for cid in link.get("component_ids", ())
                                      if cid not in members and cid in components_by_id})
        owner = components_by_id[members[0]]
        record = {
            "assembly_id": assembly_id, "component_ids": members,
            "atom_ids": sorted(aid for cid in members for aid in components_by_id[cid].atom_ids),
            "links": deepcopy(group_links), "attachment_component_ids": attached_components,
            "diagnostics": [], "status": "preparing",
            "product_kind": "modified_polymer" if any(components_by_id[cid].metadata.get("modified_polymer") for cid in members) else "ligand",
            "component_roles": {cid: components_by_id[cid].molecule_type for cid in members},
            "source_components": {cid: {key: components_by_id[cid].metadata[key]
                                        for key in ("source_entry_id", "ccd_type", "parent_component_id", "source_polymer_membership")
                                        if key in components_by_id[cid].metadata} for cid in members},
        }
        for cid in members:
            components_by_id[cid].metadata["chemical_product_id"] = assembly_id
            if components_by_id[cid].molecule_type == "ligand":
                components_by_id[cid].metadata["ligand_assembly_id"] = assembly_id
                for aid in components_by_id[cid].atom_ids:
                    by_id[aid].metadata["ligand_assembly_id"] = assembly_id
        owner.metadata["chemical_product"] = record
        if all(components_by_id[cid].molecule_type == "ligand" for cid in members):
            owner.metadata["ligand_assembly"] = record
        matching = [p for p in products if set(p["components"]) & set(members)]
        try:
            if any(link.get("error") for link in group_links):
                raise LigandTemplateError("; ".join(sorted({str(l["error"]) for l in group_links if l.get("error")})))
            if matching:
                if len(matching) != 1 or set(matching[0]["components"]) != set(members):
                    raise LigandTemplateError("An override must cover the whole connected product seed group")
                expected_source = matching[0].get("source_entry_id")
                if expected_source and any(str(components_by_id[cid].metadata.get("source_entry_id", "")).upper() != str(expected_source).upper() for cid in members):
                    raise LigandTemplateError(f"Product override is scoped to source entry {expected_source}")
                if expected_source:
                    record["override_source_entry_id"] = str(expected_source).upper()
                if matching[0].get("description"):
                    record["override_description"] = str(matching[0]["description"])
                # Even a rejected override identifies potentially altered polymer
                # chemistry. Suppress those sites on failure, not just known links.
                referenced = matching[0].get("atom_map", {})
                if isinstance(referenced, dict):
                    requested_updates = matching[0].get("update_polymer_atoms", [])
                    if not isinstance(requested_updates, (list, tuple)):
                        raise LigandTemplateError("update_polymer_atoms must be a list of atom-map numbers")
                    record["requested_update_component_ids"] = sorted({
                        ref.get("component", ref.get("component_id"))
                        for number in requested_updates
                        if isinstance(ref := referenced.get(str(number)), dict)
                        and ref.get("component", ref.get("component_id")) in components_by_id
                    })
                    attached_components = sorted(set(attached_components) | {
                        ref.get("component", ref.get("component_id"))
                        for ref in referenced.values() if isinstance(ref, dict)
                        and ref.get("component", ref.get("component_id")) in components_by_id
                        and components_by_id[ref.get("component", ref.get("component_id"))].molecule_type in {"protein", "nucleotide"}
                    })
                record["attachment_component_ids"] = attached_components
                molecule, mapping, updates, boundary = _override_product(matching[0], atoms, components, record)
                source = "assembly_override"
                attached_components = sorted({by_id[aid].component_id for aid in mapping.values()
                                              if by_id[aid].molecule_type != "ligand"})
                record["attachment_component_ids"] = attached_components
            else:
                if record["product_kind"] == "modified_polymer":
                    from .polymer import automatic_polymer_product
                    molecule, mapping, updates, boundary = automatic_polymer_product(
                        members, atoms, components_by_id, fetch_ccd, ccd_timeout, cache)
                else:
                    molecule, mapping, updates = _automatic_product(
                        members, attached_components, group_links, atoms, components_by_id,
                        ligand_smiles or {}, fetch_ccd, ccd_timeout, cache,
                    )
                    boundary = {}
                source = "linked_templates"
            bound = bool(attached_components) or record["product_kind"] == "modified_polymer"
            # User product overrides are final chemical states even when unattached.
            if protonate and not bound and not matching:
                molecule, mapping = _protonate_product(
                    molecule, mapping, atoms, record, ph_min, ph_max, precision, max_variants, debug,
                )
            elif protonate:
                record["diagnostics"].append("bound_product_protonation_preserved" if bound else "explicit_product_protonation_preserved")
                if debug or protonate:
                    print(f"[chemur] ligand {assembly_id}: retaining final product protonation", file=sys.stderr)
            _commit_product(molecule, mapping, updates, boundary, atoms, components_by_id, record, source)
        except (LigandTemplateError, ValueError, RuntimeError) as exc:
            if strict:
                raise LigandTemplateError(f"Assembly {assembly_id}: {exc}") from exc
            record["status"] = "unresolved"
            record["diagnostics"].append(str(exc))
            _exclude_assembly(record, attached_components, by_id, components_by_id, str(exc))
            warnings.warn(f"Skipping chemical product {assembly_id}; unresolved chemistry: {exc}", RuntimeWarning, stacklevel=2)
        ligand_members = [cid for cid in members if components_by_id[cid].molecule_type == "ligand"]
        if ligand_members and len(ligand_members) != len(members):
            # The shared chemical product can include a modified polymer, but
            # ligand selection/scoring owns only the actual ligand components.
            ligand_view = dict(record)
            ligand_view["component_ids"] = ligand_members
            ligand_view["atom_ids"] = sorted(aid for cid in ligand_members for aid in components_by_id[cid].atom_ids)
            components_by_id[ligand_members[0]].metadata["ligand_assembly"] = ligand_view
    return handled


def _template(component, fetch_ccd, timeout, cache):
    template = deepcopy(component.metadata.get("chemical_template"))
    local_error = None
    if template and template.get("atoms"):
        try:
            _named_molecule(template, component)
            return template
        except LigandTemplateError as exc:
            local_error = exc
    if not fetch_ccd:
        if local_error is not None:
            raise local_error
        raise LigandTemplateError(f"No complete named atom/bond template for {component.component_id}")
    from .ccd import fetch_ccd_template
    key = component.name.upper()
    if key not in cache:
        try:
            cache[key] = fetch_ccd_template(key, timeout=timeout)
        except Exception as exc:
            reason = f"; embedded chemistry: {local_error}" if local_error is not None else ""
            raise LigandTemplateError(f"No named chemistry for {component.component_id}: {exc}{reason}") from exc
    fetched = deepcopy(cache[key])
    if not template or not template.get("atoms"):
        _named_molecule(fetched, component)
        return fetched
    # A fetched dictionary may complete an embedded one, but cannot silently
    # replace deposited chemistry or assign an unrelated graph to named sites.
    for name, entry in template["atoms"].items():
        other = fetched.get("atoms", {}).get(name)
        if other is None or entry["element"] != other["element"]:
            raise LigandTemplateError(f"Embedded/CCD atom mismatch in {component.component_id}:{name}")
        if entry.get("charge") is not None and other.get("charge") is not None and entry["charge"] != other["charge"]:
            raise LigandTemplateError(f"Embedded/CCD charge mismatch in {component.component_id}:{name}")
        if entry.get("stereo") in {"R", "S"} and other.get("stereo") in {"R", "S"} and entry["stereo"] != other["stereo"]:
            raise LigandTemplateError(f"Embedded/CCD stereochemistry mismatch in {component.component_id}:{name}")
        for field in ("charge", "stereo"):
            if entry.get(field) is not None:
                other[field] = entry[field]
        other["leaving"] = bool(other.get("leaving") or entry.get("leaving"))
    fetched_bonds = {frozenset((bond["atom1"], bond["atom2"])): bond["order"]
                     for bond in fetched.get("bonds", ())}
    for bond in template.get("bonds", ()):
        if fetched_bonds.get(frozenset((bond["atom1"], bond["atom2"]))) != bond["order"]:
            raise LigandTemplateError(f"Embedded/CCD bond mismatch in {component.component_id}")
    if template.get("smiles"):
        fetched["smiles"] = template["smiles"]
    fetched["source"] = "embedded_cif+ccd"
    _named_molecule(fetched, component)
    return fetched


def _bond_type(order):
    from rdkit import Chem
    try:
        return {1.0: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC,
                2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE}[float(order)]
    except (TypeError, ValueError, KeyError) as exc:
        raise LigandTemplateError(f"Unsupported or unspecified covalent bond order: {order}") from exc


def _named_molecule(template, component):
    from rdkit import Chem
    if template.get("error"):
        raise LigandTemplateError(str(template["error"]))
    graph = Chem.RWMol()
    names = {}
    stereo = {}
    for name, entry in sorted(template["atoms"].items()):
        element = entry["element"]
        atom = Chem.Atom("H" if element in {"D", "T"} else element)
        if element in {"D", "T"}:
            atom.SetIsotope(2 if element == "D" else 3)
        charge = entry.get("charge")
        if charge is not None:
            atom.SetFormalCharge(int(charge))
        else:
            # Sanitizing an incomplete O/N graph must not silently invent a
            # proton and mistake an unresolved charged site for a neutral one.
            atom.SetNoImplicit(True)
        if entry.get("aromatic"):
            atom.SetIsAromatic(True)
        atom.SetProp("_chemur_component", component.component_id)
        atom.SetProp("_chemur_name", name)
        atom.SetBoolProp("_chemur_charge_known", charge is not None)
        names[name] = graph.AddAtom(atom)
        if entry.get("stereo") in {"R", "S"}:
            stereo[names[name]] = entry["stereo"]
    for bond in template["bonds"]:
        if bond["atom1"] not in names or bond["atom2"] not in names:
            raise LigandTemplateError(f"Template bond names absent atom in {component.component_id}")
        graph.AddBond(names[bond["atom1"]], names[bond["atom2"]], _bond_type(bond["order"]))
    molecule = graph.GetMol()
    if template.get("smiles"):
        reference = Chem.MolFromSmiles(template["smiles"])
        if reference is None:
            raise LigandTemplateError(f"Invalid named-template SMILES in {component.component_id}")
        reference = Chem.RemoveHs(reference)
        heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1]
        if reference.GetNumHeavyAtoms() != len(heavy):
            raise LigandTemplateError(f"Named dictionary/SMILES atom mismatch in {component.component_id}")
        query = Chem.RWMol()
        query_index = {}
        for idx in heavy:
            query_index[idx] = query.AddAtom(Chem.AtomFromSmarts(f"[#{molecule.GetAtomWithIdx(idx).GetAtomicNum()}]"))
        for bond in molecule.GetBonds():
            first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if first in query_index and second in query_index:
                query.AddBond(query_index[first], query_index[second], bond.GetBondType())
        matches = reference.GetSubstructMatches(query.GetMol(), uniquify=False, maxMatches=4097)
        if not matches:
            kekule = Chem.Mol(reference)
            try:
                Chem.Kekulize(kekule, clearAromaticFlags=True)
                matches = kekule.GetSubstructMatches(query.GetMol(), uniquify=False, maxMatches=4097)
            except (ValueError, RuntimeError):
                pass
        if len(matches) > 4096:
            raise LigandTemplateError(f"Named-template SMILES mapping search limit in {component.component_id}")
        signatures = set()
        for match in matches:
            signature = []
            valid = True
            for idx, ref_idx in zip(heavy, match):
                atom, ref_atom = molecule.GetAtomWithIdx(idx), reference.GetAtomWithIdx(ref_idx)
                named_h = sum(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors())
                hydrogen_count = ref_atom.GetTotalNumHs(includeNeighbors=True)
                ref_stereo = ref_atom.GetProp("_CIPCode") if ref_atom.HasProp("_CIPCode") else None
                if (hydrogen_count < named_h
                        or (atom.GetBoolProp("_chemur_charge_known") and atom.GetFormalCharge() != ref_atom.GetFormalCharge())
                        or (idx in stereo and ref_stereo and stereo[idx] != ref_stereo)):
                    valid = False
                    break
                signature.append((ref_atom.GetFormalCharge(), hydrogen_count, ref_stereo))
            if valid:
                signatures.add(tuple(signature))
        if len(signatures) != 1:
            reason = "ambiguous" if signatures else "inconsistent"
            raise LigandTemplateError(f"Named dictionary/SMILES chemistry is {reason} in {component.component_id}")
        for idx, (charge, hydrogen_count, ref_stereo) in zip(heavy, signatures.pop()):
            atom = molecule.GetAtomWithIdx(idx)
            named_h = sum(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors())
            atom.SetFormalCharge(charge)
            atom.SetBoolProp("_chemur_charge_known", True)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(hydrogen_count - named_h)
            if ref_stereo in {"R", "S"}:
                stereo.setdefault(idx, ref_stereo)
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise LigandTemplateError(f"Unresolved valence or charge in {component.component_id}: {exc}") from exc
    # An omitted charge cannot be resolved by RDKit accepting a neutral radical.
    if any(a.GetNumRadicalElectrons() for a in molecule.GetAtoms()):
        raise LigandTemplateError(f"Unresolved radical/charge in {component.component_id}")
    # R/S describes CIP order, not RDKit's index-dependent CW/CCW tag. Establish
    # all centers together, invert mismatches, then validate the resulting CIP.
    for idx in stereo:
        atom = molecule.GetAtomWithIdx(idx)
        # A manually tagged tetrahedral carbon with an implicit H can invert
        # when emitted as the first SMILES atom. Make that H explicit in the
        # atom property (without adding coordinates), as [C@H] SMILES does.
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + atom.GetNumImplicitHs())
        atom.SetNoImplicit(True)
        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    if stereo:
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        for idx, desired in stereo.items():
            atom = molecule.GetAtomWithIdx(idx)
            if not atom.HasProp("_CIPCode"):
                raise LigandTemplateError(f"Unresolved named stereocenter {component.component_id}:{atom.GetProp('_chemur_name')}")
            if atom.GetProp("_CIPCode") != desired:
                atom.InvertChirality()
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        if any(not molecule.GetAtomWithIdx(idx).HasProp("_CIPCode")
               or molecule.GetAtomWithIdx(idx).GetProp("_CIPCode") != desired
               for idx, desired in stereo.items()):
            raise LigandTemplateError(f"Inconsistent named stereochemistry in {component.component_id}")
    return molecule


def _component_override(molecule, smiles, component):
    """Map a user monomer template onto the complete named monomer first."""
    from rdkit import Chem
    replacement = Chem.MolFromSmiles(smiles)
    if replacement is None:
        raise LigandTemplateError(f"Invalid component SMILES for {component.component_id}")
    records = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        records.append(AtomRecord(atom.GetIdx(), atom.GetProp("_chemur_name"), atom.GetSymbol(),
                                  (float(atom.GetIdx()), 0.0, 0.0), component.name,
                                  component.residue_id, component.chain_id, component.component_id, "ligand",
                                  bonds=tuple(n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() != 1),
                                  metadata={"structure_formal_charge": atom.GetFormalCharge()}))
    surrogate = ComponentRecord(component.component_id, component.name, component.chain_id,
                                component.residue_id, "ligand", tuple(a.atom_id for a in records),
                                metadata={"bond_orders": [{"atom_id_1": b.GetBeginAtomIdx(),
                                          "atom_id_2": b.GetEndAtomIdx(), "order": b.GetBondTypeAsDouble()}
                                         for b in molecule.GetBonds()]})
    mapping = _map_template_to_component(replacement, surrogate, records)
    replacement = Chem.AddHs(replacement)
    for idx, original_idx in mapping.items():
        atom = replacement.GetAtomWithIdx(idx)
        atom.SetProp("_chemur_component", component.component_id)
        atom.SetProp("_chemur_name", molecule.GetAtomWithIdx(original_idx).GetProp("_chemur_name"))
    # Carry named monomer hydrogens where possible so link records can remove them.
    for idx, original_idx in mapping.items():
        old_h = sorted(n.GetProp("_chemur_name") for n in molecule.GetAtomWithIdx(original_idx).GetNeighbors() if n.GetAtomicNum() == 1)
        new_h = [n for n in replacement.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]
        for position, atom in enumerate(new_h):
            atom.SetProp("_chemur_component", component.component_id)
            atom.SetProp("_chemur_name", old_h[position] if position < len(old_h) else f"__H{idx}_{position}")
    return replacement


def _automatic_product(members, attachments, links, atoms, components, user_smiles, fetch_ccd, timeout, cache):
    from rdkit import Chem
    by_id = {a.atom_id: a for a in atoms}
    all_components = members + attachments
    molecule = None
    templates = {}
    for cid in all_components:
        component = components[cid]
        template = _template(component, fetch_ccd, timeout, cache)
        templates[cid] = template
        monomer = _named_molecule(template, component)
        override = user_smiles.get(cid.upper()) or user_smiles.get(component.name.upper())
        if override and cid in members:
            monomer = _component_override(monomer, override, component)
        molecule = monomer if molecule is None else Chem.CombineMols(molecule, monomer)
    named = {(a.GetProp("_chemur_component"), a.GetProp("_chemur_name")): a.GetIdx()
             for a in molecule.GetAtoms()}
    deposited = {}
    for cid in all_components:
        for aid in components[cid].atom_ids:
            atom = by_id[aid]
            key = (cid, atom.name)
            if key in deposited:
                raise LigandTemplateError(f"Ambiguous deposited atom identity {cid}:{atom.name}")
            deposited[key] = aid
    removals = set()
    additions = []
    update_ids = set()
    for link in links:
        ids = link.get("atom_ids", ())
        if len(ids) != 2 or any(aid not in by_id for aid in ids):
            raise LigandTemplateError("Covalent link has unresolved atom endpoints")
        endpoint_keys = [(by_id[aid].component_id, by_id[aid].name) for aid in ids]
        if any(key not in named for key in endpoint_keys):
            raise LigandTemplateError("Covalent link endpoint is missing from named chemistry")
        order = link.get("order")
        # A disulfide has a specified single bond even in legacy PDB records.
        if order is None and link.get("kind") == "disulfide":
            order = 1.0
        if order is None:
            raise LigandTemplateError("Covalent link lacks a chemically specified bond order")
        additions.append((named[endpoint_keys[0]], named[endpoint_keys[1]], order))
        update_ids.update(aid for aid in ids if by_id[aid].molecule_type != "ligand")
        for ref in link.get("leaving_atoms", []):
            key = (ref["component_id"], ref["atom_name"])
            if key not in named:
                raise LigandTemplateError(f"Declared leaving atom absent from template: {key}")
            idx = named[key]
            leaving = molecule.GetAtomWithIdx(idx)
            if key in deposited and leaving.GetAtomicNum() != 1:
                raise LigandTemplateError(f"Deposited heavy atom conflicts with leaving-group declaration: {key}")
            removals.add(idx)
            removals.update(n.GetIdx() for n in leaving.GetNeighbors() if n.GetAtomicNum() == 1)
    # Any absent heavy atom must have an explicit justification, also in context.
    for key, idx in named.items():
        if molecule.GetAtomWithIdx(idx).GetAtomicNum() != 1 and key not in deposited and idx not in removals:
            raise LigandTemplateError(f"Unexplained missing template atom {key[0]}:{key[1]}")
    extra_heavy = [key for key, aid in deposited.items() if by_id[aid].element != "H" and key not in named]
    if extra_heavy:
        raise LigandTemplateError(f"Deposited heavy atoms absent from named template: {sorted(extra_heavy)}")
    for key in deposited.keys() & named.keys():
        if by_id[deposited[key]].element != molecule.GetAtomWithIdx(named[key]).GetSymbol():
            raise LigandTemplateError(f"Named template element mismatch at {key[0]}:{key[1]}")
    graph = Chem.RWMol(molecule)
    for first, second, order in additions:
        if first in removals or second in removals:
            raise LigandTemplateError("Link endpoint is also declared a leaving atom")
        if graph.GetBondBetweenAtoms(first, second):
            raise LigandTemplateError("Duplicate product bond")
        graph.AddBond(first, second, _bond_type(order))
    for idx in sorted(removals, reverse=True):
        graph.RemoveAtom(idx)
    molecule = graph.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise LigandTemplateError(f"Declared link operations do not specify a valid product: {exc}") from exc
    if any(a.GetNumRadicalElectrons() for a in molecule.GetAtoms()):
        raise LigandTemplateError("Declared link operations leave unresolved radical chemistry")
    mapping = {a.GetIdx(): deposited[(a.GetProp("_chemur_component"), a.GetProp("_chemur_name"))]
               for a in molecule.GetAtoms() if a.GetAtomicNum() != 1}
    # A complete attachment residue is chemically interpreted, but only the
    # functional groups affected by the new bond are written back.
    update_ids = _affected_polymer_sites(molecule, mapping, update_ids, by_id)
    if attachments:
        _validate_context(molecule, mapping, update_ids, {}, atoms, list(components.values()))
    return molecule, mapping, update_ids


def _affected_polymer_sites(molecule, mapping, endpoints, by_id):
    selected = {idx for idx, aid in mapping.items() if aid in endpoints}
    changed = True
    while changed:
        changed = False
        for bond in molecule.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a in selected) != (b in selected) and (bond.GetIsConjugated() or bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1):
                selected.update((a, b)); changed = True
        for ring in molecule.GetRingInfo().AtomRings():
            if selected.intersection(ring) and not set(ring) <= selected:
                selected.update(ring); changed = True
    return {mapping[idx] for idx in selected if idx in mapping and by_id[mapping[idx]].molecule_type != "ligand"}


def _reference(ref, atoms):
    if not isinstance(ref, dict):
        raise LigandTemplateError("Atom references must name a component and atom")
    cid = ref.get("component", ref.get("component_id"))
    name = ref.get("atom", ref.get("atom_name"))
    matches = [a for a in atoms if a.component_id == cid and a.name == name]
    for qualifier in ("altloc", "model"):
        if qualifier in ref:
            matches = [a for a in matches if str(a.metadata.get(qualifier, a.metadata.get(f"source_{qualifier}", ""))) == str(ref[qualifier])]
    if len(matches) != 1:
        raise LigandTemplateError(f"Atom reference does not resolve uniquely: {ref}")
    return matches[0]


def _override_product(product, atoms, components, record):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(str(product.get("product_smiles", "")))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise LigandTemplateError("Invalid product_smiles in assembly override")
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise LigandTemplateError("Radical product chemistry is unsupported")
    refs = product.get("atom_map", {})
    boundary_refs = product.get("boundary_map", {})
    if not isinstance(refs, dict) or not isinstance(boundary_refs, dict):
        raise LigandTemplateError("Product atom_map and boundary_map must be objects")
    mapping, boundary, map_indices = {}, {}, {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        number = atom.GetAtomMapNum()
        if not number or number in map_indices:
            raise LigandTemplateError("Every product heavy/boundary atom needs a unique nonzero map number")
        map_indices[number] = atom.GetIdx()
        if atom.GetAtomicNum() == 0:
            if str(number) not in boundary_refs:
                raise LigandTemplateError(f"Boundary atom {number} has no boundary_map declaration")
            boundary[atom.GetIdx()] = _reference(boundary_refs[str(number)], atoms).atom_id
        else:
            if str(number) not in refs:
                raise LigandTemplateError(f"Product atom {number} has no deposited-site mapping")
            original = _reference(refs[str(number)], atoms)
            if original.element != atom.GetSymbol():
                raise LigandTemplateError(f"Product element mismatch at {original.label}")
            mapping[atom.GetIdx()] = original.atom_id
    if len(set(mapping.values())) != len(mapping):
        raise LigandTemplateError("Product maps multiple atoms onto one deposited site")
    if set(refs) != {str(molecule.GetAtomWithIdx(i).GetAtomMapNum()) for i in mapping}:
        raise LigandTemplateError("atom_map contains unused entries")
    if set(boundary_refs) != {str(molecule.GetAtomWithIdx(i).GetAtomMapNum()) for i in boundary}:
        raise LigandTemplateError("boundary_map contains unused entries")
    ligand_ids = set(record["atom_ids"])
    required = {a.atom_id for a in atoms if a.atom_id in ligand_ids and a.element != "H"}
    if not required <= set(mapping.values()):
        raise LigandTemplateError("Product override omits ligand heavy atoms")
    by_id = {a.atom_id: a for a in atoms}
    extra_ligands = {by_id[aid].component_id for aid in mapping.values() if by_id[aid].molecule_type == "ligand" and aid not in ligand_ids}
    if extra_ligands:
        raise LigandTemplateError(f"Product includes undeclared ligand components: {sorted(extra_ligands)}")
    numbers = product.get("update_polymer_atoms", [])
    try:
        updates = {mapping[map_indices[int(number)]] for number in numbers}
    except (KeyError, TypeError, ValueError) as exc:
        raise LigandTemplateError("Invalid update_polymer_atoms map number") from exc
    if any(by_id[aid].molecule_type not in {"protein", "nucleotide"} for aid in updates):
        raise LigandTemplateError("update_polymer_atoms must reference protein/nucleotide sites")
    updates.update(aid for aid in ligand_ids if by_id[aid].molecule_type in {"protein", "nucleotide"} and by_id[aid].element != "H")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise LigandTemplateError("An assembly product must be connected")
    inverse = {aid: idx for idx, aid in mapping.items()}
    for link in record["links"]:
        ids = link.get("atom_ids", [])
        if len(ids) == 2:
            if any(aid not in inverse for aid in ids):
                raise LigandTemplateError("Product omits a declared attachment endpoint")
            bond = molecule.GetBondBetweenAtoms(inverse[ids[0]], inverse[ids[1]])
            if bond is None or (link.get("order") is not None and bond.GetBondTypeAsDouble() != float(link["order"])):
                raise LigandTemplateError("Product conflicts with a declared covalent link")
    # Explicit product edges can specify otherwise absent attachments.
    unchanged_polymer_links = {frozenset(link.get("atom_ids", ())) for link in _links(components)
                               if link.get("polymer_linkage") and not link.get("error")}
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in mapping or j not in mapping:
            continue
        a, b = by_id[mapping[i]], by_id[mapping[j]]
        if frozenset((a.atom_id, b.atom_id)) in unchanged_polymer_links:
            continue
        if a.component_id != b.component_id and ({a.molecule_type, b.molecule_type} & {"protein", "nucleotide"}):
            for atom in (a, b):
                if atom.molecule_type != "ligand" and atom.atom_id not in updates:
                    raise LigandTemplateError("Attachment endpoint must be listed in update_polymer_atoms")
    expected = _affected_polymer_sites(molecule, mapping, updates, by_id)
    if not expected <= updates:
        raise LigandTemplateError("update_polymer_atoms must cover complete affected rings/conjugated groups")
    _validate_context(molecule, mapping, updates, boundary, atoms, components)
    _validate_polymer_changes(molecule, mapping, updates, atoms, components)
    return molecule, mapping, updates, boundary


def _source_bond_orders(components, atoms):
    """Named deposited dictionaries supply evidence without inventing orders."""
    named = {(a.component_id, a.name): a.atom_id for a in atoms}
    orders = {}
    for link in _links(components):
        if len(link.get("atom_ids", ())) == 2 and link.get("order") is not None and not link.get("error"):
            orders[frozenset(link["atom_ids"])] = float(link["order"])
    for component in components:
        for bond in component.metadata.get("bond_orders", []):
            orders[frozenset((int(bond["atom_id_1"]), int(bond["atom_id_2"])))] = float(bond["order"])
        for bond in component.metadata.get("chemical_template", {}).get("bonds", []):
            first = named.get((component.component_id, bond["atom1"]))
            second = named.get((component.component_id, bond["atom2"]))
            if first is not None and second is not None:
                orders[frozenset((first, second))] = float(bond["order"])
    return orders


def _validate_polymer_changes(molecule, mapping, updates, atoms, components):
    by_id = {a.atom_id: a for a in atoms}
    inverse = {aid: idx for idx, aid in mapping.items()}
    orders = _source_bond_orders(components, atoms)
    component_index = {c.component_id: c for c in components}
    hydrogen_counts = {}
    for component in components:
        if component.molecule_type not in {"protein", "nucleotide"} or not set(component.atom_ids).intersection(mapping.values()):
            continue
        template = component.metadata.get("chemical_template")
        if template:
            try:
                reference = _named_molecule(template, component)
            except LigandTemplateError:
                continue
            hydrogen_counts.update({
                (component.component_id, atom.GetProp("_chemur_name")): atom.GetTotalNumHs(includeNeighbors=True)
                for atom in reference.GetAtoms() if atom.GetAtomicNum() != 1
            })
            named = {a.name: a.atom_id for a in atoms if a.component_id == component.component_id}
            for bond in reference.GetBonds():
                first, second = (bond.GetBeginAtom().GetProp("_chemur_name"), bond.GetEndAtom().GetProp("_chemur_name"))
                if first in named and second in named and bond.GetIsAromatic():
                    orders[frozenset((named[first], named[second]))] = 1.5
    # Source dictionaries are monomers. Established sequence links remove one
    # proton from peptide N / nucleotide O3' before context is compared.
    for link in _links(components):
        if not link.get("polymer_linkage") or link.get("error"):
            continue
        donor_name = "N" if link["polymer_linkage"] == "peptide" else "O3'"
        for aid in link.get("atom_ids", ()):
            original = by_id.get(aid)
            if original is not None and original.name == donor_name:
                key = (original.component_id, original.name)
                if key in hydrogen_counts:
                    hydrogen_counts[key] = max(0, hydrogen_counts[key] - 1)
    for idx, aid in mapping.items():
        original = by_id[aid]
        if original.molecule_type not in {"protein", "nucleotide"} or aid in updates:
            continue
        product_atom = molecule.GetAtomWithIdx(idx)
        old_neighbors = {n for n in original.bonds if n in inverse}
        new_neighbors = {mapping[n.GetIdx()] for n in product_atom.GetNeighbors() if n.GetIdx() in mapping}
        if old_neighbors != new_neighbors:
            raise LigandTemplateError("Changed polymer bond endpoints must be listed in update_polymer_atoms")
        for neighbor in old_neighbors:
            expected = orders.get(frozenset((aid, neighbor)))
            observed = molecule.GetBondBetweenAtoms(idx, inverse[neighbor]).GetBondTypeAsDouble()
            if expected is not None and expected != observed:
                raise LigandTemplateError("Changed polymer bond orders require update_polymer_atoms")
        template_atom = component_index[original.component_id].metadata.get("chemical_template", {}).get("atoms", {}).get(original.name, {})
        expected_charge = template_atom.get("charge", original.metadata.get("structure_formal_charge"))
        if expected_charge is not None and product_atom.GetFormalCharge() != int(expected_charge):
            raise LigandTemplateError("Changed polymer charge requires update_polymer_atoms")
        expected_hydrogens = hydrogen_counts.get((original.component_id, original.name))
        observed_hydrogens = sum(by_id[n].element == "H" for n in original.bonds if n in by_id)
        product_hydrogens = product_atom.GetTotalNumHs(includeNeighbors=True)
        if ((expected_hydrogens is not None and expected_hydrogens != product_hydrogens)
                or observed_hydrogens > product_hydrogens):
            raise LigandTemplateError("Changed polymer hydrogen state requires update_polymer_atoms")


def _validate_context(molecule, mapping, updates, boundary, atoms, components=()):
    """No invented caps: every cut must name a known, distant single bond."""
    by_id = {a.atom_id: a for a in atoms}
    inverse = {aid: idx for idx, aid in mapping.items()}
    source_orders = _source_bond_orders(components, atoms)
    for aid in updates:
        if aid not in inverse:
            raise LigandTemplateError("Rewritten polymer site is outside the product context")
        queue = deque([(aid, 0)])
        visited = set()
        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if by_id[current].element == "H":
                continue
            if current not in inverse:
                raise LigandTemplateError("Product lacks three bonds of real attachment context")
            if depth < 3:
                queue.extend((neighbor, depth + 1) for neighbor in by_id[current].bonds if neighbor in by_id)
    for idx, omitted_id in boundary.items():
        stub = molecule.GetAtomWithIdx(idx)
        if stub.GetDegree() != 1:
            raise LigandTemplateError("Each boundary stub must represent one cut bond")
        bond = stub.GetBonds()[0]
        other = bond.GetOtherAtomIdx(idx)
        if other not in mapping or omitted_id not in by_id[mapping[other]].bonds:
            raise LigandTemplateError("Boundary stub does not match a deposited bond")
        if bond.GetBondTypeAsDouble() != 1 or bond.GetIsAromatic() or bond.GetIsConjugated() or bond.IsInRing():
            raise LigandTemplateError("Only single nonconjugated nonring context cuts are supported")
        real_id = mapping[other]
        source_order = source_orders.get(frozenset((real_id, omitted_id)))
        if source_order != 1.0:
            raise LigandTemplateError("Boundary cuts need a known deposited single bond")
        # Reject a cut in any deposited ring, independently of how the truncated
        # product labels the bond. Reachability without this edge detects rings.
        pending, visited = [real_id], set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in by_id[current].bonds:
                if neighbor not in by_id or by_id[neighbor].element == "H":
                    continue
                if frozenset((current, neighbor)) == frozenset((real_id, omitted_id)):
                    continue
                pending.append(neighbor)
        if omitted_id in visited:
            raise LigandTemplateError("A context boundary cannot cut a deposited ring")
        if by_id[real_id].is_aromatic or by_id[omitted_id].is_aromatic:
            raise LigandTemplateError("A context boundary cannot cut aromatic context")
        def unsaturated(aid):
            return any(order > 1.0 and aid in pair for pair, order in source_orders.items())
        if unsaturated(real_id) and (unsaturated(omitted_id) or by_id[omitted_id].element in {"N", "O", "S"}):
            raise LigandTemplateError(f"A context boundary cannot cut conjugated context: {by_id[real_id].label}–{by_id[omitted_id].label}")
        if unsaturated(omitted_id) and by_id[real_id].element in {"N", "O", "S"}:
            raise LigandTemplateError(f"A context boundary cannot cut conjugated context: {by_id[real_id].label}–{by_id[omitted_id].label}")
        for aid in updates:
            from rdkit import Chem
            if len(Chem.GetShortestPath(molecule, inverse[aid], idx)) - 1 <= 3:
                raise LigandTemplateError("Boundary stub is too close to an updated polymer atom")
    # Every omitted heavy neighbor of the represented context needs a stub.
    for idx, aid in mapping.items():
        for neighbor_id in by_id[aid].bonds:
            if neighbor_id not in by_id or by_id[neighbor_id].element == "H" or neighbor_id in inverse:
                continue
            if not any(boundary.get(n.GetIdx()) == neighbor_id for n in molecule.GetAtomWithIdx(idx).GetNeighbors()):
                raise LigandTemplateError("Product context truncates an undeclared heavy-atom bond")


def _protonate_product(molecule, mapping, atoms, record, ph_min, ph_max, precision, max_variants, debug):
    from rdkit import Chem
    # Atom-map labels are not assumed to survive the external protonator.
    smiles, _ = _serialize_molecule(molecule, mapping)
    prepared = prepare_ligand_smiles({record["assembly_id"]: smiles}, protonate=True,
                                     ph_min=ph_min, ph_max=ph_max, precision=precision,
                                     max_variants=max_variants, debug=debug)
    product = Chem.MolFromSmiles(prepared[record["assembly_id"]])
    if product is None:
        raise LigandTemplateError("Protonation returned invalid product chemistry")
    ids = set(record["atom_ids"])
    surrogate = ComponentRecord(record["assembly_id"], "ASSEMBLY", "", "", "ligand", tuple(ids),
                                metadata={"bond_orders": _bond_orders(molecule, mapping)})
    records = [deepcopy(a) for a in atoms if a.atom_id in ids]
    # Use validated product topology, not the original independently inferred graph.
    neighbors = {aid: set() for aid in ids}
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in mapping and j in mapping:
            neighbors[mapping[i]].add(mapping[j]); neighbors[mapping[j]].add(mapping[i])
    for atom in records:
        if atom.element != "H":
            atom.bonds = tuple(sorted(neighbors[atom.atom_id] | {n for n in atom.bonds if n in ids and next(a for a in records if a.atom_id == n).element == "H"}))
    remapping = _map_template_to_component(product, surrogate, records)
    return product, remapping


def _serialize_molecule(molecule, mapping):
    from rdkit import Chem
    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    # Retain isotope H; ordinary explicit template H can be implicit safely.
    copy = Chem.RemoveHs(copy)
    encoded = Chem.MolToSmiles(copy, canonical=True)
    parsed = Chem.MolFromSmiles(encoded)
    translated = {a.GetIdx(): mapping[a.GetAtomMapNum() - 1] for a in parsed.GetAtoms()
                  if a.GetAtomMapNum() - 1 in mapping}
    for atom in parsed.GetAtoms():
        atom.SetAtomMapNum(0)
    # canonical=False still serializes by traversal, so use RDKit's output order
    # to translate the original indices rather than assuming identity.
    smiles = Chem.MolToSmiles(parsed, canonical=False)
    order = json.loads(parsed.GetProp("_smilesAtomOutputOrder"))
    translated = {new_idx: translated[old_idx] for new_idx, old_idx in enumerate(order) if old_idx in translated}
    return smiles, {str(idx): aid for idx, aid in translated.items()}


def _bond_orders(molecule, mapping):
    return [{"atom_id_1": mapping[b.GetBeginAtomIdx()], "atom_id_2": mapping[b.GetEndAtomIdx()],
             "order": float(b.GetBondTypeAsDouble())} for b in molecule.GetBonds()
            if b.GetBeginAtomIdx() in mapping and b.GetEndAtomIdx() in mapping]


def _commit_product(molecule, mapping, updates, boundary, atoms, components, record, source):
    molecule = normalize_sulfoxide_resonance(molecule)
    by_id = {a.atom_id: a for a in atoms}
    ligand_ids = set(record["atom_ids"])
    written = ligand_ids | set(updates)
    annotations = _template_annotations(molecule)
    smiles, all_serialized_mapping = _serialize_molecule(molecule, {**mapping, **boundary})
    real_ids, external_ids = set(mapping.values()), set(boundary.values())
    serialized_mapping = {index: aid for index, aid in all_serialized_mapping.items() if aid in real_ids}
    serialized_boundary = {index: aid for index, aid in all_serialized_mapping.items() if aid in external_ids}
    orders = _bond_orders(molecule, mapping)
    rings = [[mapping[idx] for idx in ring] for ring in molecule.GetRingInfo().AtomRings() if all(idx in mapping for idx in ring)]
    carbonyls, amides = _template_carbonyls(molecule, mapping), _template_amides(molecule, mapping)
    leaving_hydrogens = {
        atom.atom_id for link in record["links"] for ref in link.get("leaving_atoms", [])
        for atom in atoms if atom.element == "H" and atom.component_id == ref["component_id"]
        and atom.name == ref["atom_name"]
    }
    # Restoring a template must restore previously excluded H coordinates too.
    # The product policy below will exclude exactly the now-inapplicable sites.
    refreshed = written | {n for aid in written for n in by_id[aid].bonds
                           if n in by_id and by_id[n].element == "H"}
    # Everything above must succeed before modifying deposited chemistry.
    for cid in record["attachment_component_ids"]:
        for aid in components[cid].atom_ids:
            atom = by_id[aid]
            baseline = atom.metadata.get("assembly_baseline_chemistry")
            suppressed = set(atom.metadata.get("assembly_unresolved_assemblies", []))
            suppressed.discard(record["assembly_id"])
            if suppressed:
                atom.metadata["assembly_unresolved_assemblies"] = sorted(suppressed)
            elif baseline is not None and atom.metadata.get("assembly_chemistry_unresolved"):
                for name, value in baseline.items():
                    setattr(atom, name, value)
                for key in ("assembly_chemistry_unresolved", "assembly_unresolved_reason", "assembly_unresolved_assemblies"):
                    atom.metadata.pop(key, None)
    for aid in refreshed:
        for key in ("ignored_by_template", "ignored_reason", "ignored_hydrogen_atom_ids",
                    "structure_hydrogen_count", "template_hydrogen_count", "assembly_chemistry_unresolved"):
            by_id[aid].metadata.pop(key, None)
    apply_interaction_typing_metadata(molecule, mapping, by_id, atom_ids=written)
    for idx, aid in mapping.items():
        if aid not in written:
            continue
        atom, annotation = by_id[aid], annotations[idx]
        for field in fields(AtomAnnotation):
            setattr(atom, field.name, getattr(annotation, field.name))
        atom.metadata["chemistry_source"] = source
        atom.metadata["acceptor_source"] = "rdkit_lipinski"
        atom.metadata.pop("acceptor_unresolved_reason", None)
        forced = leaving_hydrogens.intersection(atom.bonds)
        if forced:
            policy_atom = deepcopy(atom)
            policy_atom.bonds = tuple(n for n in atom.bonds if n not in forced)
            _apply_template_hydrogen_policy(policy_atom, annotation, by_id)
            atom.metadata.update(policy_atom.metadata)
            atom.metadata["structure_hydrogen_count"] += len(forced)
            atom.metadata["ignored_hydrogen_atom_ids"] = sorted(
                forced | set(policy_atom.metadata.get("ignored_hydrogen_atom_ids", [])))
            for hydrogen_id in forced:
                by_id[hydrogen_id].metadata["ignored_by_template"] = True
                by_id[hydrogen_id].metadata["ignored_reason"] = "declared_assembly_leaving_hydrogen"
        else:
            _apply_template_hydrogen_policy(atom, annotation, by_id)
    # Reconstruct all internal product edges while retaining deposited H and
    # explicitly represented polymer-boundary connections.
    mapped_ids = set(mapping.values())
    neighbors = {aid: {n for n in by_id[aid].bonds if n not in mapped_ids} for aid in mapped_ids}
    for bond in orders:
        first, second = bond["atom_id_1"], bond["atom_id_2"]
        neighbors[first].add(second); neighbors[second].add(first)
        if by_id[first].component_id != by_id[second].component_id:
            for aid, other in ((first, second), (second, first)):
                existing = set(by_id[aid].metadata.get("assembly_link_neighbors", []))
                existing.add(other)
                by_id[aid].metadata["assembly_link_neighbors"] = sorted(existing)
            if not any(set(link.get("atom_ids", [])) == {first, second} for link in record["links"]):
                record["links"].append({
                    "atom_ids": [first, second],
                    "component_ids": [by_id[first].component_id, by_id[second].component_id],
                    "order": bond["order"], "kind": "covalent", "source": source,
                    "leaving_atoms": [], "provenance": [{"source": source}],
                })
    for aid, adjacent in neighbors.items():
        by_id[aid].bonds = tuple(sorted(adjacent))
    record.update(status="validated", smiles=smiles, template_mapping=serialized_mapping,
                  boundary_template_mapping=serialized_boundary,
                  template_rings=rings, template_carbonyls=carbonyls, template_amides=amides,
                  bond_orders=orders, chemistry_source=source,
                  updated_polymer_atom_ids=sorted(updates), context_atom_ids=sorted(mapped_ids - ligand_ids),
                  boundary_atom_ids=sorted(boundary.values()),
                  leaving_hydrogen_atom_ids=sorted(leaving_hydrogens))
    # Scoring receives ligand-only chemistry. Keep boundary valence as dummy
    # sites in the serialized product; scorer selection bounds environments.
    for cid in record["component_ids"]:
        component = components[cid]
        component.metadata.pop("ignored_by_template", None)
        component.metadata.pop("ignored_reason", None)
        for key in ("smiles", "template_mapping", "template_rings", "template_carbonyls", "template_amides"):
            component.metadata.pop(key, None)
    for cid in {by_id[aid].component_id for aid in written}:
        component = components[cid]
        unresolved = [{"atom_id": aid, "atom_name": by_id[aid].name,
                       "reason": by_id[aid].metadata["acceptor_unresolved_reason"]}
                      for aid in component.atom_ids if by_id[aid].metadata.get("acceptor_unresolved_reason")]
        if unresolved:
            component.metadata["acceptor_unresolved_sites"] = unresolved
        else:
            component.metadata.pop("acceptor_unresolved_sites", None)


def _exclude_assembly(record, attachments, by_id, components, reason):
    for cid in record["component_ids"]:
        components[cid].metadata["ignored_by_template"] = True
        components[cid].metadata["ignored_reason"] = reason
    affected = set(record["atom_ids"])
    if record.get("product_kind") == "modified_polymer":
        reactive = set(record.get("requested_update_component_ids", ()))
        reactive.update(cid for link in record["links"] if not link.get("polymer_linkage")
                        for cid in link.get("component_ids", ()))
        # Ordinary peptide/phosphodiester neighbours provide context. Their
        # baseline chemistry is not changed by failure to resolve the seed's
        # unusual chemistry, so do not suppress whole flanking residues.
        attachments = [cid for cid in attachments if cid in reactive]
    for cid in attachments:
        affected.update(components[cid].atom_ids)
    for aid in affected:
        atom = by_id[aid]
        if atom.molecule_type in {"protein", "nucleotide"}:
            atom.metadata.setdefault("assembly_baseline_chemistry", {
                field.name: getattr(atom, field.name) for field in fields(AtomAnnotation)
            })
            suppressed = set(atom.metadata.get("assembly_unresolved_assemblies", []))
            suppressed.add(record["assembly_id"])
            atom.metadata["assembly_unresolved_assemblies"] = sorted(suppressed)
        atom.metadata["assembly_chemistry_unresolved"] = True
        atom.metadata["assembly_unresolved_reason"] = reason
        # Keep heavy coordinates available to the geometric occlusion stage.
        for field in fields(AtomAnnotation):
            setattr(atom, field.name, getattr(AtomAnnotation(), field.name))
