"""Connected product chemistry from named templates and explicit overrides."""

from copy import deepcopy

import pytest
from rdkit import Chem

from chemur.api import assign_for_atoms
from chemur.assemblies import assembly_registry, prepare_assemblies
from chemur.chemistry import AtomAnnotation
from chemur.errors import LigandTemplateError
from chemur.models import AtomRecord, ComponentRecord
from chemur.profile import load_profile


def _monomer(smiles, component_id, *, start=0, molecule_type="ligand", absent=()):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    names = {atom.GetIdx(): f"{atom.GetSymbol()}{atom.GetIdx()}" for atom in molecule.GetAtoms()}
    missing = set(absent)
    chain, residue, number = component_id.split(":")
    ids = {atom.GetIdx(): start + atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetIdx() not in missing}
    atoms = [
        AtomRecord(ids[atom.GetIdx()], names[atom.GetIdx()], atom.GetSymbol(),
                   (float(start + atom.GetIdx()), 0., 0.), residue, number, chain,
                   component_id, molecule_type, formal_charge=atom.GetFormalCharge(),
                   bonds=tuple(ids[neighbor.GetIdx()] for neighbor in atom.GetNeighbors() if neighbor.GetIdx() in ids))
        for atom in molecule.GetAtoms() if atom.GetIdx() in ids
    ]
    template = {
        "atoms": {names[atom.GetIdx()]: {"element": atom.GetSymbol(), "charge": atom.GetFormalCharge()}
                  for atom in molecule.GetAtoms()},
        "bonds": [{"atom1": names[bond.GetBeginAtomIdx()], "atom2": names[bond.GetEndAtomIdx()],
                   "order": bond.GetBondTypeAsDouble()} for bond in molecule.GetBonds()],
    }
    component = ComponentRecord(component_id, residue, chain, number, molecule_type,
                                tuple(atom.atom_id for atom in atoms), metadata={"chemical_template": template})
    return atoms, component, molecule, names, ids


def _connect(first, second, first_id, second_id, *, leaving=(), order=1.0):
    link = {
        "component_ids": [first.component_id, second.component_id],
        "atom_ids": [first_id, second_id], "order": order,
        "kind": "covalent", "source": "synthetic_named_link",
        "leaving_atoms": [{"component_id": cid, "atom_name": name} for cid, name in leaving],
    }
    for component in (first, second):
        component.metadata["structure_links"] = [deepcopy(link)]


def _record(components):
    record, = assembly_registry(components).values()
    return record


def _glycosidic_fixture():
    smiles = "OCC1OC(O)C(O)C(O)C1O"
    monomer = Chem.AddHs(Chem.MolFromSmiles(smiles))
    anomeric = next(atom for atom in monomer.GetAtoms()
                    if atom.GetSymbol() == "C" and sum(n.GetSymbol() == "O" for n in atom.GetNeighbors()) == 2)
    leaving_oxygen = next(n for n in anomeric.GetNeighbors() if n.GetSymbol() == "O" and not n.IsInRing())
    absent = {leaving_oxygen.GetIdx()} | {n.GetIdx() for n in leaving_oxygen.GetNeighbors() if n.GetSymbol() == "H"}
    left, left_component, _, left_names, left_ids = _monomer(smiles, "L:GL1:1", absent=absent)
    right, right_component, right_molecule, right_names, right_ids = _monomer(smiles, "L:GL2:2", start=100)
    acceptor = right_molecule.GetAtomWithIdx(0)
    leaving_hydrogen = next(n for n in acceptor.GetNeighbors() if n.GetSymbol() == "H")
    _connect(left_component, right_component, left_ids[anomeric.GetIdx()], right_ids[acceptor.GetIdx()],
             leaving=[(left_component.component_id, left_names[leaving_oxygen.GetIdx()]),
                      (right_component.component_id, right_names[leaving_hydrogen.GetIdx()])])
    return left + right, [left_component, right_component], left_ids[anomeric.GetIdx()], right_ids[acceptor.GetIdx()], right_ids[leaving_hydrogen.GetIdx()]


def test_named_glycan_monomers_form_one_product_with_declared_leaving_atoms():
    atoms, components, carbon_id, oxygen_id, hydrogen_id = _glycosidic_fixture()
    original_ids = {atom.atom_id for atom in atoms}
    handled = prepare_assemblies(atoms, components, fetch_ccd=False, strict=True)
    record = _record(components)
    by_id = {atom.atom_id: atom for atom in atoms}

    assert handled == {component.component_id for component in components}
    assert record["status"] == "validated"
    assert {atom.atom_id for atom in atoms} == original_ids
    assert oxygen_id in by_id[carbon_id].bonds
    assert carbon_id in by_id[oxygen_id].bonds
    assert oxygen_id in by_id[carbon_id].metadata["assembly_link_neighbors"]
    assert by_id[oxygen_id].donor_capacity == 0
    assert by_id[hydrogen_id].metadata["ignored_by_template"] is True
    assert len(record["template_rings"]) == 2
    assert all("template_rings" not in component.metadata for component in components)
    assert len(Chem.GetMolFrags(Chem.MolFromSmiles(record["smiles"]))) == 1
    assert set(record["template_mapping"].values()) == {atom.atom_id for atom in atoms if atom.element != "H"}


@pytest.mark.parametrize("representation", ["explicit", "implicit", "deuterium", "tritium"])
def test_connected_heavy_atom_chemistry_is_invariant_to_hydrogen_representation(representation):
    baseline_atoms, baseline_components, _, _, _ = _glycosidic_fixture()
    prepare_assemblies(baseline_atoms, baseline_components, fetch_ccd=False, strict=True)
    atoms, components, _, _, _ = _glycosidic_fixture()
    if representation == "implicit":
        atoms = [atom for atom in atoms if atom.element != "H"]
        kept = {atom.atom_id for atom in atoms}
        for atom in atoms:
            atom.bonds = tuple(neighbor for neighbor in atom.bonds if neighbor in kept)
        for component in components:
            component.atom_ids = tuple(atom_id for atom_id in component.atom_ids if atom_id in kept)
    elif representation in {"deuterium", "tritium"}:
        for atom in atoms:
            if atom.element == "H":
                atom.metadata["isotope"] = "D" if representation == "deuterium" else "T"
    prepare_assemblies(atoms, components, fetch_ccd=False, strict=True)

    def annotations(records):
        return {(atom.component_id, atom.name): tuple(getattr(atom, name) for name in AtomAnnotation.__dataclass_fields__)
                for atom in records if atom.element != "H"}
    assert annotations(atoms) == annotations(baseline_atoms)


def test_named_polymer_attachment_removes_declared_hydrogen_and_retypes_endpoint():
    ligand, ligand_component, _, ligand_names, ligand_ids = _monomer("CCl", "L:LIG:1", absent={1})
    protein, protein_component, molecule, protein_names, protein_ids = _monomer("CS", "P:CYS:2", start=100, molecule_type="protein")
    sulfur = molecule.GetAtomWithIdx(1)
    sulfur_hydrogen = next(n for n in sulfur.GetNeighbors() if n.GetSymbol() == "H")
    next(atom for atom in protein if atom.atom_id == protein_ids[1]).donor_capacity = 1
    _connect(ligand_component, protein_component, ligand_ids[0], protein_ids[1],
             leaving=[(ligand_component.component_id, ligand_names[1]),
                      (protein_component.component_id, protein_names[sulfur_hydrogen.GetIdx()])])
    atoms, components = ligand + protein, [ligand_component, protein_component]
    protein_carbon = deepcopy(next(atom for atom in protein if atom.atom_id == protein_ids[0]))
    prepare_assemblies(atoms, components, fetch_ccd=False, strict=True)
    record = _record(components)
    by_id = {atom.atom_id: atom for atom in atoms}

    assert record["updated_polymer_atom_ids"] == [protein_ids[1]]
    assert by_id[protein_ids[1]].donor_capacity == 0
    assert by_id[protein_ids[sulfur_hydrogen.GetIdx()]].metadata["ignored_by_template"] is True
    assert ligand_ids[0] in by_id[protein_ids[1]].bonds
    assert by_id[protein_ids[0]] == protein_carbon
    assert "ligand_assembly_id" not in by_id[protein_ids[1]].metadata


def test_named_leaving_hydrogen_identity_overrides_generic_excess_hydrogen_order():
    left, left_component, _, left_names, left_ids = _monomer("C", "L:ONE:1")
    right, right_component, _, right_names, right_ids = _monomer("C", "L:TWO:2", start=100)
    _connect(left_component, right_component, left_ids[0], right_ids[0],
             leaving=[(left_component.component_id, left_names[1]),
                      (right_component.component_id, right_names[1])])
    atoms, components = left + right, [left_component, right_component]
    prepare_assemblies(atoms, components, fetch_ccd=False, strict=True)
    by_id = {atom.atom_id: atom for atom in atoms}

    assert {atom.atom_id for atom in atoms if atom.metadata.get("ignored_by_template")} == {left_ids[1], right_ids[1]}
    for ids in (left_ids, right_ids):
        assert by_id[ids[0]].metadata["ignored_hydrogen_atom_ids"] == [ids[1]]
        assert not any(by_id[ids[index]].metadata.get("ignored_by_template") for index in (2, 3, 4))


def _ethane_override():
    atoms, component, _, names, ids = _monomer("CC", "L:LIG:1")
    product = {
        "components": [component.component_id], "product_smiles": "[CH2:1]=[CH2:2]",
        "atom_map": {"1": {"component": component.component_id, "atom": names[0]},
                     "2": {"component": component.component_id, "atom": names[1]}},
    }
    return atoms, [component], {"version": 1, "products": [product]}, ids


def test_explicit_product_can_change_an_internal_bond_order():
    atoms, components, overrides, ids = _ethane_override()
    prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)
    record = _record(components)
    by_id = {atom.atom_id: atom for atom in atoms}

    assert record["bond_orders"] == [{"atom_id_1": ids[0], "atom_id_2": ids[1], "order": 2.0}]
    assert [by_id[ids[index]].hydrogen_count for index in (0, 1)] == [2, 2]
    assert sum(bool(atom.metadata.get("ignored_by_template")) for atom in atoms) == 2
    assert record["chemistry_source"] == "assembly_override"


@pytest.mark.parametrize("invalid", ["duplicate_map_number", "duplicate_site", "missing_heavy_site", "unused_mapping"])
def test_product_override_rejects_invalid_atom_coverage(invalid):
    atoms, components, overrides, _ = _ethane_override()
    product = overrides["products"][0]
    if invalid == "duplicate_map_number":
        product["product_smiles"] = "[CH2:1]=[CH2:1]"
    elif invalid == "duplicate_site":
        product["atom_map"]["2"] = product["atom_map"]["1"]
    elif invalid == "missing_heavy_site":
        product["product_smiles"] = "[CH4:1]"
        del product["atom_map"]["2"]
    else:
        product["atom_map"]["3"] = product["atom_map"]["1"]

    with pytest.raises(LigandTemplateError):
        prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)


def test_automatic_product_refuses_undeclared_missing_heavy_atoms():
    atoms, components, _, _, _ = _glycosidic_fixture()
    for component in components:
        component.metadata["structure_links"][0]["leaving_atoms"] = [
            ref for ref in component.metadata["structure_links"][0]["leaving_atoms"]
            if ref["component_id"] != components[0].component_id
        ]
    with pytest.raises(LigandTemplateError, match="Unexplained missing template atom"):
        prepare_assemblies(atoms, components, fetch_ccd=False, strict=True)


def test_automatic_product_refuses_extra_deposited_heavy_atoms():
    atoms, component, _, _, _ = _monomer("CC", "L:LIG:1")
    atoms.append(AtomRecord(90, "UNDECLARED_O", "O", (0., 3., 0.), "LIG", "1", "L",
                            component.component_id, "ligand", acceptor_capacity=2))
    component.atom_ids += (90,)

    with pytest.raises(LigandTemplateError):
        prepare_assemblies(atoms, [component], fetch_ccd=False, strict=True)


def test_automatic_product_refuses_deposited_element_mismatch():
    atoms, component, _, _, _ = _monomer("CC", "L:LIG:1")
    atoms[0].element = "N"

    with pytest.raises(LigandTemplateError):
        prepare_assemblies(atoms, [component], fetch_ccd=False, strict=True)


def _bound_override(*, retained=7, wildcard=False, polymer_smiles="SCCCCCC"):
    ligand, ligand_component, _, ligand_names, _ = _monomer("C", "L:LIG:1")
    polymer, polymer_component, polymer_molecule, polymer_names, polymer_ids = _monomer(
        polymer_smiles, "P:CYS:2", start=100, molecule_type="protein"
    )
    pieces = ["[CH3:1]", "[S:2]"]
    for index in range(1, retained):
        terminal = index == polymer_molecule.GetNumHeavyAtoms() - 1
        pieces.append(f"[CH{3 if terminal else 2}:{index + 2}]")
    atom_map = {"1": {"component": ligand_component.component_id, "atom": ligand_names[0]}}
    atom_map.update({str(index + 2): {"component": polymer_component.component_id, "atom": polymer_names[index]}
                     for index in range(retained)})
    product = {
        "components": [ligand_component.component_id], "product_smiles": "".join(pieces),
        "atom_map": atom_map, "update_polymer_atoms": [2],
    }
    if wildcard:
        number = retained + 2
        product["product_smiles"] += f"[*:{number}]"
        product["boundary_map"] = {str(number): {"component": polymer_component.component_id,
                                                 "atom": polymer_names[retained]}}
    elif retained < polymer_molecule.GetNumHeavyAtoms():
        product["product_smiles"] = product["product_smiles"].replace(
            f"[CH2:{retained + 1}]", f"[CH3:{retained + 1}]"
        )
    return ligand + polymer, [ligand_component, polymer_component], {"version": 1, "products": [product]}, polymer_ids


def test_bound_override_has_complete_context_and_preserves_product_protonation(monkeypatch):
    atoms, components, overrides, polymer_ids = _bound_override()
    def unexpected_protonation(*args, **kwargs):
        pytest.fail("A bound product must retain its explicitly supplied protonation")
    monkeypatch.setattr("chemur.assemblies.prepare_ligand_smiles", unexpected_protonation)
    prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False,
                       protonate=True, strict=True)
    record = _record(components)

    assert record["status"] == "validated"
    assert record["updated_polymer_atom_ids"] == [polymer_ids[0]]
    assert "bound_product_protonation_preserved" in record["diagnostics"]
    assert next(atom for atom in atoms if atom.atom_id == polymer_ids[0]).hydrogen_count == 0


def test_bound_override_must_declare_polymer_endpoint_updates():
    atoms, components, overrides, _ = _bound_override()
    overrides["products"][0]["update_polymer_atoms"] = []
    with pytest.raises(LigandTemplateError, match="Attachment endpoint"):
        prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)


@pytest.mark.parametrize("radical_site", ["ligand", "polymer_context"])
def test_explicit_product_overrides_reject_unsupported_radicals(radical_site):
    atoms, components, overrides, _ = _bound_override()
    before, after = ("[CH3:1]", "[CH2:1]") if radical_site == "ligand" else ("[CH2:3]", "[CH:3]")
    product = overrides["products"][0]
    product["product_smiles"] = product["product_smiles"].replace(before, after)

    with pytest.raises(LigandTemplateError, match="(?i)radical"):
        prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)


def test_untouched_polymer_context_cannot_gain_hydrogen_when_source_charge_is_omitted():
    atoms, components, overrides, _ = _bound_override(retained=6, polymer_smiles="SCCNCC")
    product = overrides["products"][0]
    product["product_smiles"] = product["product_smiles"].replace("[CH2:5]", "[NH2+:5]")
    components[1].metadata["chemical_template"]["atoms"]["N3"]["charge"] = None

    with pytest.raises(LigandTemplateError, match="hydrogen|protonation|update_polymer_atoms"):
        prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)


@pytest.mark.parametrize("wildcard", [False, True])
def test_bound_override_needs_three_bonds_of_real_context(wildcard):
    atoms, components, overrides, _ = _bound_override(retained=3, wildcard=wildcard)
    with pytest.raises(LigandTemplateError, match="three bonds|too close"):
        prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)


def test_context_omission_beyond_three_bonds_still_needs_a_boundary_declaration():
    atoms, components, overrides, _ = _bound_override(retained=4)
    with pytest.raises(LigandTemplateError, match="undeclared heavy-atom bond"):
        prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)


def test_distant_single_bond_wildcard_boundary_preserves_real_topology():
    atoms, components, overrides, polymer_ids = _bound_override(retained=4, wildcard=True)
    original_count = len(atoms)
    prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)
    record = _record(components)
    by_id = {atom.atom_id: atom for atom in atoms}

    assert record["status"] == "validated"
    assert record["boundary_atom_ids"] == [polymer_ids[4]]
    assert set(record["context_atom_ids"]) == {polymer_ids[i] for i in range(4)}
    assert len(atoms) == original_count
    assert polymer_ids[4] in by_id[polymer_ids[3]].bonds
    assert not any(atom.element == "*" for atom in atoms)


@pytest.mark.parametrize("invalid_order", ["product_double", "source_double", "source_unknown"])
def test_context_boundary_needs_a_known_single_bond_on_both_sides(invalid_order):
    atoms, components, overrides, _ = _bound_override(
        retained=4, wildcard=True, polymer_smiles="SCCC=CCC" if invalid_order == "source_double" else "SCCCCCC"
    )
    if invalid_order == "product_double":
        overrides["products"][0]["product_smiles"] = "[CH3:1][S:2][CH2:3][CH2:4][CH:5]=[*:6]"
    elif invalid_order == "source_unknown":
        components[1].metadata.clear()
    with pytest.raises(LigandTemplateError, match="single|known deposited"):
        prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)


def test_context_boundary_cannot_cut_a_source_ring_even_if_product_is_acyclic():
    atoms, components, overrides, _ = _bound_override(polymer_smiles="SC1CCCCC1")
    product = overrides["products"][0]
    # Atom C4 lies four source bonds from the updated sulfur. Omitting it cuts
    # two edges of the deposited ring while keeping the required local context.
    product["product_smiles"] = "[CH3:1][S:2][CH:3]([CH2:4][CH2:5][*:100])[CH2:8][CH2:7][*:101]"
    del product["atom_map"]["6"]
    product["boundary_map"] = {
        str(number): {"component": components[1].component_id, "atom": "C4"}
        for number in (100, 101)
    }
    with pytest.raises(LigandTemplateError, match="deposited ring"):
        prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)


def test_explicit_override_cannot_conflict_with_a_declared_link_order():
    left, left_component, _, left_names, left_ids = _monomer("C", "L:ONE:1")
    right, right_component, _, right_names, right_ids = _monomer("C", "L:TWO:2", start=100)
    _connect(left_component, right_component, left_ids[0], right_ids[0], order=1.0)
    overrides = {"version": 1, "products": [{
        "components": [left_component.component_id, right_component.component_id],
        "product_smiles": "[CH2:1]=[CH2:2]",
        "atom_map": {"1": {"component": left_component.component_id, "atom": left_names[0]},
                     "2": {"component": right_component.component_id, "atom": right_names[0]}},
    }]}
    with pytest.raises(LigandTemplateError, match="conflicts with a declared covalent link"):
        prepare_assemblies(left + right, [left_component, right_component], assembly_overrides=overrides,
                           fetch_ccd=False, strict=True)


def test_repeated_templating_restores_hydrogens_allowed_by_the_new_product():
    atoms, components, overrides, ids = _ethane_override()
    prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)
    assert sum(bool(atom.metadata.get("ignored_by_template")) for atom in atoms) == 2
    overrides["products"][0]["product_smiles"] = "[CH3:1][CH3:2]"
    prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)

    assert not any(atom.metadata.get("ignored_by_template") for atom in atoms)
    assert not any(atom.metadata.get("ignored_hydrogen_atom_ids") for atom in atoms)
    assert all(atom.hydrogen_count == 3 for atom in atoms if atom.atom_id in {ids[0], ids[1]})


def test_successful_retry_restores_unaffected_polymer_chemistry_after_failure():
    atoms, components, overrides, polymer_ids = _bound_override()
    for atom in atoms:
        if atom.molecule_type == "protein" and atom.element == "C":
            atom.is_hydrophobe = True
    invalid = deepcopy(overrides)
    invalid["products"][0]["update_polymer_atoms"] = []
    with pytest.warns(RuntimeWarning, match="unresolved chemistry"):
        prepare_assemblies(atoms, components, assembly_overrides=invalid, fetch_ccd=False)
    assert all(atom.metadata.get("assembly_chemistry_unresolved")
               for atom in atoms if atom.molecule_type == "protein")

    prepare_assemblies(atoms, components, assembly_overrides=overrides, fetch_ccd=False, strict=True)
    assert _record(components)["status"] == "validated"
    assert not any(atom.metadata.get("assembly_chemistry_unresolved") for atom in atoms)
    assert all(atom.is_hydrophobe for atom in atoms if atom.molecule_type == "protein" and atom.element == "C")
    assert next(atom for atom in atoms if atom.atom_id == polymer_ids[0]).hydrogen_count == 0


def test_three_components_can_form_a_branched_connected_product():
    center, center_component, molecule, names, ids = _monomer("CC", "L:CTR:1")
    left, left_component, _, left_names, left_ids = _monomer("C", "L:ONE:2", start=100)
    right, right_component, _, right_names, right_ids = _monomer("C", "L:TWO:3", start=200)
    center_hydrogens = [neighbor.GetIdx() for neighbor in molecule.GetAtomWithIdx(0).GetNeighbors()
                        if neighbor.GetSymbol() == "H"]
    _connect(center_component, left_component, ids[0], left_ids[0],
             leaving=[(center_component.component_id, names[center_hydrogens[0]]),
                      (left_component.component_id, left_names[1])])
    first_link = deepcopy(center_component.metadata["structure_links"][0])
    _connect(center_component, right_component, ids[0], right_ids[0],
             leaving=[(center_component.component_id, names[center_hydrogens[1]]),
                      (right_component.component_id, right_names[1])])
    center_component.metadata["structure_links"].append(first_link)
    atoms, components = center + left + right, [center_component, left_component, right_component]
    prepare_assemblies(atoms, components, fetch_ccd=False, strict=True)
    record = _record(components)
    by_id = {atom.atom_id: atom for atom in atoms}

    assert len(record["component_ids"]) == 3
    assert Chem.MolToSmiles(Chem.MolFromSmiles(record["smiles"])) == Chem.MolToSmiles(Chem.MolFromSmiles("CC(C)C"))
    assert {ids[1], left_ids[0], right_ids[0]} <= set(by_id[ids[0]].bonds)
    assert by_id[ids[0]].hydrogen_count == 1


def test_attachment_to_nucleotide_retypes_its_complete_conjugated_group():
    ligand, ligand_component, _, ligand_names, ligand_ids = _monomer("CCl", "L:LIG:1", absent={1})
    nucleotide, nucleotide_component, molecule, names, ids = _monomer(
        "Nc1ccncc1", "N:A:2", start=100, molecule_type="nucleotide"
    )
    leaving_hydrogen = next(neighbor for neighbor in molecule.GetAtomWithIdx(0).GetNeighbors()
                            if neighbor.GetSymbol() == "H")
    _connect(ligand_component, nucleotide_component, ligand_ids[0], ids[0],
             leaving=[(ligand_component.component_id, ligand_names[1]),
                      (nucleotide_component.component_id, names[leaving_hydrogen.GetIdx()])])
    atoms, components = ligand + nucleotide, [ligand_component, nucleotide_component]
    prepare_assemblies(atoms, components, fetch_ccd=False, strict=True)
    record = _record(components)

    assert set(record["updated_polymer_atom_ids"]) == {atom.atom_id for atom in nucleotide if atom.element != "H"}
    assert next(atom for atom in nucleotide if atom.atom_id == ids[0]).hydrogen_count == 1
    assert all(atom.molecule_type == "nucleotide" for atom in nucleotide)
    assert all("ligand_assembly_id" not in atom.metadata for atom in nucleotide)


def test_unresolved_assembly_retains_coordinates_for_occlusion():
    atoms, components, _, _, _ = _glycosidic_fixture()
    for component in components:
        component.metadata["structure_links"][0]["order"] = None
    for index, atom in enumerate(atoms):
        atom.coord = (0., 0., float(index))
    original_coordinates = {atom.atom_id: atom.coord for atom in atoms}
    external = [
        AtomRecord(500, "CB", "C", (-1.5, 0., 0.), "ALA", "1", "A", "A:ALA:1", "protein", is_hydrophobe=True),
        AtomRecord(501, "CB", "C", (1.5, 0., 0.), "ALA", "2", "B", "B:ALA:2", "protein", is_hydrophobe=True),
    ]
    atoms += external
    components += [ComponentRecord(atom.component_id, atom.residue_name, atom.chain_id, atom.residue_id,
                                    "protein", (atom.atom_id,)) for atom in external]
    with pytest.warns(RuntimeWarning, match="unresolved chemistry"):
        prepare_assemblies(atoms, components, fetch_ccd=False)

    assert _record(components)["status"] == "unresolved"
    assert all(atom.coord == original_coordinates[atom.atom_id] for atom in atoms if atom.atom_id in original_coordinates)
    assert all(atom.metadata.get("assembly_chemistry_unresolved") for atom in atoms if atom.atom_id in original_coordinates)
    assert not any(atom.metadata.get("ignored_by_template") for atom in atoms)
    features, candidates, assigned = assign_for_atoms(atoms, components, load_profile())
    assert {atom_id for feature in features for atom_id in feature.atom_ids} == {500, 501}
    contact, = candidates
    assert contact.rejection_reason == "occluded"
    assert assigned == []
