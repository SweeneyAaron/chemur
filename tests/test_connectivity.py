"""Named dictionaries and deposited covalent connections, without network calls."""
import io

import gemmi
import pytest

from chemur import ccd
from chemur.connectivity import named_templates_from_cif
from chemur.parser import parse_structure


ATOMS = """data_linked
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
HETATM 1 C C1 . GLC B 2 . ? 0.0 0.0 0.0 1.0 10.0 1 B 1
HETATM 2 O O4 . GLC B 2 . ? 1.4 0.0 0.0 1.0 10.0 1 B 1
HETATM 3 C C1 . GLC B 2 . ? 2.8 0.0 0.0 1.0 10.0 2 B 1
#
"""

DICTIONARY = """
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_leaving_atom_flag
GLC C1 C N
GLC O4 O N
GLC O1 O Y
GLC HO4 H Y
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
GLC C1 O4 sing
GLC C1 O1 sing
GLC O4 HO4 sing
"""

CONNECTION = """
_struct_conn.id link1
_struct_conn.conn_type_id covale
_struct_conn.ptnr1_auth_asym_id B
_struct_conn.ptnr1_auth_comp_id GLC
_struct_conn.ptnr1_auth_seq_id 1
_struct_conn.ptnr1_label_atom_id O4
_struct_conn.ptnr1_symmetry 1_555
_struct_conn.ptnr2_auth_asym_id B
_struct_conn.ptnr2_auth_comp_id GLC
_struct_conn.ptnr2_auth_seq_id 2
_struct_conn.ptnr2_label_atom_id C1
_struct_conn.ptnr2_symmetry 1_555
_struct_conn.pdbx_value_order ?
"""

BRANCH = """
loop_
_pdbx_branch_scheme.asym_id
_pdbx_branch_scheme.entity_id
_pdbx_branch_scheme.mon_id
_pdbx_branch_scheme.num
_pdbx_branch_scheme.pdb_asym_id
_pdbx_branch_scheme.pdb_seq_num
_pdbx_branch_scheme.auth_seq_num
B 2 GLC 1 B 1 371
B 2 GLC 2 B 2 372
_pdbx_entity_branch_link.link_id 1
_pdbx_entity_branch_link.entity_id 2
_pdbx_entity_branch_link.entity_branch_list_num_1 2
_pdbx_entity_branch_link.comp_id_1 GLC
_pdbx_entity_branch_link.atom_id_1 C1
_pdbx_entity_branch_link.leaving_atom_id_1 O1
_pdbx_entity_branch_link.entity_branch_list_num_2 1
_pdbx_entity_branch_link.comp_id_2 GLC
_pdbx_entity_branch_link.atom_id_2 O4
_pdbx_entity_branch_link.leaving_atom_id_2 HO4
_pdbx_entity_branch_link.value_order sing
"""


def parse_cif(tmp_path, extra="", atoms=ATOMS):
    path = tmp_path / "linked.cif"
    path.write_text(atoms + extra)
    return parse_structure(path)


def test_named_dictionary_retains_unknown_charge_and_named_leaving_atoms():
    template = named_templates_from_cif(gemmi.cif.read_string("data_GLC\n" + DICTIONARY).sole_block())["GLC"]
    assert template["atoms"]["O1"]["leaving"]
    assert template["atoms"]["O1"]["charge"] is None
    assert template["bonds"][0] == {"atom1": "C1", "atom2": "O4", "order": 1.0}


def test_branch_and_connection_merge_and_preserve_identity(tmp_path):
    atoms, components = parse_cif(tmp_path, DICTIONARY + CONNECTION + BRANCH)
    assert atoms[2].atom_id in atoms[1].bonds
    assert atoms[1].metadata["assembly_link_neighbors"] == [atoms[2].atom_id]
    link = components[0].metadata["structure_links"][0]
    assert len(components[0].metadata["structure_links"]) == 1
    assert link["atom_ids"] == [1, 2]
    assert link["order"] == 1.0
    assert {item["source"] for item in link["provenance"]} == {"struct_conn", "entity_branch_link"}
    assert link["leaving_atoms"] == [
        {"component_id": "B:GLC:2", "atom_name": "O1"},
        {"component_id": "B:GLC:1", "atom_name": "HO4"},
    ]
    assert "error" not in link
    assert atoms[0].metadata["source_label_asym_id"] == "B"
    assert atoms[0].metadata["source_model"] == 1


def test_branch_link_without_struct_conn_still_resolves_instances(tmp_path):
    atoms, components = parse_cif(tmp_path, BRANCH)
    assert set(components[0].metadata["structure_links"][0]["atom_ids"]) == {1, 2}
    assert 2 in atoms[1].bonds


def test_close_residues_do_not_gain_inferred_cross_component_bonds(tmp_path):
    atoms, components = parse_cif(tmp_path)
    assert 2 not in atoms[1].bonds
    assert all("structure_links" not in component.metadata for component in components)


@pytest.mark.parametrize("kind", ["metalc", "hydrog", "saltbr"])
def test_noncovalent_records_do_not_form_assemblies(tmp_path, kind):
    atoms, components = parse_cif(tmp_path, CONNECTION.replace("conn_type_id covale", f"conn_type_id {kind}"))
    assert 2 not in atoms[1].bonds
    assert all("structure_links" not in component.metadata for component in components)


def test_symmetry_mate_link_is_diagnostic_and_does_not_add_bond(tmp_path):
    atoms, components = parse_cif(tmp_path, CONNECTION.replace("ptnr2_symmetry 1_555", "ptnr2_symmetry 2_555"))
    assert "symmetry" in components[0].metadata["structure_links"][0]["error"]
    assert 2 not in atoms[1].bonds


def test_missing_atom_keeps_component_and_named_endpoint_diagnostic(tmp_path):
    atoms, components = parse_cif(tmp_path, CONNECTION.replace("ptnr2_label_atom_id C1", "ptnr2_label_atom_id MISSING"))
    link = components[1].metadata["structure_links"][0]
    assert "Missing or ambiguous" in link["error"]
    assert link["endpoints"][1]["atom_names"] == ["MISSING"]
    assert 2 not in atoms[1].bonds


def test_conflicting_sources_do_not_commit_a_bond(tmp_path):
    atoms, components = parse_cif(tmp_path, CONNECTION.replace("pdbx_value_order ?", "pdbx_value_order doub") + BRANCH)
    assert "Conflicting" in components[0].metadata["structure_links"][0]["error"]
    assert 2 not in atoms[1].bonds


def test_ambiguous_altloc_is_not_arbitrarily_selected(tmp_path):
    rows = ATOMS.replace("HETATM 3 C C1 .", "HETATM 3 C C1 A")
    rows += "HETATM 4 C C1 B GLC B 2 . ? 2.9 0.0 0.0 0.5 10.0 2 B 1\n"
    # The added atom row must remain inside the atom_site loop.
    rows = rows.replace("#\nHETATM 4", "HETATM 4")
    atoms, components = parse_cif(tmp_path, CONNECTION, rows)
    assert "ambiguous" in components[0].metadata["structure_links"][0]["error"]
    assert not atoms[1].metadata.get("assembly_link_neighbors")
    atoms, components = parse_cif(tmp_path, CONNECTION + "\n_struct_conn.pdbx_ptnr2_label_alt_id A\n", rows)
    assert "error" not in components[0].metadata["structure_links"][0]
    assert atoms[1].metadata["assembly_link_neighbors"] == [2]


def test_branch_link_instantiates_each_asym_without_crossing_instances(tmp_path):
    more = ("HETATM 4 C C1 . GLC C 2 . ? 10.0 0.0 0.0 1.0 10.0 1 C 1\n"
            "HETATM 5 O O4 . GLC C 2 . ? 11.4 0.0 0.0 1.0 10.0 1 C 1\n"
            "HETATM 6 C C1 . GLC C 2 . ? 12.8 0.0 0.0 1.0 10.0 2 C 1\n")
    rows = ATOMS.replace("#\n", more + "#\n")
    branch = BRANCH.replace("B 2 GLC 2 B 2 372", "B 2 GLC 2 B 2 372\nC 2 GLC 1 C 1 471\nC 2 GLC 2 C 2 472")
    atoms, components = parse_cif(tmp_path, branch, rows)
    assert atoms[1].metadata["assembly_link_neighbors"] == [2]
    assert atoms[4].metadata["assembly_link_neighbors"] == [5]
    assert len(components[0].metadata["structure_links"]) == 1


def test_missing_branch_scheme_is_recorded_conservatively(tmp_path):
    branch = BRANCH[BRANCH.index("_pdbx_entity_branch_link.link_id"):]
    _, components = parse_cif(tmp_path, branch)
    assert "branch scheme" in components[0].metadata["structure_links"][0]["error"]


def _pdb_atom(serial, name, residue, number, x, element):
    return (f"HETATM{serial:5d} {name:^4s} {residue:3s} A{number:4d}    "
            f"{x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{10.:6.2f}          {element:>2s}\n")


def test_pdb_conect_is_independent_of_gemmi_connections(tmp_path):
    path = tmp_path / "linked.pdb"
    path.write_text(_pdb_atom(1, "C1", "LIG", 1, 0., "C")
                    + _pdb_atom(2, "O1", "LIG", 2, 1.4, "O")
                    + "CONECT    1    2\nCONECT    2    1\nEND\n")
    atoms, components = parse_structure(path)
    assert atoms[0].bonds == (1,)
    assert len(components[0].metadata["structure_links"]) == 1
    assert components[0].metadata["structure_links"][0]["source"] == "pdb_conect"


@pytest.mark.parametrize("metal_residue", ["ZN", "LIG"])
def test_conect_metal_coordination_is_not_covalent(tmp_path, metal_residue):
    path = tmp_path / "metal.pdb"
    path.write_text(_pdb_atom(1, "N1", "LIG", 1, 0., "N")
                    + _pdb_atom(2, "ZN", metal_residue, 2, 2., "Zn")
                    + "CONECT    1    2\nEND\n")
    atoms, components = parse_structure(path)
    assert atoms[0].bonds == ()
    assert all("structure_links" not in component.metadata for component in components)


@pytest.mark.parametrize("kind", [gemmi.ConnectionType.Covale, gemmi.ConnectionType.Disulf])
def test_pdb_link_and_ssbond_are_imported(tmp_path, kind):
    text = (_pdb_atom(1, "SG", "CYS", 1, 0., "S")
            + _pdb_atom(2, "SG", "CYS", 2, 2.0, "S") + "END\n")
    structure = gemmi.read_pdb_string(text)
    connection = gemmi.Connection()
    connection.name = "link1"
    connection.type = kind
    connection.asu = gemmi.Asu.Same
    connection.partner1 = gemmi.AtomAddress("A", gemmi.SeqId(1, " "), "CYS", "SG")
    connection.partner2 = gemmi.AtomAddress("A", gemmi.SeqId(2, " "), "CYS", "SG")
    structure.connections.append(connection)
    path = tmp_path / "linked.pdb"
    path.write_text(structure.make_pdb_string())
    atoms, components = parse_structure(path)
    assert atoms[0].bonds == (1,)
    assert components[0].metadata["structure_links"][0]["kind"] == (
        "disulfide" if kind == gemmi.ConnectionType.Disulf else "covalent")


def test_omitted_structure_ligand_does_not_attach_to_sdf_replacement(tmp_path):
    path = tmp_path / "attached.pdb"
    path.write_text(_pdb_atom(1, "SG", "CYS", 1, 0., "S")
                    + _pdb_atom(2, "C1", "LIG", 2, 1.8, "C")
                    + "CONECT    1    2\nEND\n")
    atoms, components = parse_structure(path, include_ligands=False)
    assert len(atoms) == len(components) == 1
    assert "structure_links" not in components[0].metadata
    assert atoms[0].bonds == ()


def test_ccd_named_fetch_retains_existing_smiles_wrapper(monkeypatch):
    data = "data_GLC\n" + DICTIONARY + """
loop_
_pdbx_chem_comp_descriptor.comp_id
_pdbx_chem_comp_descriptor.type
_pdbx_chem_comp_descriptor.program
_pdbx_chem_comp_descriptor.descriptor
GLC SMILES CACTVS 'CO'
"""
    calls = []
    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(data.encode())
    monkeypatch.setattr(ccd, "urlopen", fake_urlopen)
    template = ccd.fetch_ccd_template(" glc ", timeout=3.)
    assert template["smiles"] == "CO"
    assert template["source"] == "ccd"
    assert ccd.fetch_ccd_smiles("GLC") == "CO"
    assert calls[0][1] == 3.


def test_ccd_batch_fetches_duplicate_component_name_once(monkeypatch, tmp_path):
    _, components = parse_cif(tmp_path)
    calls = []
    def fake_fetch(name, *, timeout):
        calls.append(name)
        return {"atoms": {}, "bonds": [], "source": "ccd"}
    monkeypatch.setattr(ccd, "fetch_ccd_template", fake_fetch)
    result = ccd.fetch_missing_ligand_templates_from_ccd(components)
    assert calls == ["GLC"]
    assert list(result.templates) == ["GLC"]
    assert not result.missing
