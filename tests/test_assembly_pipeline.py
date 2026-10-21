"""Offline integration of linked-ligand chemistry, outputs and moving coordinates."""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from chemur.api import analyze
from chemur.cosplif import build_cosplif_tokens, build_ligand_blocks
from chemur.ligand_selection import ligand_selections
from chemur.parser import parse_structure
from chemur.trajectory import ArrayFrameSource, analyze_trajectory


# The 23 deposited maltose heavy-atom coordinates, extracted with Gemmi from
# 1ANF. The compact fixture remains usable when local test_data is unavailable.
GLUCOSE_COORDS = [
    (1, "C1", 8.537, 13.141, 37.436), (1, "C2", 8.657, 12.625, 38.866),
    (1, "C3", 8.946, 13.753, 39.819), (1, "C4", 10.145, 14.523, 39.360),
    (1, "C5", 9.847, 15.161, 37.965), (1, "C6", 11.109, 15.823, 37.373),
    (1, "O1", 7.343, 13.747, 37.260), (1, "O2", 7.430, 12.031, 39.245),
    (1, "O3", 9.253, 13.149, 41.094), (1, "O4", 10.317, 15.675, 40.245),
    (1, "O5", 9.352, 14.183, 37.085), (1, "O6", 10.583, 16.542, 36.238),
    (2, "C1", 11.382, 15.596, 41.181), (2, "C2", 10.686, 16.092, 42.464),
    (2, "C3", 10.252, 17.528, 42.309), (2, "C4", 11.489, 18.365, 41.845),
    (2, "C5", 12.102, 17.810, 40.577), (2, "C6", 13.422, 18.480, 40.315),
    (2, "O2", 9.590, 15.233, 42.848), (2, "O3", 9.756, 18.105, 43.528),
    (2, "O4", 11.136, 19.749, 41.624), (2, "O5", 12.406, 16.420, 40.806),
    (2, "O6", 13.908, 18.191, 38.931),
]


def _compact_cif(tmp_path, *, reverse=False):
    import gemmi

    document = gemmi.cif.Document()
    block = document.add_new_block("linked_maltose")
    atom_rows = [("W", "HOH", 9, "O", "O", 50., 50., 50.)]
    for residue in (1, 2):
        rows = [row for row in GLUCOSE_COORDS if row[0] == residue]
        if reverse:
            rows.reverse()
        atom_rows.extend(("B", "GLC", number, name, name[0], x, y, z)
                         for number, name, x, y, z in rows)
        if residue == 1:
            atom_rows.append(("I", "NA", 5, "NA", "Na", 60., 60., 60.))
    # A protein donor outside the sugar gives one readily traced external H-bond.
    atom_rows.extend([("P", "SER", 7, "OG", "O", 4.543, 13.747, 37.260),
                      ("P", "SER", 7, "HG", "H", 5.543, 13.747, 37.260)])
    loop = block.init_loop("_atom_site.", [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
        "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
        "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id",
        "auth_atom_id", "pdbx_PDB_model_num",
    ])
    for serial, (chain, residue, number, name, element, x, y, z) in enumerate(atom_rows, 1):
        entity = "2" if residue == "GLC" else "1"
        loop.add_row(["HETATM", str(serial), element, name, ".", residue, chain,
                      entity, ".", "?", str(x), str(y), str(z), "1", "0",
                      str(number), residue, chain, name, "1"])

    heavy = [f"C{i}" for i in range(1, 7)] + [f"O{i}" for i in range(1, 7)]
    attachments = [(f"C{i}", f"H{i}") for i in range(1, 6)]
    attachments += [("C6", "H61"), ("C6", "H62")]
    attachments += [(f"O{i}", f"HO{i}") for i in (1, 2, 3, 4, 6)]
    names = heavy + [hydrogen for _, hydrogen in attachments]
    block.set_mmcif_category("_chem_comp_atom.", {
        "comp_id": ["GLC"] * len(names), "atom_id": names,
        "type_symbol": [name[0] for name in names], "charge": ["0"] * len(names),
    })
    bonds = [(f"C{i}", f"C{i + 1}") for i in range(1, 6)]
    bonds += [("C1", "O5"), ("C5", "O5")]
    bonds += [(f"C{i}", f"O{i}") for i in (1, 2, 3, 4, 6)] + attachments
    block.set_mmcif_category("_chem_comp_bond.", {
        "comp_id": ["GLC"] * len(bonds), "atom_id_1": [a for a, _ in bonds],
        "atom_id_2": [b for _, b in bonds], "value_order": ["SING"] * len(bonds),
    })
    block.set_mmcif_category("_pdbx_branch_scheme.", {
        "asym_id": ["B", "B"], "entity_id": ["2", "2"], "mon_id": ["GLC", "GLC"],
        "num": ["1", "2"], "pdb_asym_id": ["B", "B"], "pdb_seq_num": ["1", "2"],
    })
    block.set_mmcif_category("_pdbx_entity_branch_link.", {
        "link_id": ["1"], "entity_id": ["2"], "entity_branch_list_num_1": ["2"],
        "comp_id_1": ["GLC"], "atom_id_1": ["C1"], "leaving_atom_id_1": ["O1"],
        "entity_branch_list_num_2": ["1"], "comp_id_2": ["GLC"], "atom_id_2": ["O4"],
        "leaving_atom_id_2": ["HO4"], "value_order": ["sing"],
    })
    block.set_mmcif_category("_struct_conn.", {
        "id": ["covale1"], "conn_type_id": ["covale"],
        "ptnr1_auth_asym_id": ["B"], "ptnr1_auth_seq_id": ["1"],
        "ptnr1_label_comp_id": ["GLC"], "ptnr1_label_atom_id": ["O4"],
        "ptnr1_label_asym_id": ["B"], "ptnr1_symmetry": ["1_555"],
        "ptnr2_auth_asym_id": ["B"], "ptnr2_auth_seq_id": ["2"],
        "ptnr2_label_comp_id": ["GLC"], "ptnr2_label_atom_id": ["C1"],
        "ptnr2_label_asym_id": ["B"], "ptnr2_symmetry": ["1_555"],
        "pdbx_value_order": [None],
    })
    path = tmp_path / ("maltose_reordered.cif" if reverse else "maltose.cif")
    document.write_file(str(path))
    return path


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Assembly integration tests must not fetch network chemistry")
    monkeypatch.setattr("chemur.ccd.urlopen", unexpected)


def _only_assembly(result):
    selections = ligand_selections(result)
    assert len(selections) == 1
    selection = selections[0]
    assert selection.ligand_id == "assembly:B:GLC:1"
    assert selection.component_ids == ("B:GLC:1", "B:GLC:2")
    assert len(selection.atom_ids) == 23
    assert selection.chemistry["status"] == "validated"
    return selection


def _site_chemistry(result):
    return {(atom.component_id, atom.name):
            (atom.element, atom.formal_charge, atom.donor_capacity, atom.acceptor_capacity,
             atom.hydrogen_count, atom.is_aromatic)
            for atom in result.atoms if atom.molecule_type == "ligand"}


def test_local_1anf_analyzes_offline_as_one_complete_maltose():
    path = Path(__file__).parents[1] / "test_data" / "structures" / "1ANF.cif"
    if not path.is_file():
        pytest.skip("Optional full local 1ANF structure is not included in the test checkout")
    result = analyze(path, fetch_ccd_smiles=False, strict_ligand_templates=True)
    selection = _only_assembly(result)
    assert set(result.split_by_ligand()) == {selection.ligand_id}
    link_oxygen = next(atom for atom in result.atoms
                       if atom.component_id == "B:GLC:1" and atom.name == "O4")
    assert link_oxygen.donor_capacity == link_oxygen.hydrogen_count == 0
    assert link_oxygen.acceptor_capacity == 2


def test_compact_maltose_permutation_preserves_site_chemistry_and_tokens(tmp_path):
    first = analyze(_compact_cif(tmp_path), fetch_ccd_smiles=False, strict_ligand_templates=True)
    second = analyze(_compact_cif(tmp_path, reverse=True), fetch_ccd_smiles=False, strict_ligand_templates=True)
    selection = _only_assembly(first)
    _only_assembly(second)
    assert _site_chemistry(first) == _site_chemistry(second)
    assert build_cosplif_tokens(first, selection.ligand_id) == build_cosplif_tokens(second, selection.ligand_id)
    assert build_cosplif_tokens(first, selection.ligand_id)


def test_whole_maltose_sdf_matches_linked_residue_chemistry_and_scoring(tmp_path):
    from rdkit import Chem

    structure = _compact_cif(tmp_path)
    linked = analyze(structure, fetch_ccd_smiles=False, strict_ligand_templates=True)
    selection = _only_assembly(linked)
    molecule = Chem.MolFromSmiles(selection.chemistry["smiles"])
    mapping = {int(index): atom_id for index, atom_id in selection.chemistry["template_mapping"].items()}
    by_id = {atom.atom_id: atom for atom in linked.atoms}
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, atom_id in mapping.items():
        conformer.SetAtomPosition(index, by_id[atom_id].coord)
    molecule.AddConformer(conformer)
    molecule.SetProp("_Name", "MALTOSE")
    sdf_path = tmp_path / "whole_maltose.sdf"
    with Chem.SDWriter(str(sdf_path)) as writer:
        writer.write(molecule)
    sdf = analyze(structure, ligand_sdf=sdf_path, fetch_ccd_smiles=False, strict_ligand_templates=True)
    sdf_selection, = ligand_selections(sdf)
    assert len(sdf_selection.atom_ids) == 23
    def chemistry_by_coord(result):
        return {tuple(round(value, 3) for value in atom.coord):
                (atom.element, atom.formal_charge, atom.donor_capacity, atom.acceptor_capacity,
                 atom.hydrogen_count, atom.is_aromatic)
                for atom in result.atoms if atom.molecule_type == "ligand"}
    assert chemistry_by_coord(linked) == chemistry_by_coord(sdf)
    assert build_cosplif_tokens(linked, selection.ligand_id) == build_cosplif_tokens(sdf, sdf_selection.ligand_id)
    linked_blocks = build_ligand_blocks(linked, selection.ligand_id)["blocks"]
    sdf_blocks = build_ligand_blocks(sdf, sdf_selection.ligand_id)["blocks"]
    assert Counter(tuple(block["fingerprint"]) for block in linked_blocks) == Counter(tuple(block["fingerprint"]) for block in sdf_blocks)


@pytest.mark.parametrize("processes", [1, 2])
def test_linked_trajectory_retains_sparse_ids_and_moving_coordinates(tmp_path, processes):
    path = _compact_cif(tmp_path)
    atoms, _ = parse_structure(path)
    coords = np.array([[atom.coord for atom in atoms]] * 2)
    ligand_positions = [index for index, atom in enumerate(atoms) if atom.molecule_type == "ligand"]
    coords[1, ligand_positions] += (10., 2., -1.)
    result = analyze_trajectory(ArrayFrameSource(path, coords), processes=processes,
                                exclude_solvent=True, exclude_ions=True,
                                fetch_ccd_smiles=False, strict_ligand_templates=True)
    selection = _only_assembly(result)
    retained = {atom.atom_id for atom in atoms if atom.molecule_type not in {"ion", "solvent"}}
    assert {atom.atom_id for atom in result.atoms} == retained
    assert retained != set(range(len(retained)))
    assert set(selection.chemistry["template_mapping"].values()) <= retained
    assert all(set(ring) <= retained for ring in selection.chemistry["template_rings"])
    assert all(set(atom.bonds) <= retained for atom in result.atoms)
    original_positions = {atom.atom_id: position for position, atom in enumerate(atoms)}
    for atom in result.atoms:
        assert atom.coord == pytest.approx(coords[-1, original_positions[atom.atom_id]])
    oxygen = next(atom.atom_id for atom in atoms if atom.component_id == "B:GLC:1" and atom.name == "O1")
    donor = next(atom.atom_id for atom in atoms if atom.component_id == "P:SER:7" and atom.name == "OG")
    assert result.occupancy()[("hbond", frozenset({oxygen, donor}))] == pytest.approx(0.5)
    assert result.frames[0].interactions
    assert not result.frames[1].interactions


def test_real_dimorphite_processes_whole_linked_ligand(tmp_path, capsys):
    pytest.importorskip("dimorphite_dl")
    result = analyze(_compact_cif(tmp_path), protonate=True,
                     fetch_ccd_smiles=False, strict_ligand_templates=True)
    selection = _only_assembly(result)
    log = capsys.readouterr().err
    assert selection.ligand_id in log
    oxygen = next(atom for atom in result.atoms if atom.component_id == "B:GLC:1" and atom.name == "O4")
    assert oxygen.hydrogen_count == oxygen.donor_capacity == 0
