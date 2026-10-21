"""Deposited polymer roles and source-scoped final linked-product chemistry."""
from copy import deepcopy
import json
from pathlib import Path
import warnings

import gemmi
import pytest
from rdkit import Chem

from chemur.assemblies import chemical_product_registry, prepare_assemblies, assembly_registry, _named_molecule
from chemur.features import perceive_features
from chemur.parser import parse_structure
from chemur.preparation import prepare_ligand_chemistry
from chemur.polymer import automatic_polymer_product
from chemur.models import AtomRecord, ComponentRecord

FIXTURE = Path(__file__).parent / "fixtures" / "1MHT_polymer_excerpt.cif"
OVERRIDES = Path(__file__).parents[1] / "examples" / "assembly_overrides" / "1MHT-C36-products.json"


def parsed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_structure(FIXTURE)


def test_modified_nucleotides_are_polymer_before_ligand_filtering():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms, components = parse_structure(FIXTURE, include_ligands=False)
    selected = [c for c in components if c.name == "C36"]
    assert [c.component_id for c in selected] == ["B:C36:407", "C:C36:427"]
    assert all(c.molecule_type == "nucleotide" for c in selected)
    assert [c.metadata["source_polymer_sequence_index"] for c in selected] == [6, 7]
    assert all(c.metadata["parent_component_id"] == "DC" for c in selected)
    assert all(a.molecule_type == "nucleotide" for a in atoms if a.residue_name == "C36")
    for component in selected:
        sequence = [link for link in component.metadata["structure_links"] if link.get("polymer_linkage")]
        assert len(sequence) == 2
        assert all(link["order"] == 1.0 for link in sequence)


@pytest.mark.parametrize("residue_name", ["C36", "ALA", "DC"])
def test_deposited_nonpolymer_entity_wins_over_parent_or_standard_name(tmp_path, residue_name):
    document = gemmi.cif.read(str(FIXTURE)); block = document.sole_block()
    columns = block.get_mmcif_category("_atom_site.")
    selected = [i for i, name in enumerate(columns["label_comp_id"]) if name == "C36" and columns["auth_asym_id"][i] == "B"]
    columns = {key: [values[i] for i in selected] for key, values in columns.items()}
    columns["label_entity_id"] = ["4"] * len(selected)
    columns["label_asym_id"] = ["D"] * len(selected)
    columns["label_seq_id"] = [False] * len(selected)
    columns["label_comp_id"] = [residue_name] * len(selected)
    if "auth_comp_id" in columns:
        columns["auth_comp_id"] = [residue_name] * len(selected)
    block.set_mmcif_category("_atom_site.", columns)
    block.set_mmcif_category("_chemur_source_residue.", {
        "label_asym_id": ["D"], "label_seq_id": [False], "auth_asym_id": ["B"],
        "auth_seq_id": ["407"], "comp_id": [residue_name], "source_entry_id": ["ORIGINAL"],
        "source_polymer_type": ["polydeoxyribonucleotide"],
    })
    path = tmp_path / "standalone.cif"; document.write_file(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); _, components = parse_structure(path)
    assert components[0].molecule_type == "ligand"
    assert components[0].metadata["source_entry_id"] == "ORIGINAL"


def test_pdb_seqres_alignment_supplies_modified_polymer_membership(tmp_path):
    structure = gemmi.read_structure(str(FIXTURE))
    path = tmp_path / "polymer.pdb"
    structure.write_pdb(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); _, components = parse_structure(path, include_ligands=False)
    modified = [c for c in components if c.name == "C36"]
    assert len(modified) == 2
    assert all(c.molecule_type == "nucleotide" for c in modified)
    assert [c.metadata["source_polymer_sequence_index"] for c in modified] == [6, 7]


def test_snapshot_per_residue_sources_bind_overrides_without_changing_role(tmp_path):
    document = gemmi.cif.read(str(FIXTURE)); block = document.sole_block()
    block.set_pair("_entry.id", "MIXED_SNAPSHOT")
    block.set_mmcif_category("_chemur_source_residue.", {
        "label_asym_id": ["A", "B"], "label_seq_id": ["6", "7"],
        "auth_asym_id": ["B", "C"], "auth_seq_id": ["407", "427"],
        "comp_id": ["C36", "C36"], "source_entry_id": ["1MHT", "OTHER"],
        "source_polymer_type": ["polydeoxyribonucleotide", "polydeoxyribonucleotide"],
        "source_polymer_sequence_index": ["6", "7"],
    })
    path = tmp_path / "snapshot.cif"; document.write_file(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); atoms, components = parse_structure(path)
    modified = {c.component_id: c for c in components if c.name == "C36"}
    assert modified["B:C36:407"].metadata["source_entry_id"] == "1MHT"
    assert modified["C:C36:427"].metadata["source_entry_id"] == "OTHER"
    assert all(c.molecule_type == "nucleotide" for c in modified.values())
    override = json.loads(OVERRIDES.read_text()); override["products"] = override["products"][:1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); prepare_assemblies(atoms, components, assembly_overrides=override, fetch_ccd=False)
    assert chemical_product_registry(components)["B:C36:407"]["status"] == "validated"


def test_original_sequence_index_is_preserved_without_promoting_nonpolymer(tmp_path):
    document = gemmi.cif.read(str(FIXTURE)); block = document.sole_block()
    block.set_mmcif_category("_chemur_source_residue.", {
        "label_asym_id": ["A"], "label_seq_id": ["6"], "auth_asym_id": ["B"],
        "auth_seq_id": ["407"], "pdbx_PDB_ins_code": [False], "comp_id": ["C36"],
        "source_entry_id": ["1MHT"], "source_polymer_type": ["DNA"],
        "source_polymer_sequence_index": ["56"],
    })
    path = tmp_path / "sequence.cif"; document.write_file(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); _, components = parse_structure(path)
    assert next(c for c in components if c.component_id == "B:C36:407").metadata["source_polymer_sequence_index"] == 56


def test_snapshot_conflicting_sources_cannot_satisfy_source_scoped_override(tmp_path):
    document = gemmi.cif.read(str(FIXTURE)); block = document.sole_block()
    block.set_mmcif_category("_chemur_source_residue.", {
        "label_asym_id": ["A", "A"], "label_seq_id": ["6", "6"],
        "auth_asym_id": ["B", "B"], "auth_seq_id": ["407", "407"],
        "comp_id": ["C36", "C36"], "source_entry_id": ["1MHT", "OTHER"],
        "source_polymer_type": ["DNA", "DNA"],
    })
    path = tmp_path / "ambiguous.cif"; document.write_file(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); atoms, components = parse_structure(path)
    component = next(c for c in components if c.component_id == "B:C36:407")
    assert "source_entry_id" not in component.metadata
    assert component.metadata["source_identity_diagnostics"]
    override = json.loads(OVERRIDES.read_text()); override["products"] = override["products"][:1]
    with pytest.raises(Exception, match="scoped to source entry"):
        prepare_assemblies(atoms, components, assembly_overrides=override, fetch_ccd=False, strict=True)


def test_unresolved_modified_products_report_reason_without_global_ccd_replacement():
    atoms, components = parsed()
    from chemur.chemistry import AtomAnnotation
    baseline = {a.atom_id: tuple(getattr(a, key) for key in AtomAnnotation.__dataclass_fields__)
                for a in atoms if a.residue_name == "DG"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metadata = prepare_ligand_chemistry(atoms, components, fetch_ccd=False)
    report = metadata["modified_polymer_chemistry"]
    assert set(report) == {"B:C36:407", "C:C36:427"}
    assert all(record["status"] == "unresolved" for record in report.values())
    assert "missing template atom" in report["B:C36:407"]["diagnostics"][0]
    assert "explicit product override" in report["C:C36:427"]["diagnostics"][0]
    assert metadata["ligand_assemblies"] == {}
    for atom in atoms:
        if atom.atom_id in baseline:
            assert tuple(getattr(atom, key) for key in AtomAnnotation.__dataclass_fields__) == baseline[atom.atom_id]
            assert not atom.metadata.get("assembly_chemistry_unresolved")
    assert next(a for a in atoms if a.component_id == "A:CYS:81" and a.name == "SG").metadata["assembly_chemistry_unresolved"]
    assert not any(feature for feature in perceive_features(atoms, components) if any(
        atoms[aid].residue_name == "C36" for aid in feature.atom_ids))


def test_1mht_scoped_products_preserve_roles_sugar_stereo_and_distinct_base_states():
    atoms, components = parsed()
    prepare_assemblies(atoms, components, assembly_overrides=OVERRIDES, fetch_ccd=False, strict=True)
    records = chemical_product_registry(components)
    assert assembly_registry(components) == {}
    by_site = {(a.component_id, a.name): a for a in atoms}
    for cid, record in records.items():
        assert record["status"] == "validated"
        molecule = Chem.MolFromSmiles(record["smiles"])
        inverse = {aid: int(index) for index, aid in record["template_mapping"].items()}
        for name in ("N1", "N3", "N4", "O2"):
            assert by_site[cid, name].formal_charge == 0
        assert by_site[cid, "N4"].hydrogen_count == 2
        assert by_site[cid, "N4"].donor_capacity == 1
        assert by_site[cid, "N3"].acceptor_capacity == (1 if cid.startswith("B:") else 0)
        assert by_site[cid, "O2"].acceptor_capacity == 2
        assert by_site[cid, "O2P"].formal_charge == -1
        assert by_site[cid, "O2P"].hydrogen_count == 0
        assert by_site[cid, "F"].element == "F"
        component = next(c for c in components if c.component_id == cid)
        assert "acceptor_unresolved_sites" not in component.metadata
        reference = Chem.RemoveHs(_named_molecule(component.metadata["chemical_template"], component))
        def local_handedness(atom, names):
            labels = [names[n.GetIdx()] for n in atom.GetNeighbors()] + ["~H"]
            inversions = sum(a > b for i, a in enumerate(labels) for b in labels[i + 1:])
            return (1 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW else -1) * (-1) ** inversions
        reference_names = {a.GetIdx(): a.GetProp("_chemur_name") for a in reference.GetAtoms()}
        product_names = {int(index): atoms[aid].name for index, aid in record["template_mapping"].items()}
        for name in ("C1'", "C3'", "C4'"):
            old = next(a for a in reference.GetAtoms() if a.GetProp("_chemur_name") == name)
            new = molecule.GetAtomWithIdx(inverse[by_site[cid, name].atom_id])
            # Linked substituents change CIP priorities; compare the actual
            # mapped tetrahedral orientation rather than requiring unchanged R/S.
            assert local_handedness(old, reference_names) == local_handedness(new, product_names)
        for name in ("C5", "C6"):
            assert molecule.GetAtomWithIdx(inverse[by_site[cid, name].atom_id]).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
        assert not any("ligand_assembly_id" in a.metadata for a in atoms if a.component_id == cid)
        assert set(record["boundary_template_mapping"].values()) == set(record["boundary_atom_ids"])
        assert all(molecule.GetAtomWithIdx(int(index)).GetAtomicNum() == 0 for index in record["boundary_template_mapping"])
        assert not set(record["boundary_template_mapping"]) & set(record["template_mapping"])
    assert by_site["B:C36:407", "C5"].is_aromatic
    assert not by_site["C:C36:427", "C5"].is_aromatic
    assert by_site["A:CYS:81", "SG"].hydrogen_count == 0
    assert by_site["A:CYS:81", "SG"].metadata["sulfur_pi_types"] == ["thioether"]
    features = perceive_features(atoms, components)
    assert any(f.feature_type == "ring" and f.metadata.get("aromatic") and f.component_id == "B:C36:407" for f in features)
    assert any(f.feature_type == "carbonyl" and f.component_id == "C:C36:427" for f in features)


def test_instance_and_source_scope_are_enforced():
    override = json.loads(OVERRIDES.read_text())
    override["products"] = override["products"][:1]
    atoms, components = parsed()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); prepare_assemblies(atoms, components, assembly_overrides=override, fetch_ccd=False)
    records = chemical_product_registry(components)
    assert records["B:C36:407"]["status"] == "validated"
    assert records["C:C36:427"]["status"] == "unresolved"
    atoms, components = parsed()
    for component in components:
        component.metadata["source_entry_id"] = "DIFFERENT"
    with pytest.raises(Exception, match="scoped to source entry 1MHT"):
        prepare_assemblies(atoms, components, assembly_overrides=override, fetch_ccd=False, strict=True)


def test_shared_polymer_product_and_ligand_view_emit_group_features_once():
    from math import cos, sin, pi
    atoms = [AtomRecord(i, f"C{i}", "C", (cos(i * pi / 3), sin(i * pi / 3), 0.),
                        "MOD" if i < 3 else "LIG", "1", "A" if i < 3 else "B",
                        "A:MOD:1" if i < 3 else "B:LIG:1", "nucleotide" if i < 3 else "ligand",
                        is_aromatic=True) for i in range(6)]
    product = {"assembly_id": "shared", "template_rings": [list(range(6))], "updated_polymer_atom_ids": [0, 1, 2]}
    components = [ComponentRecord("A:MOD:1", "MOD", "A", "1", "nucleotide", (0, 1, 2), metadata={"chemical_product": product}),
                  ComponentRecord("B:LIG:1", "LIG", "B", "1", "ligand", (3, 4, 5), metadata={"ligand_assembly": dict(product)})]
    rings = [f for f in perceive_features(atoms, components) if f.feature_type == "ring"]
    assert len(rings) == 1
    assert rings[0].metadata["crosses_attachment_boundary"]


@pytest.mark.parametrize("role", ["protein", "nucleotide"])
def test_generic_modified_polymer_uses_named_sequence_transformation(role):
    from chemur.polymer import polymer_sequence_links
    monomer = Chem.MolFromSmiles("N[C@@H](C)C(=O)O" if role == "protein" else "OCCOP(=O)(O)O")
    names = ("N", "CA", "CB", "C", "O", "OXT") if role == "protein" else ("O3'", "C3'", "C5'", "O5'", "P", "OP1", "OP2", "OP3")
    template = {"atoms": {names[a.GetIdx()]: {"element": a.GetSymbol(), "charge": 0} for a in monomer.GetAtoms()},
                "bonds": [{"atom1": names[b.GetBeginAtomIdx()], "atom2": names[b.GetEndAtomIdx()], "order": b.GetBondTypeAsDouble()} for b in monomer.GetBonds()]}
    atoms, components = [], []
    for position in range(3):
        cid = f"A:MOD:{position + 1}"
        leaving = ({5} if position < 2 else set()) if role == "protein" else ({7} if position > 0 else set())
        selected = set(range(monomer.GetNumAtoms())) - leaving
        ids = {index: len(atoms) + n for n, index in enumerate(sorted(selected))}
        for index in sorted(selected):
            atom = monomer.GetAtomWithIdx(index)
            atoms.append(AtomRecord(ids[index], names[index], atom.GetSymbol(), (float(ids[index]), 0., 0.),
                                    "MOD", str(position + 1), "A", cid, role,
                                    bonds=tuple(ids[n.GetIdx()] for n in atom.GetNeighbors() if n.GetIdx() in ids),
                                    metadata={"source_model": 1, "source_entity_id": "1", "source_label_asym_id": "A"}))
        components.append(ComponentRecord(cid, "MOD", "A", str(position + 1), role, tuple(ids.values()),
                                          metadata={"source_polymer_sequence_index": position + 1, "modified_polymer": position == 1,
                                                    "chemical_template": deepcopy(template)}))
    links = polymer_sequence_links(atoms, components)
    for link in links:
        for cid in link["component_ids"]:
            next(c for c in components if c.component_id == cid).metadata.setdefault("structure_links", []).append(deepcopy(link))
        first, second = (atoms[aid] for aid in link["atom_ids"])
        first.bonds += (second.atom_id,); second.bonds += (first.atom_id,)
    prepare_assemblies(atoms, components, fetch_ccd=False, strict=True)
    record = chemical_product_registry(components)["A:MOD:2"]
    assert record["status"] == "validated"
    assert assembly_registry(components) == {}
    endpoint = next(a for a in atoms if a.component_id == "A:MOD:2" and a.name == ("N" if role == "protein" else "O3'"))
    assert endpoint.hydrogen_count == (1 if role == "protein" else 0)
    assert endpoint.acceptor_capacity == (0 if role == "protein" else 2)
