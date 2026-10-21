"""Named assembly templates preserve charge completeness and stereochemistry."""
from copy import deepcopy

import pytest
from rdkit import Chem

from chemur.assemblies import _named_molecule, _template
from chemur.errors import LigandTemplateError
from chemur.models import ComponentRecord


def component(template):
    return ComponentRecord("B:LIG:1", "LIG", "B", "1", "ligand", (),
                           metadata={"chemical_template": template})


def dictionary(smiles, *, unknown_charge=False, reference=False):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    names = {atom.GetIdx(): f"a{atom.GetIdx()}" for atom in molecule.GetAtoms()}
    result = {
        "atoms": {
            names[atom.GetIdx()]: {
                "element": atom.GetSymbol(),
                "charge": None if unknown_charge else atom.GetFormalCharge(),
                "aromatic": atom.GetIsAromatic(), "leaving": False,
                "stereo": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
            } for atom in molecule.GetAtoms()
        },
        "bonds": [{"atom1": names[bond.GetBeginAtomIdx()], "atom2": names[bond.GetEndAtomIdx()],
                   "order": bond.GetBondTypeAsDouble()} for bond in molecule.GetBonds()],
        "source": "embedded_cif",
    }
    if reference:
        result["smiles"] = smiles
    return result


@pytest.mark.parametrize("smiles", ["N[C@@H](C)C(=O)O", "OC[C@H](O)[C@@H](O)CO",
                                     "C[C@H](F)[C@@H](Cl)Br"])
def test_named_cip_stereo_survives_reordered_atom_indices(smiles):
    template = dictionary(smiles, unknown_charge=True)
    molecule = _named_molecule(template, component(template))
    for atom in molecule.GetAtoms():
        expected = template["atoms"][atom.GetProp("_chemur_name")]["stereo"]
        if expected in {"R", "S"}:
            assert atom.GetProp("_CIPCode") == expected
    assert Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_template_smiles_restores_omitted_named_stereo():
    template = dictionary("N[C@@H](C)C(=O)O", reference=True)
    for entry in template["atoms"].values():
        entry["stereo"] = None
    molecule = _named_molecule(template, component(template))
    assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == Chem.MolToSmiles(Chem.MolFromSmiles(template["smiles"]))


def test_inconsistent_named_and_smiles_stereo_is_rejected():
    template = dictionary("N[C@@H](C)C(=O)O", reference=True)
    template["smiles"] = "N[C@H](C)C(=O)O"
    with pytest.raises(LigandTemplateError, match="inconsistent"):
        _named_molecule(template, component(template))


def test_claimed_stereochemistry_at_achiral_site_is_rejected():
    template = dictionary("CC")
    template["atoms"]["a0"]["stereo"] = "R"
    with pytest.raises(LigandTemplateError, match="stereocenter"):
        _named_molecule(template, component(template))


@pytest.mark.parametrize("smiles", ["CC(=O)[O-]", "C[NH3+]", "[Cl-]", "C[CH2+]"])
def test_unknown_charges_cannot_invent_hydrogen_or_accept_radicals(smiles):
    template = dictionary(smiles, unknown_charge=True)
    with pytest.raises(LigandTemplateError, match="valence or charge|radical/charge"):
        _named_molecule(template, component(template))


@pytest.mark.parametrize("smiles", ["CC(=O)O", "CN", "c1ccncc1", "CS(=O)(=O)C"])
def test_complete_neutral_named_hydrogens_validate_unknown_charge(smiles):
    template = dictionary(smiles, unknown_charge=True)
    molecule = _named_molecule(template, component(template))
    assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles", ["CC(=O)[O-]", "C[NH3+]", "[Cl-]"])
def test_validated_reference_smiles_resolves_missing_charge(smiles):
    template = dictionary(smiles, unknown_charge=True, reference=True)
    molecule = _named_molecule(template, component(template))
    assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_missing_named_hydrogens_without_charge_or_smiles_are_unresolved():
    template = dictionary("CO", unknown_charge=True)
    template["atoms"] = {name: atom for name, atom in template["atoms"].items() if atom["element"] != "H"}
    template["bonds"] = [bond for bond in template["bonds"] if bond["atom1"] in template["atoms"] and bond["atom2"] in template["atoms"]]
    with pytest.raises(LigandTemplateError, match="radical/charge"):
        _named_molecule(template, component(template))


def test_embedded_charge_gap_is_filled_by_one_consistent_ccd_fetch(monkeypatch):
    embedded = dictionary("CC(=O)[O-]", unknown_charge=True)
    fetched = dictionary("CC(=O)[O-]")
    calls = []
    def fetch(name, *, timeout):
        calls.append((name, timeout))
        return deepcopy(fetched)
    monkeypatch.setattr("chemur.ccd.fetch_ccd_template", fetch)
    cache = {}
    resolved = _template(component(embedded), True, 2.5, cache)
    _template(component(embedded), True, 2.5, cache)
    assert calls == [("LIG", 2.5)]
    assert resolved["atoms"]["a3"]["charge"] == -1
    assert embedded["atoms"]["a3"]["charge"] is None


def test_network_disabled_preserves_unknown_charge_diagnostic(monkeypatch):
    template = dictionary("CC(=O)[O-]", unknown_charge=True)
    monkeypatch.setattr("chemur.ccd.fetch_ccd_template", lambda *args, **kwargs: pytest.fail("network disabled"))
    with pytest.raises(LigandTemplateError, match="radical/charge"):
        _template(component(template), False, 2., {})


@pytest.mark.parametrize("conflict", ["element", "charge", "bond", "stereo"])
def test_ccd_fallback_rejects_conflicting_known_source_fields(monkeypatch, conflict):
    embedded = dictionary("N[C@@H](C)C(=O)[O-]", unknown_charge=True)
    fetched = dictionary("N[C@@H](C)C(=O)[O-]")
    if conflict == "element":
        fetched["atoms"]["a0"]["element"] = "O"
    elif conflict == "charge":
        embedded["atoms"]["a0"]["charge"] = 0
        fetched["atoms"]["a0"]["charge"] = 1
    elif conflict == "bond":
        fetched["bonds"][0]["order"] = 2.
    else:
        entry = next(atom for atom in fetched["atoms"].values() if atom["stereo"] in {"R", "S"})
        entry["stereo"] = "S" if entry["stereo"] == "R" else "R"
    monkeypatch.setattr("chemur.ccd.fetch_ccd_template", lambda *args, **kwargs: fetched)
    with pytest.raises(LigandTemplateError, match="mismatch"):
        _template(component(embedded), True, 2., {})
