from chemur.ccd import _choose_smiles, _extract_descriptors


def test_extracts_preferred_ccd_smiles_descriptor():
    cif_text = """
data_LIG
#
loop_
_pdbx_chem_comp_descriptor.comp_id
_pdbx_chem_comp_descriptor.type
_pdbx_chem_comp_descriptor.program
_pdbx_chem_comp_descriptor.program_version
_pdbx_chem_comp_descriptor.descriptor
LIG SMILES ACDLabs 12.01 "CC(O)=O"
LIG "Canonical SMILES" CACTVS 3.385 "CC(=O)O"
LIG "Canonical SMILES" "OpenEye OEToolkits" 2.0.6 "C(C)(=O)O"
#
"""
    descriptors = _extract_descriptors(cif_text)

    assert _choose_smiles(descriptors) == "C(C)(=O)O"

