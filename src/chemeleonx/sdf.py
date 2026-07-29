from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from collections.abc import Iterable

from .chemistry import apply_rdkit_ligand_template
from .errors import DependencyMissingError, StructureParseError
from .models import AtomRecord, ComponentRecord
from .parser import normalize_element


@dataclass(frozen=True)
class SdfLigand:
    source_path: Path
    molecule_index: int
    name: str
    raw_name: str
    smiles: str
    molecule: object

    @property
    def batch_key(self) -> str:
        return f"{self.source_path.stem}:{self.molecule_index + 1}:{self.name}"


def load_sdf_ligands(
    files: str | Path | Iterable[str | Path] | None = None,
    directories: str | Path | Iterable[str | Path] | None = None,
) -> list[SdfLigand]:
    paths = collect_sdf_paths(files=files, directories=directories)
    if not paths:
        return []

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyMissingError(
            "RDKit is required to read ligand SDF files. Install this project "
            "with its runtime dependencies or install rdkit."
        ) from exc

    ligands: list[SdfLigand] = []
    for path in paths:
        supplier = Chem.SDMolSupplier(str(path), True, False)
        file_ligands: list[SdfLigand] = []
        for molecule_index, molecule in enumerate(supplier):
            if molecule is None:
                raise StructureParseError(
                    f"Could not parse molecule {molecule_index + 1} from SDF file {path}"
                )
            if molecule.GetNumConformers() == 0:
                raise StructureParseError(
                    f"Molecule {molecule_index + 1} in SDF file {path} has no coordinates"
                )

            raw_name = _raw_molecule_name(molecule, path, molecule_index)
            name = _component_name(raw_name)
            file_ligands.append(
                SdfLigand(
                    source_path=path,
                    molecule_index=molecule_index,
                    name=name,
                    raw_name=raw_name,
                    smiles=_sdf_smiles(molecule),
                    molecule=molecule,
                )
            )

        if not file_ligands:
            raise StructureParseError(f"No molecules were found in SDF file {path}")
        ligands.extend(file_ligands)

    return ligands


def collect_sdf_paths(
    files: str | Path | Iterable[str | Path] | None = None,
    directories: str | Path | Iterable[str | Path] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for file_path in _as_paths(files):
        path = file_path.expanduser()
        if not path.exists():
            raise StructureParseError(f"Ligand SDF file does not exist: {path}")
        if not path.is_file():
            raise StructureParseError(f"Ligand SDF path is not a file: {path}")
        paths.append(path)

    for directory_path in _as_paths(directories):
        path = directory_path.expanduser()
        if not path.exists():
            raise StructureParseError(f"Ligand SDF directory does not exist: {path}")
        if not path.is_dir():
            raise StructureParseError(f"Ligand SDF directory path is not a directory: {path}")
        paths.extend(sorted(item for item in path.iterdir() if item.suffix.lower() == ".sdf"))

    return paths


def add_sdf_ligands(
    atoms: list[AtomRecord],
    components: list[ComponentRecord],
    ligands: Iterable[SdfLigand],
    *,
    chain_id: str = "S",
) -> dict[str, str]:
    component_ids = {component.component_id for component in components}
    next_atom_id = max((atom.atom_id for atom in atoms), default=-1) + 1
    next_residue_number = 1
    smiles_by_component_id: dict[str, str] = {}

    for ligand in ligands:
        residue_id, component_id = _next_component_id(
            ligand.name,
            chain_id,
            next_residue_number,
            component_ids,
        )
        next_residue_number = int(residue_id) + 1
        component_ids.add(component_id)

        molecule = ligand.molecule
        conformer = molecule.GetConformer()
        atom_ids_by_rdkit_idx: dict[int, int] = {}
        ligand_atom_ids: list[int] = []
        bonds_by_atom_id: dict[int, list[int]] = {}

        for rdkit_idx, rdkit_atom in enumerate(molecule.GetAtoms()):
            atom_id = next_atom_id
            next_atom_id += 1
            atom_ids_by_rdkit_idx[rdkit_idx] = atom_id
            ligand_atom_ids.append(atom_id)
            bonds_by_atom_id[atom_id] = []

            position = conformer.GetAtomPosition(rdkit_idx)
            atom = AtomRecord(
                atom_id=atom_id,
                name=_atom_name(rdkit_atom, rdkit_idx),
                element=normalize_element(rdkit_atom.GetSymbol()),
                coord=(float(position.x), float(position.y), float(position.z)),
                residue_name=ligand.name,
                residue_id=residue_id,
                chain_id=chain_id,
                component_id=component_id,
                molecule_type="ligand",
                formal_charge=int(rdkit_atom.GetFormalCharge()),
                metadata={
                    "source": "sdf",
                    "source_path": str(ligand.source_path),
                    "sdf_molecule_index": ligand.molecule_index,
                    "sdf_atom_index": rdkit_idx,
                    "sdf_name": ligand.raw_name,
                },
            )
            atoms.append(atom)

        bond_orders: list[dict[str, float | int]] = []
        for bond in molecule.GetBonds():
            begin_id = atom_ids_by_rdkit_idx[bond.GetBeginAtomIdx()]
            end_id = atom_ids_by_rdkit_idx[bond.GetEndAtomIdx()]
            bonds_by_atom_id[begin_id].append(end_id)
            bonds_by_atom_id[end_id].append(begin_id)
            bond_orders.append(
                {
                    "atom_id_1": begin_id,
                    "atom_id_2": end_id,
                    "order": float(bond.GetBondTypeAsDouble()),
                }
            )

        atoms_by_id = {atom.atom_id: atom for atom in atoms}
        for atom_id, neighbors in bonds_by_atom_id.items():
            atoms_by_id[atom_id].bonds = tuple(sorted(neighbors))

        component = ComponentRecord(
            component_id=component_id,
            name=ligand.name,
            chain_id=chain_id,
            residue_id=residue_id,
            molecule_type="ligand",
            atom_ids=tuple(ligand_atom_ids),
            metadata={
                "source": "sdf",
                "source_path": str(ligand.source_path),
                "sdf_molecule_index": ligand.molecule_index,
                "sdf_name": ligand.raw_name,
                "batch_key": ligand.batch_key,
                "bond_orders": bond_orders,
            },
        )
        components.append(component)

        heavy_mapping = {
            rdkit_idx: atom_id
            for rdkit_idx, atom_id in atom_ids_by_rdkit_idx.items()
            if atoms_by_id[atom_id].element != "H"
        }
        apply_rdkit_ligand_template(
            atoms,
            component,
            molecule,
            heavy_mapping,
            smiles=ligand.smiles,
            source="sdf",
        )
        smiles_by_component_id[component_id] = ligand.smiles

    return smiles_by_component_id


def _as_paths(value: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def _raw_molecule_name(molecule, path: Path, molecule_index: int) -> str:
    if molecule.HasProp("_Name"):
        name = molecule.GetProp("_Name").strip()
        if name:
            return name
    if molecule_index == 0:
        return path.stem
    return f"{path.stem}_{molecule_index + 1}"


def _component_name(raw_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name.strip()).strip("_").upper()
    return name or "LIG"


def _atom_name(rdkit_atom, rdkit_idx: int) -> str:
    if rdkit_atom.HasProp("_TriposAtomName"):
        name = rdkit_atom.GetProp("_TriposAtomName").strip()
        if name:
            return name
    return f"{normalize_element(rdkit_atom.GetSymbol())}{rdkit_idx + 1}"


def _sdf_smiles(molecule) -> str:
    from rdkit import Chem

    smiles_molecule = Chem.Mol(molecule)
    try:
        smiles_molecule = Chem.RemoveHs(smiles_molecule)
    except Exception:
        pass
    return Chem.MolToSmiles(smiles_molecule, isomericSmiles=True)


def _next_component_id(
    ligand_name: str,
    chain_id: str,
    start_number: int,
    existing_component_ids: set[str],
) -> tuple[str, str]:
    residue_number = start_number
    while True:
        residue_id = str(residue_number)
        component_id = f"{chain_id}:{ligand_name}:{residue_id}"
        if component_id not in existing_component_ids:
            return residue_id, component_id
        residue_number += 1
