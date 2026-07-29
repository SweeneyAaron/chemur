class ChemurError(Exception):
    """Base exception for standalone Chemur."""


class DependencyMissingError(ChemurError):
    """Raised when an optional runtime dependency is required but unavailable."""


class LigandTemplateError(ChemurError):
    """Raised when a ligand SMILES template cannot be mapped safely."""


class StructureParseError(ChemurError):
    """Raised when a PDB/mmCIF structure cannot be parsed."""

