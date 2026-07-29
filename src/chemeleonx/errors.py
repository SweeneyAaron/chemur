class ChemeleonXError(Exception):
    """Base exception for standalone ChemeleonX."""


class DependencyMissingError(ChemeleonXError):
    """Raised when an optional runtime dependency is required but unavailable."""


class LigandTemplateError(ChemeleonXError):
    """Raised when a ligand SMILES template cannot be mapped safely."""


class StructureParseError(ChemeleonXError):
    """Raised when a PDB/mmCIF structure cannot be parsed."""

