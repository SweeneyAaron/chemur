from chemeleonx.models import (
    AnalysisResult,
    AssignedInteraction,
    AtomRecord,
    ComponentRecord,
    FeatureRecord,
)


def test_split_by_ligand_filters_interactions_and_context(tmp_path):
    atoms = [
        AtomRecord(0, "C1", "C", (0.0, 0.0, 0.0), "L1", "1", "A", "A:L1:1", "ligand"),
        AtomRecord(1, "O", "O", (2.8, 0.0, 0.0), "ASP", "2", "A", "A:ASP:2", "protein"),
        AtomRecord(2, "C1", "C", (5.0, 0.0, 0.0), "L2", "3", "B", "B:L2:3", "ligand"),
    ]
    components = [
        ComponentRecord("A:L1:1", "L1", "A", "1", "ligand", (0,)),
        ComponentRecord("A:ASP:2", "ASP", "A", "2", "protein", (1,)),
        ComponentRecord("B:L2:3", "L2", "B", "3", "ligand", (2,)),
    ]
    features = [
        FeatureRecord(0, "donor", "A:L1:1", (0,), (0.0, 0.0, 0.0)),
        FeatureRecord(1, "acceptor", "A:ASP:2", (1,), (2.8, 0.0, 0.0)),
        FeatureRecord(2, "hydrophobe", "B:L2:3", (2,), (5.0, 0.0, 0.0)),
    ]
    result = AnalysisResult(
        atoms=atoms,
        components=components,
        features=features,
        interactions=[
            AssignedInteraction("hbond", (0, 1), (0, 1), ("A:L1:1", "A:ASP:2"), 2.8, 30, "inter", 1),
        ],
    )

    split = result.split_by_ligand()

    assert set(split) == {"A:L1:1", "B:L2:3"}
    assert len(split["A:L1:1"].interactions) == 1
    assert len(split["B:L2:3"].interactions) == 0
    assert {component.component_id for component in split["A:L1:1"].components} == {"A:L1:1", "A:ASP:2"}

    written = result.write_ligand_outputs(tmp_path, output_format="json")
    assert {path.name for path in written} == {"A_L1_1.json", "B_L2_3.json"}

