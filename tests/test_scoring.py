import json

import pytest

from chemur import cli
import chemur.scoring as scoring_mod
from chemur.cosplif import cosplif_profile
from chemur.scoring import score_pose_batch


def _rec(interaction_id, ligand_id, family, target, *, donor=True):
    return {
        "interaction_id": interaction_id,
        "ligand_id": ligand_id,
        "target_id": "rec",
        "interaction_family": family,
        "interaction_subtype": None,
        "ligand_feature_id": "F1",
        "ligand_atom_ids": [1],
        "target_entity_kind": "residue",
        "target_entity_id": target,
        "target_moiety": "sidechain",
        "target_atom_ids": [10],
        "geometry": {"distance": 2.9, "angle": 160.0},
        "directionality": {"ligand_is_donor": donor, "target_is_donor": not donor},
        "strength_weight": 1.0,
        "confidence": 1.0,
    }


def _complex(ligand_id, tokens, records):
    return {
        "ligand_id": ligand_id,
        "target_id": "rec",
        "interaction_records": records,
        "cosplif": {
            "version": "cosplif_context_v1",
            "tokens": tokens,
            "ligand_blocks_v1": {
                "version": "ligand_blocks_v1",
                "ligand_id": ligand_id,
                "heavy_atom_count": 0,
                "blocks": [],
                "warnings": [],
            },
        },
    }


def _batch():
    shared = {"t1": 1, "t2": 1, "t3": 1}
    return {
        "results": {
            "ref": _complex(
                "ref",
                shared,
                [
                    _rec("r1", "ref", "hydrogen_bond", "A:ASP:25"),
                    _rec("r2", "ref", "hydrophobic", "A:LEU:30", donor=False),
                ],
            ),
            "near": _complex(
                "near",
                shared,
                [
                    _rec("n1", "near", "hydrogen_bond", "A:ASP:25"),
                    _rec("n2", "near", "hydrophobic", "A:LEU:30", donor=False),
                ],
            ),
            "far": _complex(
                "far",
                {"t9": 1},
                [_rec("f1", "far", "hydrogen_bond", "Z:GLU:99")],
            ),
        }
    }


def test_default_pose_profile_is_bw040_hybrid():
    profile = cosplif_profile("cosplif_pose_v1")
    assert profile.score_mode == "hybrid_ifm"
    assert profile.ligand_block_weight == 0.40


def test_score_pose_batch_ranks_and_blends_ifm():
    rows = score_pose_batch(_batch(), "ref")  # default profile cosplif_pose_v1, python engine
    by_ligand = {row["ligand_id"]: row for row in rows}

    # a pose matching the reference scores above a dissimilar pose
    assert by_ligand["near"]["cosplif_score"] > by_ligand["far"]["cosplif_score"]
    # the reference self-match is among the top scores
    assert by_ligand["ref"]["cosplif_score"] == max(r["cosplif_score"] for r in rows)
    # the IFM hybrid blend ran and rows are ranked
    assert all("strict_ifm_similarity" in row and "rank" in row for row in rows)
    assert [row["rank"] for row in rows] == sorted(row["rank"] for row in rows)


def test_cli_score_routes_to_score_poses(monkeypatch, tmp_path):
    captured = {}

    def fake_score_poses(structure, **kwargs):
        captured["structure"] = structure
        captured["kwargs"] = kwargs
        return [{"rank": 1, "ligand_id": "p1", "cosplif_score": 0.9}]

    monkeypatch.setattr(scoring_mod, "score_poses", fake_score_poses)

    out_path = tmp_path / "scores.json"
    status = cli.main(
        [
            "score",
            "receptor.pdb",
            "--ligand-sdf-dir",
            "poses",
            "--reference-sdf",
            "ref.sdf",
            "--out",
            str(out_path),
        ]
    )

    assert status == 0
    assert captured["structure"] == "receptor.pdb"
    assert captured["kwargs"]["ligand_sdf_dir"] == ["poses"]
    assert captured["kwargs"]["reference_sdf"] == "ref.sdf"
    assert captured["kwargs"]["profile"] == "cosplif_pose_v1"
    assert captured["kwargs"]["ifm_engine"] == "python"
    assert json.loads(out_path.read_text())[0]["ligand_id"] == "p1"


def test_cli_score_requires_a_reference():
    with pytest.raises(SystemExit):
        cli.main(["score", "receptor.pdb", "--ligand-sdf-dir", "poses"])


def test_cli_score_requires_poses():
    with pytest.raises(SystemExit):
        cli.main(["score", "receptor.pdb", "--reference-sdf", "ref.sdf"])
