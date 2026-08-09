"""Pre-run guards for the ARKitScenes x Mask3D execution contract.

Contract: `docs/arkitscenes_mask3d_contract.md`.

These assert the things that are cheap to get wrong and expensive to notice
after a GPU run: gate values drifting from the approved contract, the
reporting grid silently becoming the Replica grid, the notebook's pinned
mesh hash not matching the mesh the local side evaluates, and a scratch dry
run being able to write into the live results directory.

Runs with or without the dataset present.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import arkitscenes_mask3d_eval as M
from segmenter.base import SegmentationOutput, save_segmentation_output

NB = REPO_ROOT / "notebooks" / "c1_mask3d_colab.ipynb"
CONTRACT = REPO_ROOT / "docs" / "arkitscenes_mask3d_contract.md"
SCENE_KEY = "arkitscenes_41069021"
MESH_SHA = "ec219f56c1f9d79a17f4ba0a224d19f75188aa38accf6a4074283d5f66c70d0b"
N_VERTICES = 1008964


def _nb_src() -> str:
    nb = json.loads(NB.read_text())
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_gates_match_the_approved_contract() -> None:
    """Approved 2026-08-08: stop <=3, pass >=4, strong >=6, on Mask3D alone
    at IoU 0.50. Beating 1/18 was explicitly rejected as success because M1
    already reached 2/18."""
    if (M.GATE_STOP, M.GATE_PASS, M.GATE_STRONG) != (3, 4, 6):
        raise AssertionError(
            f"gates drifted: stop={M.GATE_STOP} pass={M.GATE_PASS} "
            f"strong={M.GATE_STRONG}, approved (3, 4, 6)")
    if M.GATE_STOP >= M.GATE_PASS:
        raise AssertionError("stop and pass bands overlap")


def test_resolve_config_is_the_replica_operating_point() -> None:
    if (M.MIN_SCORE, M.MIN_VERTICES) != (0.2, 20):
        raise AssertionError(
            f"contract fixes MIN_SCORE=0.2/min_vertices=20, got "
            f"{M.MIN_SCORE}/{M.MIN_VERTICES}")


def test_reporting_grid_is_the_contracts_not_replicas() -> None:
    """The contract reports IoU 0.10/0.25/0.50 and k=25/50/100/200. The
    Replica tool's grid is different; importing it would let an edit there
    silently redefine what this contract reported."""
    if tuple(M.IOUS) != (0.10, 0.25, 0.50):
        raise AssertionError(f"IoU grid changed: {M.IOUS}")
    if tuple(M.KS) != (25, 50, 100, 200, None):
        raise AssertionError(f"k grid changed: {M.KS}")
    from tools import p1_selector_eval as P
    if tuple(M.KS) == tuple(P.KS) and tuple(M.IOUS) == tuple(P.IOU_THRESHOLDS):
        raise AssertionError(
            "the contract grid is now identical to the Replica grid; if that "
            "is intended, delete this guard deliberately rather than letting "
            "the coupling reappear by accident")


def test_notebook_pins_the_arkitscenes_mesh() -> None:
    src = _nb_src()
    if SCENE_KEY not in src:
        raise AssertionError("notebook has no ARKitScenes scene entry")
    if MESH_SHA not in src:
        raise AssertionError(
            "notebook does not pin the canonical mesh sha256; cell [4]'s "
            "hard gate is what stops a wrong SCENE from running silently")
    if str(N_VERTICES) not in src:
        raise AssertionError(f"notebook does not pin {N_VERTICES} vertices")


def test_notebook_still_pins_every_replica_scene() -> None:
    """Adding ARKitScenes must not have disturbed the frozen Replica pins."""
    src = _nb_src()
    for scene, sha in (
            ("room_1", "21695deccc1fe76051d90178eccc1609ee1bab8b5dc715683dd17f7903cf6ee0"),
            ("room_2", "e58a7c717c7922e1300ba20ae8053c5dbfdf9bd5f2515e10c71edad98bcb7e44"),
            ("office_0", "cdb6ede0b9d455f491ef8fd63cd916a86a505b777842aabd6aa428edf9ff9032"),
            ("frl_apartment_0", "459374364b1fb6d61b28809fb2ebb722366ffc055caf990a5d659b1ebdd3e71b")):
        if sha not in src:
            raise AssertionError(f"Replica pin for {scene} was disturbed")


def test_notebook_upstream_params_are_untouched() -> None:
    """The contract fixes the model configuration; only SCENE and the pin
    tables were allowed to change."""
    src = _nb_src()
    for frozen in ("NUM_QUERIES = 150", "USE_DBSCAN = 'true'",
                   "DBSCAN_EPS = 0.95", "MIN_VERTICES = 20",
                   "3bc3fc52693b25668d0e91d55a2ea714544a4749"):
        if frozen not in src:
            raise AssertionError(f"{frozen!r} is no longer pinned")


def test_pinned_mesh_matches_the_mesh_the_evaluator_reads() -> None:
    """If the dataset is present, the notebook's pin and the local canonical
    mesh must be the same bytes -- otherwise the GPU segments one mesh and
    the evaluator scores against another."""
    from tools.arkitscenes_eval import DEFAULT_DATA_ROOT
    mesh = DEFAULT_DATA_ROOT / "41069021" / "41069021_3dod_mesh_canonical.ply"
    if not mesh.is_file():
        print("  SKIP (dataset not present)")
        return
    from segmenter.base import sha256_file
    actual = sha256_file(mesh)
    if actual != MESH_SHA:
        raise AssertionError(
            f"canonical mesh is {actual[:16]}…, notebook pins "
            f"{MESH_SHA[:16]}… — re-run the adapter or update the pin")


def test_report_is_written_beside_its_bundle() -> None:
    """A dry run against a scratch --bundle-root must not be able to drop a
    report into runs/arkitscenes_mask3d/, where a synthetic result would sit
    among real ones."""
    src = (REPO_ROOT / "tools" / "arkitscenes_mask3d_eval.py").read_text()
    if "OUT_ROOT /" in src or "OUT_ROOT.mkdir" in src:
        raise AssertionError(
            "report path is a fixed root again; it must follow "
            "args.bundle_root")
    if "args.bundle_root / f\"{scene_id}_mask3d_eval.json\"" not in src:
        raise AssertionError("report is not written beside the bundle")


def test_contract_doc_exists_and_states_the_gates() -> None:
    if not CONTRACT.is_file():
        raise AssertionError("contract doc is missing")
    text = CONTRACT.read_text()
    for needed in ("41069021", "MIN_SCORE=0.2", "4/18", "6/18", "3/18"):
        if needed not in text:
            raise AssertionError(f"contract doc does not state {needed!r}")


def test_evaluator_never_reads_annotations_itself() -> None:
    """Oracle boundary: annotations are readable only via
    tools.arkitscenes_eval, which owns that permission.

    AST, not substring. A grep flags the module docstring, which names the
    annotation file precisely in order to state the boundary it keeps --
    penalising the documentation would be the wrong incentive. What matters
    is whether any STRING THE CODE EVALUATES names it.
    """
    import ast
    path = REPO_ROOT / "tools" / "arkitscenes_mask3d_eval.py"
    tree = ast.parse(path.read_text())
    docstrings = {id(ast.get_docstring(n, clean=False))
                  for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef,
                                    ast.AsyncFunctionDef, ast.ClassDef))}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node.value) not in docstrings
                and "_3dod_annotation" in node.value):
            raise AssertionError(
                f"line {node.lineno}: code names the annotation file "
                f"directly; it must go through "
                f"tools.arkitscenes_eval.load_oracle_entities")


def test_delivered_partition_loads_exact_saved_assignment() -> None:
    """The delivered check must not re-resolve, filter, or renumber masks."""
    assignment = np.asarray([-1, 7, 7, 3, 3, 3, -1], dtype=np.int64)
    seg = SegmentationOutput(
        input_mesh_sha256="abc", n_vertices=len(assignment),
        segmenter_name="synthetic", segmenter_version="1",
        config_params_json=json.dumps({"min_score": 0.4,
                                       "min_vertices": 2}),
        vertex_instance_ids=assignment,
        instance_confidence={3: 0.9, 7: 0.8},
    ).finalize()
    with tempfile.TemporaryDirectory() as d:
        save_segmentation_output(seg, Path(d))
        proposals, ids, provenance = M.load_delivered_partition(
            Path(d), len(assignment), "abc")
    if ids != [3, 7]:
        raise AssertionError(f"instance ids were renumbered: {ids}")
    if [p.tolist() for p in proposals] != [[3, 4, 5], [1, 2]]:
        raise AssertionError(f"assignment was changed: {proposals}")
    if provenance["resolve_config"] != {"min_score": 0.4,
                                         "min_vertices": 2}:
        raise AssertionError(f"artifact config lost: {provenance}")


def test_delivered_partition_reports_effective_local_reresolve() -> None:
    """A local operating-point artifact must not report the Colab cutoff."""
    assignment = np.asarray([-1, 7, 7, 3, 3, 3, -1], dtype=np.int64)
    seg = SegmentationOutput(
        input_mesh_sha256="abc", n_vertices=len(assignment),
        segmenter_name="synthetic", segmenter_version="1",
        config_params_json=json.dumps({
            "min_score": 0.4,
            "min_vertices": 2,
            "reresolved_locally": {
                "min_score": 0.2,
                "min_vertices": 2,
                "source_bundle_output_sha256": "source",
            },
        }),
        vertex_instance_ids=assignment,
        instance_confidence={3: 0.9, 7: 0.8},
    ).finalize()
    with tempfile.TemporaryDirectory() as d:
        save_segmentation_output(seg, Path(d))
        _proposals, _ids, provenance = M.load_delivered_partition(
            Path(d), len(assignment), "abc")
    if provenance["resolve_config"] != {"min_score": 0.2,
                                         "min_vertices": 2}:
        raise AssertionError(f"effective local config lost: {provenance}")
    if provenance["original_resolve_config"]["min_score"] != 0.4:
        raise AssertionError(f"original config lost: {provenance}")
    if provenance["reresolved_locally"] is not True:
        raise AssertionError(f"local re-resolution not disclosed: {provenance}")


def test_delivered_report_is_comparable_to_proposal_ceiling() -> None:
    proposals = [np.asarray([0, 1]), np.asarray([2, 3]),
                 np.asarray([6])]
    # instances 10 and 11 are perfect for the two entities; 12 is junk.
    ious = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    report = M.delivered_partition_report(
        proposals, [10, 11, 12], ious, n_vertices=10,
        provenance={"n_instances": 3, "unassigned_vertex_rate": 0.5},
        proposal_ceiling={"0.10": 2, "0.25": 2, "0.50": 2})
    if report["entity_recovery"] != {"0.10": 2, "0.25": 2, "0.50": 2}:
        raise AssertionError(f"wrong delivered recovery: {report}")
    if report["ceiling_retention"]["0.50"] != 1.0:
        raise AssertionError(f"wrong ceiling retention: {report}")
    if report["matched_instances"]["0.50"] != 2:
        raise AssertionError(f"wrong matched-instance count: {report}")
    if report["oracle_matched_instance_rate"]["0.50"] != round(2 / 3, 4):
        raise AssertionError(f"wrong oracle-matched rate: {report}")
    if report["zero_overlap_rate"] != round(1 / 3, 4):
        raise AssertionError(f"wrong junk rate: {report}")


def test_per_entity_delivery_identifies_resolution_loss() -> None:
    class Entity:
        def __init__(self, uid, label):
            self.uid, self.label = uid, label
    entities = [Entity("a", "chair"), Entity("b", "cabinet")]
    rows = M.per_entity_delivery(
        entities,
        proposal_ious=np.asarray([[0.8, 0.0], [0.0, 0.7]]),
        delivered_ious=np.asarray([[0.75, 0.0], [0.0, 0.4]]),
        delivered_ids=[4, 9])
    if rows[0]["lost_in_resolution_at_050"]:
        raise AssertionError(f"surviving entity marked lost: {rows[0]}")
    if not rows[1]["lost_in_resolution_at_050"]:
        raise AssertionError(f"lost entity was not identified: {rows[1]}")
    if rows[1]["best_delivered_instance_id"] != 9:
        raise AssertionError(f"wrong best instance attribution: {rows[1]}")


TESTS = [
    test_gates_match_the_approved_contract,
    test_resolve_config_is_the_replica_operating_point,
    test_reporting_grid_is_the_contracts_not_replicas,
    test_notebook_pins_the_arkitscenes_mesh,
    test_notebook_still_pins_every_replica_scene,
    test_notebook_upstream_params_are_untouched,
    test_pinned_mesh_matches_the_mesh_the_evaluator_reads,
    test_report_is_written_beside_its_bundle,
    test_contract_doc_exists_and_states_the_gates,
    test_evaluator_never_reads_annotations_itself,
    test_delivered_partition_loads_exact_saved_assignment,
    test_delivered_partition_reports_effective_local_reresolve,
    test_delivered_report_is_comparable_to_proposal_ceiling,
    test_per_entity_delivery_identifies_resolution_loss,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
