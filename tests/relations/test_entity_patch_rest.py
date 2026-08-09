"""Synthetic tests for target-to-entity-patch resting evidence.

Run: python3 tests/relations/test_entity_patch_rest.py
"""
from __future__ import annotations

from dataclasses import replace
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane, SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
)
from geometry.entity_support_patches import (
    EntityHorizontalPatchEstimate, HorizontalPatchConfig,
    HorizontalPatchEvidence, OwnerPatchEvidence,
)
from graph.relations.entity_patch_rest import (
    EntityPatchRestConfig, evaluate_entity_patch_resting,
)


def _entity(uid: str, instance: int, lo, hi, label: str = "anonymous"):
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid, display_label=label,
            source_instance_ref=f"segmenter:{instance}",
        ),
        bbox_aabb=(tuple(lo), tuple(hi)), bbox_obb=None,
        centroid=tuple((lo[i] + hi[i]) / 2.0 for i in range(3)),
        geometry_handle=None,
    )


def _patch(instance: int, uid: str, x0: float, y0: float,
           x1: float, y1: float, z: float,
           *, qualifies: bool = True) -> HorizontalPatchEvidence:
    area = (x1 - x0) * (y1 - y0)
    return HorizontalPatchEvidence(
        patch_uid=uid, owner_instance_id=instance,
        plane=Plane(a=0.0, b=0.0, c=1.0, d=-z),
        normal=(0.0, 0.0, 1.0),
        polygon=[(x0, y0, z), (x1, y0, z),
                 (x1, y1, z), (x0, y1, z)],
        footprint_kind="convex_hull_on_fitted_plane",
        height_m=z, mesh_area_m2=area, projected_area_m2=area,
        footprint_area_m2=area, coverage_ratio=1.0,
        projected_area_ratio_owner_bbox=1.0,
        roughness_rms_m=0.0, roughness_ratio_owner_diagonal=0.0,
        tilt_to_up_deg=0.0, n_faces=2, n_vertices=4,
        source_face_ids_sha256=uid * 4,
        geometry_qualifies=qualifies,
        geometry_rejection_reasons=(() if qualifies else ("synthetic_reject",)),
    )


def _owner(instance: int, lo, hi, patches) -> OwnerPatchEvidence:
    dx, dy, dz = (hi[i] - lo[i] for i in range(3))
    horizontal = (dx * dx + dy * dy) ** 0.5
    diagonal = (dx * dx + dy * dy + dz * dz) ** 0.5
    return OwnerPatchEvidence(
        owner_instance_id=instance, n_assigned_vertices=4,
        n_strict_faces=2, n_horizontal_faces=2,
        n_horizontal_components=len(patches), n_components_below_face_min=0,
        bbox_aabb=(tuple(lo), tuple(hi)),
        horizontal_diagonal_m=horizontal, owner_diagonal_m=diagonal,
        owner_xy_bbox_area_m2=dx * dy, patches=list(patches),
    )


def _scene(scale: float = 1.0, *, labels: bool = False,
           floor_qualifies: bool = True):
    def p(values):
        return tuple(scale * value for value in values)

    entities = [
        _entity("obj_floor", 0, p((0, 0, 0)), p((10, 10, 0)),
                "floor" if labels else "segment_0"),
        _entity("obj_table", 1, p((2, 2, 0)), p((4, 4, 1)),
                "table" if labels else "segment_1"),
        _entity("obj_target", 2, p((2.5, 2.5, 1)), p((3, 3, 1.3)),
                "plate" if labels else "segment_2"),
        _entity("obj_platform", 3, p((6, 6, 0)), p((8, 8, 0.1)),
                "cabinet" if labels else "segment_3"),
        _entity("obj_floor_target", 4, p((6.5, 6.5, 0)), p((7, 7, 0.3)),
                "vase" if labels else "segment_4"),
        _entity("obj_high", 5, p((2.5, 2.5, 1.1)), p((3, 3, 1.4)),
                "book" if labels else "segment_5"),
        _entity("obj_penetrating", 6, p((2.5, 2.5, 0.9)), p((3, 3, 1.2)),
                "cup" if labels else "segment_6"),
    ]
    floor_patch = _patch(0, "floor_patch", 0, 0, 10 * scale, 10 * scale,
                         0, qualifies=floor_qualifies)
    # _patch's x/y inputs are absolute; scale non-zero lower coordinates too.
    table_patch = _patch(1, "table_patch", 2 * scale, 2 * scale,
                         4 * scale, 4 * scale, 1 * scale)
    platform_patch = _patch(3, "platform_patch", 6 * scale, 6 * scale,
                            8 * scale, 8 * scale, 0)
    owners = [
        _owner(0, p((0, 0, 0)), p((10, 10, 0)), [floor_patch]),
        _owner(1, p((2, 2, 0)), p((4, 4, 1)), [table_patch]),
        _owner(3, p((6, 6, 0)), p((8, 8, 0.1)), [platform_patch]),
    ]
    artifacts = EntityArtifacts(
        schema_version=2, bundle_hash="synthetic", scene_id="synthetic",
        frame=SceneFrame(
            gravity=(0.0, 0.0, -1.0), canonical_forward=None,
            canonical_right=None, units="meters", notes="synthetic",
            kind="scene_canonical",
        ),
        representation_hash="repr", extractor_name="synthetic",
        extractor_version="0.1", entities=entities,
        structural_surfaces=[], geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities), n_structural_surfaces=0,
            runtime_seconds=0.0, coverage_score=None, notes="synthetic",
        ), notes={"oracle_free": True},
    )
    patches = EntityHorizontalPatchEstimate(
        owners=owners, config=HorizontalPatchConfig(),
        diagnostics={"uses_semantics": False, "uses_oracle": False},
    )
    return artifacts, patches


def _pair(result, target: str, owner: str):
    matches = [p for p in result.pairs
               if p.target_entity_uid == target and p.owner_entity_uid == owner]
    if len(matches) != 1:
        raise AssertionError(f"expected one pair {target}->{owner}, got {len(matches)}")
    return matches[0]


def test_resting_target_has_overlap_gap_and_scale_evidence() -> None:
    artifacts, patches = _scene()
    result = evaluate_entity_patch_resting(artifacts, patches)
    pair = _pair(result, "obj_target", "obj_table")
    if not pair.relation_candidate or pair.selected_patch_uid != "table_patch":
        raise AssertionError("clear tabletop resting evidence was rejected")
    evidence = pair.evaluated_patches[0]
    if evidence.overlap_ratio_target != 1.0:
        raise AssertionError("fully contained target did not have full overlap")
    if not evidence.target_footprint_contained:
        raise AssertionError("containment evidence missing")
    if abs(evidence.vertical_gap_at_overlap_centroid_m or 0.0) > 1e-12:
        raise AssertionError("resting target should have zero vertical gap")
    if evidence.floor_contact_state != "clear":
        raise AssertionError("elevated target was mistaken for floor contact")
    if evidence.target_to_owner_diagonal_ratio <= 0.0:
        raise AssertionError("relative-scale evidence missing")


def test_floor_contact_blocks_otherwise_matching_low_patch() -> None:
    artifacts, patches = _scene()
    result = evaluate_entity_patch_resting(artifacts, patches)
    pair = _pair(result, "obj_floor_target", "obj_platform")
    if pair.relation_candidate:
        raise AssertionError("floor-supported target survived floor exclusion")
    evidence = pair.evaluated_patches[0]
    if evidence.floor_contact_state != "contact":
        raise AssertionError("geometry floor census missed target contact")
    if "target_floor_contact" not in evidence.rejection_reasons:
        raise AssertionError("floor rejection reason is not explicit")


def test_spatial_failure_reasons_are_separate() -> None:
    artifacts, patches = _scene()
    result = evaluate_entity_patch_resting(artifacts, patches)
    far = _pair(result, "obj_target", "obj_platform").evaluated_patches[0]
    if "footprint_overlap_below_min" not in far.rejection_reasons:
        raise AssertionError("missing footprint rejection")
    high = _pair(result, "obj_high", "obj_table").evaluated_patches[0]
    if "vertical_gap_above_contact_band" not in high.rejection_reasons:
        raise AssertionError("missing above-band rejection")
    penetrating = _pair(
        result, "obj_penetrating", "obj_table",
    ).evaluated_patches[0]
    if "vertical_gap_below_penetration_band" not in penetrating.rejection_reasons:
        raise AssertionError("missing penetration rejection")
    self_pair = _pair(result, "obj_table", "obj_table")
    if self_pair.rejection_reasons != ("self_support_excluded",):
        raise AssertionError("self-support rejection drifted")


def test_uniform_scale_preserves_candidate_and_ratios() -> None:
    artifacts_a, patches_a = _scene(1.0)
    artifacts_b, patches_b = _scene(10.0)
    a = _pair(evaluate_entity_patch_resting(artifacts_a, patches_a),
              "obj_target", "obj_table").evaluated_patches[0]
    b = _pair(evaluate_entity_patch_resting(artifacts_b, patches_b),
              "obj_target", "obj_table").evaluated_patches[0]
    if a.relation_candidate != b.relation_candidate:
        raise AssertionError("uniform scaling changed candidate decision")
    for name in ("overlap_ratio_target", "overlap_ratio_patch",
                 "target_to_owner_diagonal_ratio",
                 "vertical_gap_ratio_relation_scale"):
        if abs(float(getattr(a, name)) - float(getattr(b, name))) > 1e-9:
            raise AssertionError(f"uniform scaling changed {name}")


def test_labels_cannot_change_evidence() -> None:
    anonymous, patches = _scene(labels=False)
    labeled, _ = _scene(labels=True)
    a = evaluate_entity_patch_resting(anonymous, patches).to_dict()
    b = evaluate_entity_patch_resting(labeled, patches).to_dict()
    if a != b:
        raise AssertionError("display labels changed geometry-only evidence")
    if a["diagnostics"]["uses_labels"] is not False:
        raise AssertionError("output does not declare label isolation")
    json.dumps(a, sort_keys=True, allow_nan=False)


def test_unknown_floor_state_is_visible_and_optionally_conservative() -> None:
    artifacts, patches = _scene(floor_qualifies=False)
    permissive = evaluate_entity_patch_resting(artifacts, patches)
    pair = _pair(permissive, "obj_target", "obj_table")
    if not pair.relation_candidate:
        raise AssertionError("default unknown-floor policy should remain explicit, not reject")
    if pair.evaluated_patches[0].floor_contact_state != "unknown":
        raise AssertionError("missing floor must report unknown")
    strict = evaluate_entity_patch_resting(
        artifacts, patches,
        config=replace(EntityPatchRestConfig(), reject_when_floor_unknown=True),
    )
    evidence = _pair(strict, "obj_target", "obj_table").evaluated_patches[0]
    if evidence.relation_candidate:
        raise AssertionError("strict unknown-floor policy did not reject")
    if "target_floor_contact_unknown" not in evidence.rejection_reasons:
        raise AssertionError("unknown-floor rejection reason missing")


TESTS = [
    test_resting_target_has_overlap_gap_and_scale_evidence,
    test_floor_contact_blocks_otherwise_matching_low_patch,
    test_spatial_failure_reasons_are_separate,
    test_uniform_scale_preserves_candidate_and_ratios,
    test_labels_cannot_change_evidence,
    test_unknown_floor_state_is_visible_and_optionally_conservative,
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
