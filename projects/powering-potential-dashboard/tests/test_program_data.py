"""Golden-value and data-quality tests for the Program dashboard data build.

Run with `pytest` from powering-potential-dashboard/ (after `pip install -r
requirements.txt`). These tests exist to protect the specific numbers the
board is relying on (e.g. Karatu 23 -> "11 of 23") from silently changing
when the underlying timeline xlsx or taxonomy is edited, and to catch
unmerged school-name duplicates before they reach the dashboard.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_program_data import (  # noqa: E402
    Builder,
    TaxonomyError,
    build_payload,
    check_for_unmerged_duplicates,
    load_taxonomy,
    write_canonical_schools_csv,
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return build_payload()


@pytest.fixture(scope="module")
def taxonomy() -> dict:
    return load_taxonomy()


@pytest.fixture()
def builder(taxonomy) -> Builder:
    return Builder(taxonomy)


# --- Golden values: the exact facts the board has been told -----------------


def test_karatu_23_progress_label_is_11_of_23(payload):
    assert payload["karatu23"]["progressLabel"] == "11 of 23"
    assert payload["karatu23"]["denominator"] == 23
    assert payload["karatu23"]["progressForBar"] == 11


def test_karatu_23_is_kdp1_plus_kdp2(payload):
    k23 = payload["karatu23"]
    assert k23["kdp1EquipmentCount"] == 7
    assert k23["kdp2Named"] == 4
    assert k23["installedNamed"] == k23["kdp1Named"] + k23["kdp2Named"]


def test_original_karatu_6_are_excluded_from_the_23(payload):
    original = set(payload["karatu23"]["originalBase"])
    assert original == {"Banjika", "Welwel", "Florian", "Slahamo", "Endallah", "Baray"}
    karatu23_names = {row["name"] for row in payload["karatu23"]["rows"]}
    assert original.isdisjoint(karatu23_names), (
        "Original 6 Karatu schools must never appear in the Karatu 23 rows "
        "(AJ Poole Key tab: installed base is not part of the 23)."
    )
    for school in payload["schools"]:
        if school["canonicalName"] in original:
            assert school["inKaratu23"] is False


def test_canonical_school_count_is_stable(payload):
    # 47 as of the 2007-2025 timeline extract. If this changes, confirm the
    # change is expected (new school added/removed) before updating the
    # expected count here.
    assert payload["network"]["canonicalSchools"] == 47


def test_training_is_never_plus_one_school(payload):
    contract = {item["metric"]: item["definition"] for item in payload["metricContract"]}
    assert "Training" in contract
    assert "not +1 school" in contract["Training"] or "never +1 school" in contract["Training"].lower()


def test_public_headline_figures_are_flagged_not_endorsed(payload):
    assert any("42K" in m["title"] or "130" in m["detail"] or "42" in m["detail"] for m in payload["mismatches"]), (
        "Expected a mismatch entry noting the public 130/42K/50%/58% figures "
        "are not yet backed by these tables."
    )


# --- Structural invariants ----------------------------------------------


def test_every_school_has_required_fields(payload):
    required = {"schoolId", "canonicalName", "cluster", "status", "siteType"}
    for school in payload["schools"]:
        missing = required - school.keys()
        assert not missing, f"{school.get('canonicalName')} is missing fields: {missing}"


def test_no_duplicate_school_ids(payload):
    ids = [s["schoolId"] for s in payload["schools"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"Duplicate schoolId values: {dupes}"


def test_known_discrepancies_have_owner_and_status(payload):
    assert payload["knownDiscrepancies"], "expected at least one known discrepancy to be tracked"
    for item in payload["knownDiscrepancies"]:
        for field in ("id", "severity", "title", "detail", "owner", "status"):
            assert item.get(field), f"known discrepancy {item.get('id')} missing {field}"


def test_generated_at_is_iso8601_utc(payload):
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", payload["generatedAt"])


# --- Data-quality gate ----------------------------------------------------


def test_no_unmerged_name_duplicates_in_current_taxonomy(payload, taxonomy):
    canonical_names = [s["canonicalName"] for s in payload["schools"]]
    suspects = check_for_unmerged_duplicates(canonical_names, taxonomy)
    assert not suspects, (
        "Possible unmerged school-name duplicates: "
        f"{suspects}. Add an alias or a confirmed_distinct_pairs entry in "
        "data/school-taxonomy.yml."
    )


def test_duplicate_detector_actually_catches_variants():
    """Sanity check that the detector isn't a no-op (catches an obvious typo)."""
    suspects = check_for_unmerged_duplicates(["Banjika", "Banjik", "Unrelated School"], {})
    assert suspects, "detector should flag 'Banjika' vs 'Banjik' as a likely duplicate"


def test_confirmed_distinct_pairs_are_silenced(taxonomy):
    pair = taxonomy["confirmed_distinct_pairs"][0]["names"]
    suspects = check_for_unmerged_duplicates(pair, taxonomy)
    assert not suspects, "confirmed_distinct_pairs entries must not be flagged"


# --- Reproducibility: committed program-data.js matches a fresh build ----


def test_committed_program_data_matches_fresh_build(payload):
    """Guards against hand-editing program-data.js instead of regenerating it.

    Ignores `generatedAt`, since that legitimately changes on every build.
    """
    committed_path = ROOT / "program-data.js"
    text = committed_path.read_text()
    prefix = "window.PROGRAM_DATA = "
    assert text.startswith(prefix)
    committed = json.loads(text[len(prefix):].rstrip("\n;"))

    fresh = json.loads(json.dumps(payload))  # normalize (Counters -> plain ints already done by build)
    committed.pop("generatedAt", None)
    fresh.pop("generatedAt", None)

    assert committed == fresh, (
        "program-data.js does not match a fresh run of build_program_data.py. "
        "Regenerate it with `python3 build_program_data.py` and commit the result."
    )


def test_committed_canonical_schools_csv_matches_fresh_build(payload, tmp_path):
    """Guards against the picklist (used for spreadsheet data validation)
    drifting from what the dashboard actually recognizes.
    """
    fresh_path = tmp_path / "canonical-schools.csv"
    write_canonical_schools_csv(payload["schools"], path=fresh_path)
    committed_path = ROOT / "canonical-schools.csv"
    assert committed_path.read_text() == fresh_path.read_text(), (
        "canonical-schools.csv does not match a fresh run of build_program_data.py. "
        "Regenerate it with `python3 build_program_data.py` and commit the result."
    )


# --- Ongoing data collection: data/activity-log.csv -----------------------


ACTIVITY_LOG_HEADER = "Year,Month,Implementation,School,District,Country,Notes,EnteredBy,EntryDate\n"


def write_log(tmp_path, *rows: str) -> Path:
    path = tmp_path / "activity-log.csv"
    path.write_text(ACTIVITY_LOG_HEADER + "".join(row + "\n" for row in rows))
    return path


def test_checked_in_activity_log_template_is_empty_and_builds_clean(payload):
    """The template ships with a header and no data rows: it should not
    change canonicalSchools/datedTimelineRows from the xlsx-only baseline.
    """
    assert payload["network"]["activityLogRows"] == 0


def test_activity_log_row_is_classified_and_merged(builder, tmp_path):
    path = write_log(
        tmp_path,
        "2026,June,Training - Follow-up visit,Banjika,Karatu,Tanzania,Refresher for 3 teachers,Field Officer,2026-06-15",
    )
    events = builder.load_activity_log_events(path=path)
    assert len(events) == 1
    event = events[0]
    assert event["school"] == "Banjika"
    assert event["activityType"] == "training"
    assert event["source"] == "activity-log"
    assert event["loggedNote"] == "Refresher for 3 teachers"


def test_activity_log_new_school_is_picked_up(builder, tmp_path):
    path = write_log(
        tmp_path,
        "2026,July,SPARC+ Installation,A Brand New School,Karatu,Tanzania,,Field Officer,2026-07-01",
    )
    events = builder.load_activity_log_events(path=path)
    assert events[0]["school"] == "A Brand New School"
    assert events[0]["generation"] == "SPARC+"
    assert events[0]["activityType"] == "deploy"


def test_activity_log_missing_file_returns_no_events(builder, tmp_path):
    assert builder.load_activity_log_events(path=tmp_path / "does-not-exist.csv") == []


def test_activity_log_blank_trailing_row_is_skipped(builder, tmp_path):
    path = write_log(
        tmp_path,
        "2026,June,Training,Banjika,Karatu,Tanzania,,,",
        ",,,,,,,,",
    )
    events = builder.load_activity_log_events(path=path)
    assert len(events) == 1


def test_activity_log_missing_columns_raises(builder, tmp_path):
    path = tmp_path / "activity-log.csv"
    path.write_text("Year,Month,School\n2026,June,Banjika\n")
    with pytest.raises(TaxonomyError, match="missing required column"):
        builder.load_activity_log_events(path=path)


def test_activity_log_missing_school_raises(builder, tmp_path):
    path = write_log(tmp_path, "2026,June,Training,,Karatu,Tanzania,,,")
    with pytest.raises(TaxonomyError, match="no School value"):
        builder.load_activity_log_events(path=path)


def test_activity_log_missing_implementation_raises(builder, tmp_path):
    path = write_log(tmp_path, "2026,June,,Banjika,Karatu,Tanzania,,,")
    with pytest.raises(TaxonomyError, match="no Implementation"):
        builder.load_activity_log_events(path=path)


def test_activity_log_bad_year_raises(builder, tmp_path):
    path = write_log(tmp_path, "not-a-year,June,Training,Banjika,Karatu,Tanzania,,,")
    with pytest.raises(TaxonomyError, match="non-numeric Year"):
        builder.load_activity_log_events(path=path)


def test_activity_log_typo_is_caught_by_duplicate_gate(taxonomy, tmp_path):
    """End-to-end: a logged row with a near-duplicate school name should
    fail the same way a messy timeline row would, not silently create a
    second school.
    """
    import build_program_data as bpd

    log_path = write_log(tmp_path, "2026,June,Training,Banjica,Karatu,Tanzania,Typo test,,")
    original_activity_log = bpd.ACTIVITY_LOG
    bpd.ACTIVITY_LOG = log_path
    try:
        with pytest.raises(TaxonomyError, match="unmerged school-name duplicates"):
            build_payload()
    finally:
        bpd.ACTIVITY_LOG = original_activity_log
