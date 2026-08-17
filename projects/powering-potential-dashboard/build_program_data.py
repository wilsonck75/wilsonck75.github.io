#!/usr/bin/env python3
"""Build program-data.js from the implementation timeline workbook."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent
TIMELINE = ROOT / "data" / "Implementation_Timeline_for_Website_2007-2025.xlsx"
OUT = ROOT / "program-data.js"

NAME_ALIASES = {
    "banjika": "Banjika",
    "banjka": "Banjika",
    "mlimani sumawe": "Mlimani Sumawe",
    "mlimani sumwae": "Mlimani Sumawe",
    "mlimane sumawe": "Mlimani Sumawe",
    "rigicha": "Rigicha",
    "rigichia": "Rigicha",
    "soitsambu": "Soit Sambu",
    "soit sambu": "Soit Sambu",
    "slahamo": "Slahamo",
    "slahhamo": "Slahamo",
    "endallah": "Endallah",
    "endalah": "Endallah",
    "olturoto": "Olturoto",
    "olturotu": "Olturoto",
    "olturoto primary school": "Olturoto",
    "nainokanoka": "Nainokanoka",
    "lake natron": "Lake Natron",
    "san francisco rio itaya (peru)": "San Francisco Rio Itaya",
    "san francisco rio itaya": "San Francisco Rio Itaya",
}

META_LABELS = {
    "16 schools",
    "zanzibar – 16 schools",
    "zanzibar-16 schools",
    "zanzibar-training",
    "karatu district project year 1 schools",
    "banjika (students from banjika and other district schools)",
}

ZANZIBAR_SCHOOLS = [
    ("Shungi", "South Pemba", "Chake-Chake"),
    ("Jongowe", "North Unguja", "North A Unguja"),
    ("Kandwi", "North Unguja", "North A Unguja"),
    ("Kidoti", "North Unguja", "North A Unguja"),
    ("Kijini", "North Unguja", "North A Unguja"),
    ("Mbuyutende", "North Unguja", "North A Unguja"),
    ("Pwani Mchangani", "North Unguja", "North A Unguja"),
    ("Tumbatu", "North Unguja", "North A Unguja"),
    ("Charawe", "South Unguja", "Central Unguja"),
    ("Ukongoroni", "South Unguja", "Central Unguja"),
    ("Uzi", "South Unguja", "Central Unguja"),
    ("Michamvi", "South Unguja", "South Unguja"),
    ("Tumbe", "North Pemba", "Micheweni"),
    ("Kisiwa Panza", "South Pemba", "Mkoani"),
    ("Makoongwe", "South Pemba", "Mkoani"),
    ("Fundo", "North Pemba", "Wete"),
]

ORIGINAL_KARATU = ["Banjika", "Welwel", "Florian", "Slahamo", "Endallah", "Baray"]
KDP1_NAMED = ["Mlimani Sumawe", "Domel", "Endabash", "Chaenda", "Oldeani", "Upper Kitete"]
KDP2_NAMED = ["Diego", "Kilimatembo", "Kilimamoja", "Marang"]
NOT_TRACKED = {
    "Noonkodin": "Solar system damaged; no longer tracked (2023 proposal).",
    "Olturoto": "Primary-school training only; no lab.",
    "Shimbwe": "Lab was not maintained; no longer tracked.",
    "Mgutwa": "Pi-oneer one-off for a donor; no longer tracked.",
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def norm_name(raw: str | None) -> str | None:
    if not raw:
        return None
    text = " ".join(str(raw).split())
    key = text.lower()
    if key in META_LABELS or "for all 12 karatu" in key:
        return None
    if key == "7 districts":
        return "Charawe"
    return NAME_ALIASES.get(key, text)


def classify_activity(label: str) -> str:
    s = label.lower()
    if "shule direct" in s and "train" in s:
        return "training"
    if "shule direct" in s:
        return "content"
    if "upgrade" in s or "replace missing" in s:
        return "upgrade"
    if "train" in s or "workshop" in s:
        return "training"
    if any(token in s for token in ("phase", "sparc", "pi-oneer", "pioneer", "solar", "laptop", "computer", "rachel", "network")):
        return "deploy"
    return "other"


def classify_generation(label: str) -> str | None:
    s = label.lower()
    if "sparc+" in s or "sparc +" in s:
        return "SPARC+"
    if re.search(r"\bsparc\b", s):
        return "SPARC"
    if "pi-oneer" in s or "pioneer" in s:
        return "Pi-oneer"
    if "phase 2" in s:
        return "Phase 2"
    if "phase 1" in s:
        return "Phase 1"
    if "laptop" in s:
        return "Pilot laptop"
    return None


def cluster_for(school: str, district: str, country: str) -> str:
    if school in ORIGINAL_KARATU:
        return "Original Karatu"
    if school in KDP1_NAMED or school == "Gyekrum Arusha":
        return "KDP 1"
    if school in KDP2_NAMED:
        return "KDP 2"
    if school == "Kainam":
        return "Karatu in progress"
    if country == "Peru":
        return "Peru"
    if district and "zanzibar" in district.lower():
        return "Zanzibar"
    mapping = {
        "Bunda": "Bunda",
        "Ngorongoro": "Ngorongoro",
        "Serengeti": "Serengeti",
        "Monduli": "Monduli",
        "Moshi": "Kilimanjaro",
        "Morogoro Urban": "Morogoro",
        "Simanjiro": "Manyara",
        "Arusha": "Arusha other",
    }
    return mapping.get(district, district or "Other")


def site_type(school: str, generations: list[str], activities: list[str]) -> str:
    if school in NOT_TRACKED and school == "Olturoto":
        return "training_only"
    if school in {"Mgutwa"} or (generations and set(generations) <= {"Pi-oneer"}):
        return "pioneer_only"
    if any(g in {"Phase 1", "Phase 2", "SPARC", "SPARC+", "Pilot laptop"} for g in generations):
        return "full_lab"
    if activities and set(activities) <= {"training"}:
        return "training_only"
    return "unknown"


def status_for(school: str, site: str) -> str:
    if school == "Kainam":
        return "in_progress"
    if school == "Gyekrum Arusha":
        return "proposed"
    if school in NOT_TRACKED:
        return "inactive"
    if site in {"full_lab", "pioneer_only"}:
        return "active"
    if site == "training_only":
        return "inactive"
    return "unknown"


def main() -> None:
    wb = load_workbook(TIMELINE, data_only=True)
    by_date = wb["By Date"]
    zanzibar_sheet = wb["Zanzibar Schools"]

    events = []
    for year, month, impl, school, district, country in by_date.iter_rows(min_row=5, max_col=6, values_only=True):
        if not impl and not school:
            continue
        label = " ".join(str(impl).split()) if impl else ""
        canonical = norm_name(school)
        events.append(
            {
                "year": int(year) if year else None,
                "month": str(month).replace("\n", " ").strip() if month else "",
                "label": label,
                "rawSchool": " ".join(str(school).split()) if school else "",
                "school": canonical,
                "district": " ".join(str(district).split()) if district else "",
                "country": str(country).strip() if country else "Tanzania",
                "activityType": classify_activity(label),
                "generation": classify_generation(label),
            }
        )

    # Expand the Zanzibar bundle into 16 named Pi-oneer schools.
    zanzibar_names = []
    for name, region, district in zanzibar_sheet.iter_rows(min_row=2, max_col=3, values_only=True):
        if not name:
            continue
        canonical = "Charawe" if str(name).strip() == "7 Districts" else " ".join(str(name).split())
        zanzibar_names.append((canonical, district, region))
    if len(zanzibar_names) < 16:
        zanzibar_names = [(n, d, r) for n, r, d in ZANZIBAR_SCHOOLS]

    schools: dict[str, dict] = {}

    def ensure(name: str, **kwargs) -> dict:
        row = schools.setdefault(
            name,
            {
                "schoolId": slug(name),
                "canonicalName": name,
                "nameVariants": [],
                "district": kwargs.get("district", ""),
                "region": kwargs.get("region", ""),
                "country": kwargs.get("country", "Tanzania"),
                "cluster": kwargs.get("cluster", ""),
                "firstYear": None,
                "latestYear": None,
                "currentGeneration": None,
                "generations": [],
                "activityCounts": Counter(),
                "inKaratu23": False,
                "karatu23Role": None,
                "notes": [],
            },
        )
        for key in ("district", "region", "country", "cluster"):
            if kwargs.get(key) and not row.get(key):
                row[key] = kwargs[key]
        return row

    for name, district, region in zanzibar_names:
        ensure(
            name,
            district=str(district or ""),
            region=str(region or "Zanzibar"),
            cluster="Zanzibar",
        )

    for event in events:
        name = event["school"]
        if not name:
            continue
        row = ensure(name, district=event["district"], country=event["country"])
        raw = event["rawSchool"]
        if raw and raw not in row["nameVariants"] and raw != name:
            row["nameVariants"].append(raw)
        if event["year"]:
            row["firstYear"] = event["year"] if row["firstYear"] is None else min(row["firstYear"], event["year"])
            row["latestYear"] = event["year"] if row["latestYear"] is None else max(row["latestYear"], event["year"])
        row["activityCounts"][event["activityType"]] += 1
        if event["generation"] and event["generation"] not in row["generations"]:
            row["generations"].append(event["generation"])
            row["currentGeneration"] = event["generation"]

    # Schools known from other archive files but missing from the timeline.
    ensure("Gyekrum Arusha", district="Karatu", country="Tanzania", cluster="KDP 1")["notes"].append(
        "Named in the 2023 Addax & Oryx proposal with Upper Kitete; not on the By Date sheet."
    )
    kainam = ensure("Kainam", district="Karatu", country="Tanzania", cluster="Karatu in progress")
    kainam["notes"].append("From PEF site survey, 18 Feb 2026: solar installed March 2026, training expected May 2026, ~500 students.")
    kainam["currentGeneration"] = "SPARC+"
    kainam["latestYear"] = 2026

    for name, row in schools.items():
        counts = dict(row["activityCounts"])
        activities = [k for k, v in counts.items() if v]
        site = site_type(name, row["generations"], activities)
        row["siteType"] = site
        row["status"] = status_for(name, site)
        row["cluster"] = row["cluster"] or cluster_for(name, row["district"], row["country"])
        row["activityCounts"] = counts
        if name in NOT_TRACKED:
            row["notes"].append(NOT_TRACKED[name])
            row["status"] = "inactive"
        if row["cluster"] == "Zanzibar":
            row["siteType"] = "pioneer_only"
            row["status"] = "active"
            row["currentGeneration"] = row["currentGeneration"] or "Pi-oneer"
            if "Pi-oneer" not in row["generations"]:
                row["generations"] = ["Pi-oneer"] + row["generations"]
            row["firstYear"] = row["firstYear"] or 2016
            row["district"] = row["district"] or "Zanzibar"
        if name in KDP1_NAMED:
            row["inKaratu23"] = True
            row["karatu23Role"] = "KDP 1 installed"
            row["cluster"] = "KDP 1"
        if name in KDP2_NAMED:
            row["inKaratu23"] = True
            row["karatu23Role"] = "KDP 2 installed"
            row["cluster"] = "KDP 2"
        if name == "Gyekrum Arusha":
            row["inKaratu23"] = True
            row["karatu23Role"] = "Proposed; not in timeline"
            row["status"] = "proposed"
        if name == "Kainam":
            row["inKaratu23"] = True
            row["karatu23Role"] = "In progress (site survey)"
            row["status"] = "in_progress"
            row["siteType"] = "full_lab"
        if name in ORIGINAL_KARATU:
            row["cluster"] = "Original Karatu"
            row["inKaratu23"] = False
            row["karatu23Role"] = "Installed base; not in the 23"

    # Computer counts from the By Region tab where names match.
    region_computers = {
        "Banjika": 20,
        "Welwel": 20,
        "Florian": 20,
        "Slahamo": 5,
        "Endallah": 20,
        "Baray": 20,
        "Mlimani Sumawe": 20,
        "Domel": 20,
        "Endabash": 20,
        "Chaenda": 20,
        "Oldeani": 20,
        "Soit Sambu": 20,
        "Nainokanoka": 20,
        "Lake Natron": 5,
        "Noonkodin": 5,
        "Mgutwa": 1,
        "Rigicha": 5,
        "Kabasa": 20,
        "Mekomariro": 20,
        "Sazira": 20,
        "Shimbwe": 10,
        "Nanenane": 5,
        "San Francisco Rio Itaya": 26,
        "Upper Kitete": 20,
        "Diego": 20,
        "Kilimatembo": 20,
        "Kilimamoja": 20,
        "Marang": 20,
        "Kainam": None,
        "Gyekrum Arusha": 20,
    }
    for name, row in schools.items():
        if row["cluster"] == "Zanzibar":
            row["computers"] = 1
        else:
            row["computers"] = region_computers.get(name)

    school_list = sorted(schools.values(), key=lambda r: (r["cluster"], r["canonicalName"]))

    named_kdp = [s for s in school_list if s["karatu23Role"] in {"KDP 1 installed", "KDP 2 installed"}]
    karatu23 = {
        "denominator": 23,
        "definition": "Karatu public secondary schools that did not already have a lab when the 2023 district plan was written (32 public schools minus 9 with existing labs). The original 6 PPI Karatu schools are the installed base and are not in this 23.",
        "installedNamed": len(named_kdp),
        "kdp1Named": len(KDP1_NAMED),
        "kdp1EquipmentCount": 7,
        "kdp2Named": len(KDP2_NAMED),
        "inProgress": 1,
        "proposedUnconfirmed": 1,
        "remainingUnnamed": 23 - 7 - 4,
        "progressForBar": 11,
        "progressLabel": "11 of 23",
        "progressNote": "Uses the equipment file definition: KDP 1 (7) + KDP 2 (4). Timeline names 6 of the 7 KDP 1 schools. Gyekrum Arusha is the likely seventh but is not logged as installed. Kainam is in progress and may occupy one of the remaining 12 slots.",
        "originalBase": ORIGINAL_KARATU,
        "rows": [],
    }
    for name in KDP1_NAMED + ["Gyekrum Arusha"] + KDP2_NAMED + ["Kainam"]:
        row = schools[name]
        karatu23["rows"].append(
            {
                "schoolId": row["schoolId"],
                "name": name,
                "status": row["status"],
                "role": row["karatu23Role"],
                "year": row["latestYear"],
                "generation": row["currentGeneration"],
                "computers": row.get("computers"),
                "notes": " ".join(row["notes"]),
            }
        )
    karatu23["rows"].append(
        {
            "schoolId": "remaining-unnamed",
            "name": "Remaining Karatu schools not yet in the archive",
            "status": "not_started",
            "role": f"{karatu23['remainingUnnamed']} of 23 still unnamed in these files",
            "year": None,
            "generation": None,
            "computers": None,
            "notes": "Keep as a counted remainder until PEF supplies the official 23-school list.",
        }
    )

    activity_by_year = defaultdict(lambda: Counter())
    for event in events:
        if event["year"]:
            activity_by_year[event["year"]][event["activityType"]] += 1
    years = sorted(activity_by_year)
    activity_series = [
        {
            "year": year,
            "deploy": activity_by_year[year]["deploy"],
            "upgrade": activity_by_year[year]["upgrade"],
            "content": activity_by_year[year]["content"],
            "training": activity_by_year[year]["training"],
            "other": activity_by_year[year]["other"],
        }
        for year in years
    ]

    network = {
        "canonicalSchools": len(school_list),
        "activeFullLabs": sum(1 for s in school_list if s["siteType"] == "full_lab" and s["status"] == "active"),
        "pioneerOnly": sum(1 for s in school_list if s["siteType"] == "pioneer_only" and s["status"] != "inactive"),
        "inactive": sum(1 for s in school_list if s["status"] == "inactive"),
        "inProgress": sum(1 for s in school_list if s["status"] == "in_progress"),
        "proposed": sum(1 for s in school_list if s["status"] == "proposed"),
        "datedTimelineRows": len(events),
        "trainingRows": sum(1 for e in events if e["activityType"] == "training"),
        "deployRows": sum(1 for e in events if e["activityType"] == "deploy"),
        "knownComputers": sum(s["computers"] or 0 for s in school_list),
    }

    mismatches = [
        {
            "severity": "high",
            "title": "Two different meanings of “11 of 23”",
            "detail": "The public Karatu tile and the equipment file mean KDP 1 (7) + KDP 2 (4). The By Region tab also totals 11 Karatu schools, but those are the original 6 plus five 2023 schools, omitting Upper Kitete and the 2025 four.",
        },
        {
            "severity": "high",
            "title": "KDP 1 is 7 in equipment, 6 on the timeline",
            "detail": "Named 2023 SPARC+ schools: Mlimani Sumawe, Domel, Endabash, Chaenda, Oldeani, Upper Kitete. Gyekrum Arusha is in the AOF proposal with Upper Kitete but has no By Date install row.",
        },
        {
            "severity": "medium",
            "title": "2024 has no timeline rows",
            "detail": "Treat 2024 as not logged, not as a year with zero impact.",
        },
        {
            "severity": "medium",
            "title": "2025 four schools have no training rows yet",
            "detail": "Diego, Kilimatembo, Kilimamoja, and Marang are logged as SPARC+ Installation only.",
        },
        {
            "severity": "medium",
            "title": "Kainam is not on the timeline",
            "detail": "The March 2026 site survey shows a Karatu lab in progress that the implementation workbook does not list.",
        },
        {
            "severity": "low",
            "title": "Public 130 / 42K / 50% / 58% are not these tables",
            "detail": "113 dated activity rows are not unique projects. The graduate survey is 270 respondents, not an organization-wide rate.",
        },
    ]

    metric_contract = [
        {
            "metric": "School served",
            "definition": "Distinct canonical school with a full lab or Pi-oneer. Training-only and group labels do not count.",
        },
        {
            "metric": "Active lab",
            "definition": "Full lab whose latest known status is not inactive. Solar/computer function still needs a current site survey.",
        },
        {
            "metric": "Karatu progress",
            "definition": "KDP installs ÷ 23. Show the original 6 separately as installed base.",
        },
        {
            "metric": "Students reached",
            "definition": "Sum of latest known enrollment at active schools, labeled as latest headcount — not 42,000 until a school-year table exists.",
        },
        {
            "metric": "Training",
            "definition": "An activity on an existing site (install week, follow-up, or requested topic). Never +1 school. AJ Poole / Albin Key tab, 2026.",
        },
        {
            "metric": "Alumni percentage",
            "definition": "Named survey + year range + respondent n + exact question. Do not use 10× until the comparison group is documented.",
        },
    ]

    payload = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "timeline": "Implementation Timeline for Website_2007-2025.xlsx",
            "keyTab": "AJ Poole / Albin meeting notes in the Key tab and PEF Solution Generations Specification",
            "equipment": "PROGRAM EQUIPMENTS_2008-2023.xlsx summaries already extracted into dashboard-data.js",
            "siteSurveys": "PEF School Site Surveys.xlsx (Banjika, Kainam)",
            "proposal": "Powering Potential Proposal UPDATED 2023-06-27.pdf",
        },
        "network": network,
        "karatu23": karatu23,
        "schools": school_list,
        "activityByYear": activity_series,
        "mismatches": mismatches,
        "metricContract": metric_contract,
        "generationKey": [
            {"name": "Phase 1", "detail": "First-gen lab: 5 desktops, open-source software, RACHEL, basic solar."},
            {"name": "Phase 2", "detail": "Same hardware generation, expanded to about 20 computers and a larger solar system."},
            {"name": "Pi-oneer", "detail": "Classroom kit: 1 Raspberry Pi, battery projector, solar charger. Not a full lab."},
            {"name": "SPARC", "detail": "Phase 1 using Raspberry Pi: 5 computers, 3 servers, solar, RACHEL / Office / Shule Direct / Scratch."},
            {"name": "SPARC+", "detail": "Phase 2 using Raspberry Pi: 20 computers and expanded solar. Later years add a projector, then Windows mini PCs."},
            {"name": "Windows mini (unnamed)", "detail": "Newest SPARC+ variant with mini computers and Microsoft Windows/Office. Albin still needs to name this generation."},
        ],
    }

    OUT.write_text("window.PROGRAM_DATA = " + json.dumps(payload, indent=2) + ";\n")
    print(f"Wrote {OUT} ({len(school_list)} schools, {len(events)} events)")


if __name__ == "__main__":
    main()
