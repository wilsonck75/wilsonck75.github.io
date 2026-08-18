# School survey process

This is the process for measuring what actually happens at a school after
Powering Potential shows up — the piece that turns this from a *data
quality review* (auditing messy historical files) into a *true dashboard*
(tracking real, denominator-backed outcomes over time).

Read `DATA_COLLECTION_PROCESS.md` first if you haven't — that covers
**activities** (installs, upgrades, trainings). This document covers
**outcomes** (enrollment, exam results, lab usage, graduate outcomes) at
the schools those activities happen at.

## Why this exists

The dashboard's metric contract has always been explicit that some of
PPI's public numbers aren't backed by a real table yet:

> "Students reached... not 42,000 until a school-year table exists."
> "Alumni percentage... do not use 10× until the comparison group is
> documented."

Those aren't code bugs — they're missing data. This process is how that
data gets collected, going forward, with the exact numerator/denominator
so those figures can eventually be reported with confidence instead of
flagged as Partial/Weak.

## The three surveys

| Survey | When | Cadence | File |
|---|---|---|---|
| **Baseline** | Once, when a school first engages with PPI — ideally *before* any install | Once per school | `data/survey-baseline.csv` |
| **Annual follow-up** | Ongoing, for any school with a baseline | Every school year (or every visit, whichever is more frequent) | `data/survey-annual.csv` |
| **Graduate/exit** | Ongoing, for each graduating cohort | Once per graduating class per school | `data/survey-graduates.csv` |

**A school must already exist in the system (via its first logged activity
in `data/activity-log.csv`, per `DATA_COLLECTION_PROCESS.md`) before it can
be surveyed.** Surveys reference schools; they don't create them. This is
enforced by the build: an unrecognized school name in any survey file fails
the build with a suggestion (e.g. "did you mean Banjika?") rather than
silently creating a near-duplicate school.

## Roles

Same as `DATA_COLLECTION_PROCESS.md`: field staff/whoever runs the visit
logs the raw numbers; Caitlin Kelley owns the shared spreadsheet; Charlie
Wilson exports and rebuilds.

## 1. Baseline survey — one per school, done once

**When**: as close as possible to a school's *first* PPI engagement —
ideally before installation, so it's a genuine "before" snapshot. If a
school already has activity history and never got a baseline, do one now
retroactively (label it clearly as retroactive in Notes) — a late baseline
is still far better than none.

**Columns** (`data/survey-baseline.csv`):

| Column | Required | Notes |
|---|---|---|
| `School` | yes | Must match an existing canonical name exactly (see `canonical-schools.csv`) |
| `SurveyDate` | yes | When the survey was done |
| `SurveyedBy` | no | Who did it |
| `TotalEnrollment` | yes | Whole number |
| `MaleEnrollment` / `FemaleEnrollment` | no | If available |
| `GradesServed` | no | e.g. `Form 1-4` |
| `TeacherCount` | no | |
| `ExistingComputers` | no | Computers already at the school before PPI, if any |
| `ElectricityAccess` | no | e.g. `Grid`, `Solar`, `None` |
| `InternetAccess` | no | e.g. `None`, `Mobile data`, `Fixed` |
| `ExamName` | no | e.g. `CSEE` (Form 4) or `FTNA` (Form 2) |
| `ExamYear` | no | The year of that exam sitting |
| `ExamSat` | no | Number of students who **sat** the exam |
| `ExamPassed` | no | Number who **passed** — this plus `ExamSat` is the real denominator behind a pass-rate percentage |
| `ContactName` / `ContactPhone` | no | Head teacher or main contact |
| `Notes` | no | Anything else worth recording |

**Example row:**
```
Banjika,2024-01-15,Field Officer,450,230,220,Form 1-4,18,0,Solar,None,CSEE,2023,80,42,Head Teacher,+255700000000,Baseline before SPARC+ upgrade
```

## 2. Annual follow-up survey — one per school per school year

**When**: repeat the same measurements at least once per school year for
every school that has a baseline, so each year becomes another comparison
point against the baseline (or against the prior year).

**Columns** (`data/survey-annual.csv`): same enrollment/exam fields as
baseline, plus `SchoolYear` (required) and lab-specific fields:
`LabFunctional` (yes/no), `WorkingComputers`, `BrokenComputers`,
`WeeklyLabHours`, `ClassesUsingLab`, `TeacherRefresherNeeded` (yes/no).

**Example row:**
```
Banjika,2026,2026-06-01,Field Officer,470,240,230,CSEE,2025,85,58,yes,20,0,25,6,no,One year after upgrade
```

With both rows above, the dashboard can show a real, sourced before/after:
**52.5% (42/80) → 68.2% (58/85)** — not a percentage pulled from thin air.

## 3. Graduate/exit survey — one per graduating cohort per school

Formalizes, going forward, the same kind of survey already in the legacy
archive (`dashboard-data.js` -> `legacyOutcomes.graduateSurvey`: 270
respondents, 2009–2023) so it keeps being collected instead of becoming
another one-off archive extract.

**Columns** (`data/survey-graduates.csv`): `School`, `GraduationYear`,
`SurveyDate`, `SurveyedBy`, `RespondentCount` (required — the denominator
for every percentage below), then count columns for however many of the
respondents answered yes: `HasJobCount`, `TechRelatedJobCount`,
`ComputerHelpedJobCount`, `WentHighSchoolCount`, `VocationalTrainingCount`,
`CollegeOrUniversityCount`.

Every count is validated to not exceed `RespondentCount` — a build-time
sanity check against the exact kind of typo that would otherwise produce a
"110% of graduates have jobs" chart.

## Export and rebuild

Same mechanics as the activity log (`DATA_COLLECTION_PROCESS.md`, "Export
and rebuild"): export each survey's spreadsheet to its CSV file, then:

```bash
cd powering-potential-dashboard/
pip install -r requirements.txt
python3 build_program_data.py
```

The build prints a coverage line so you can see rollout progress at a
glance:

```
Survey coverage: 12 baseline, 8 with a follow-up, 6 of 47 schools have a before/after pair
```

Then run the tests and publish per `CONTRIBUTING.md`:

```bash
python3 -m pytest tests/ -v
```

## What the build validates for you

- Every school name must already be canonical (with a "did you mean"
  suggestion if it looks like a typo of an existing school) — surveys never
  silently create a new school.
- `ExamPassed` can never exceed `ExamSat`; no graduate outcome count can
  exceed `RespondentCount`.
- No school can have two baseline rows, or two rows for the same
  `SchoolYear` / `GraduationYear` — duplicates are a data-entry error, not
  a second survey.
- `Year`/count fields must be whole numbers.

If any of these fail, the build stops with a specific row number and file,
so the bad row can be fixed in the spreadsheet and re-exported rather than
silently corrupting the dashboard's numbers.

## What shows up on the dashboard

Once at least one school has both a baseline and a follow-up survey, the
Program view's "School Outcomes" panel shows:
- A coverage summary (how many of the 47 schools have baseline data, a
  follow-up, or a full before/after pair) — this is itself a real KPI for
  tracking the *survey rollout*, visible from day one even before any
  survey data exists.
- A before/after table for every school with a comparison pair: enrollment,
  exam pass rate (with the sat/passed counts shown, not just the
  percentage), and lab status.

See `CONTRIBUTING.md` for where this lives in `index.html`/`program.js` if
the panel itself needs changes.

## Cross-repo note

Like the rest of this pipeline, this currently lives only in
`wilsonck75.github.io` (see `CONTRIBUTING.md`, "Where things live"). If
`D-Cubed-Data-Lab` becomes writable again, these survey files and this
process should move there too.
