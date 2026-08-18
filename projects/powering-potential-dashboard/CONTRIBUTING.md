# Contributing to the Powering Potential dashboard

This page exists to keep the dashboard from going stale or silently drifting
from the source data after the initial build fades from memory. If you're
about to change something, find your task below.

## Ownership

| Area | Owner | Notes |
|---|---|---|
| M&E source files (timeline xlsx, equipment file, surveys) | Caitlin Kelley (ED), with Charlie Wilson (board) | Lives in Google Drive / the org's M&E archive, not in this repo. |
| School taxonomy / Karatu 23 definitions (`data/school-taxonomy.yml`) | AJ Poole (board), confirmed with Albin / PEF | Any change to `original_karatu`, `kdp1_named`, `kdp2_named`, or `known_discrepancies` should be reviewed by AJ before merging. |
| Dashboard code (`index.html`, `*.js`, `*.css`, build scripts) | Charlie Wilson (board) | This repo. |
| Publishing to the live site | Charlie Wilson | See "Publishing" below. |

If you're none of the above and you're reading this because something looks
wrong on the live dashboard, the fastest path is to open an issue or ping
Charlie Wilson rather than editing the generated files directly.

## Where things live

This dashboard is developed in
[`wilsonck75/D-Cubed-Data-Lab`](https://github.com/wilsonck75/D-Cubed-Data-Lab)
(`powering-potential-dashboard/`) and published to this portfolio site at
`projects/powering-potential-dashboard/`. See "Publishing" below for how an
update gets from one repo to the other.

## Making a data change

1. Get the updated source file (usually
   `data/Implementation_Timeline_for_Website_2007-2025.xlsx`) into
   `data/` in the source repo.
2. If the change involves a new school-name spelling, a Karatu 23
   classification change, or a tracked/not-tracked status change, edit
   `data/school-taxonomy.yml` first (not `build_program_data.py`). Every
   entry should have a `reason`, and classification changes (Karatu 23
   membership, not-tracked schools) should have a `confirmed_by`.
3. Regenerate:
   ```bash
   pip install -r requirements.txt
   python3 build_program_data.py
   ```
4. Run the tests. They protect the specific numbers the board has been
   told (e.g. "11 of 23") from silently changing, and will fail loudly if
   the taxonomy update introduces an unmerged spelling-variant duplicate:
   ```bash
   python3 -m pytest tests/ -v
   ```
5. If a test's *expected value* needs to change (e.g. the canonical school
   count legitimately went up because a new school was added), update the
   test alongside the data change in the same commit/PR, with a note on why.
6. Commit `data/school-taxonomy.yml` and the regenerated `program-data.js`
   together.

## Making a UI/code change

1. Edit the relevant file directly:
   - Program-view-only markup: `index.html`
   - Program-view-only styling: `program-extra.css`
   - Program-view-only behavior: `program.js`
   - Workbench-view markup/styling/behavior: `workbench-original.html`
     (the single hand-authored source for that view)
2. If you touched `workbench-original.html`, `program-extra.css`, or
   `program.js`, regenerate the combined bundle:
   ```bash
   python3 assemble_index.py
   ```
   This rewrites `dashboard.css` and `dashboard.js`. `index.html` itself is
   not generated and can be edited directly.
3. Build and eyeball it:
   ```bash
   cd /path/to/wilsonck75.github.io  # or wherever this checkout lives
   bundle exec jekyll serve
   ```
   then open `http://127.0.0.1:4000/projects/powering-potential-dashboard/`
   (adjust the path if testing inside the source repo instead).

## Publishing (source repo -> portfolio site)

The dashboard is developed in `D-Cubed-Data-Lab` and published to
`wilsonck75.github.io`, which is the repo that's actually live at
https://wilsonck75.github.io/projects/powering-potential-dashboard/.

A scheduled sync (`.github/workflows/sync-powering-potential-dashboard.yml`
in this repo) checks `D-Cubed-Data-Lab`'s `main` branch for changes to
`powering-potential-dashboard/` and opens a pull request here automatically
when it finds one, stamping `sync-metadata.js` with the source commit it
copied from. Review and merge that PR to publish. You can also trigger it
manually from the Actions tab ("Sync Powering Potential dashboard" ->
"Run workflow") instead of waiting for the schedule.

If you need to publish immediately and can't wait for the workflow, copy
these files by hand from the source repo's `powering-potential-dashboard/`
to this repo's `projects/powering-potential-dashboard/`, then update
`sync-metadata.js`'s `sourceCommit`/`sourceCommitShort`/`syncedAt` to match:

```
index.html, dashboard.css, dashboard.js, program-extra.css, program.js,
dashboard-data.js, program-data.js, workbench-original.html,
build_program_data.py, assemble_index.py, requirements.txt,
data/Implementation_Timeline_for_Website_2007-2025.xlsx,
data/school-taxonomy.yml, tests/
```

**Do not hand-edit `program-data.js`, `dashboard.css`, or `dashboard.js` in
either repo.** They're generated; hand edits will be silently overwritten
(or, in CI, flagged as drift) the next time someone runs the build scripts.

## Usage analytics (opt-in)

Page-view analytics are disabled by default (see the commented-out
GoatCounter script tag near the bottom of `index.html`). Before turning
this on, decide with Charlie/Caitlin whether the dashboard's audience is
public or board-only (see `PRIVACY_REVIEW.md`) — analytics tools have
their own privacy posture to check regardless. To enable:

1. Create a free account at https://www.goatcounter.com (no cookies, no
   personal data collected, GDPR-friendly).
2. Uncomment the `<script data-goatcounter=...>` line in `index.html` and
   replace `YOUR-CODE` with your site code.
3. Regenerate/publish as above.

## Running CI locally before pushing

```bash
cd projects/powering-potential-dashboard
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 assemble_index.py && git diff --stat dashboard.css dashboard.js  # should be empty
node --check dashboard.js
```

The same checks run automatically in
`.github/workflows/powering-potential-dashboard.yml` on every push/PR that
touches this directory.
