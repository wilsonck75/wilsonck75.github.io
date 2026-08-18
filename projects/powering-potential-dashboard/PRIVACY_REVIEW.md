# Privacy and data-governance review

This is a review of what the Powering Potential dashboard exposes, done
before treating it as "production" for any audience beyond the people who
already have the underlying M&E files. It is not a legal opinion; if PPI/PEF
has data-protection obligations tied to a specific donor or country, get
that reviewed separately.

## What the dashboard shows

Everything rendered by `program-data.js` and `dashboard-data.js` is
**aggregated at the school or cohort level**. Specifically:

- **School master (`program-data.js`)**: per-school fields are
  institutional, not personal — school name, district, cluster, install
  year(s), generation of equipment, computer count, activity counts, status.
  No student or staff names, no individual records.
- **Exam archive (`dashboard-data.js` -> `legacyOutcomes.examArchive`)**:
  counts of school-named workbook sheets and pass-rate data points per
  school/cluster group. This is summary statistics (e.g. "school sheets",
  "% passed points"), not individual student scores.
- **Graduate survey (`dashboard-data.js` -> `legacyOutcomes.graduateSurvey`)**:
  respondent-level *counts* (e.g. "270 respondents", "hasJob: N") rolled up
  across the whole survey, not individual responses. The dashboard never
  renders a single respondent's row.
- **Graduate tracking (`legacyOutcomes` / "Recent Graduate Tracking")**: a
  per-year count of post-secondary placements, again aggregated.

## What the dashboard does NOT show

- No individual student names, exam scores, or survey responses.
- No staff/teacher personal data.
- No geolocation finer than district/region (no GPS coordinates, no
  household-level data).
- No photos or other media from site surveys.

## Residual risk to check with Caitlin/Charlie before wider distribution

- **Small-N re-identification**: a few of the aggregates are computed over
  small groups (e.g. a single school's exam pass-rate points, or KDP2's 4
  named schools). For counts this small, "aggregate" doesn't always mean
  "anonymous" — if a school has very few students in a given cohort, a
  school-level average can effectively reveal information about a specific
  small group of individuals. This is a judgment call for Caitlin (who has
  visibility into cohort sizes) rather than something this dashboard can
  verify automatically.
- **Underlying source files**: this review only covers what's *displayed*.
  The xlsx/PDF source files referenced in `SOURCES` (e.g. the graduate
  survey workbook, individual site survey PDFs) are not part of this repo,
  but confirm they are not separately, accidentally made public elsewhere
  (e.g. attached to a public Google Drive link).

## Audience decision — RESOLVED: board-only, first pass (2026-08-18)

Charlie confirmed this dashboard is board-only and this iteration is a
first pass. Implemented accordingly:

- **`<meta name="robots" content="noindex, nofollow">`** on the page, plus
  a repo-level `robots.txt` disallowing the path, so well-behaved search
  engines won't index or crawl it.
- **A shared-passphrase gate** (`access-gate.js`): the page content is
  hidden behind a passphrase prompt until the correct shared passphrase is
  entered; the unlock is remembered for that browser tab's session.

**Read this before relying on it as real security.** GitHub Pages serves
this entire repo as public static files — there is no server to enforce
access control. The passphrase gate is a deterrent, not a security
boundary:

- `dashboard-data.js` and `program-data.js` (the actual data) are still
  directly fetchable by URL by anyone who knows or guesses the filename,
  gate or not. This is acceptable *today* because that data is already
  school/cohort-level aggregates per the review above — there's nothing in
  those files more sensitive than what the gate is protecting.
- The passphrase's SHA-256 hash is visible in `access-gate.js` source; a
  weak, guessable passphrase could be brute-forced offline. Use a
  passphrase that isn't in a common wordlist, and rotate it (see
  `CONTRIBUTING.md`) if a non-board member gets it.
- `noindex`/`robots.txt` only stop compliant crawlers; anyone with the
  direct URL can still open it (that's what the passphrase gate is for).

**If the dashboard's content ever gets more sensitive** than aggregate
school/cohort data (e.g. individual student records get added, which
should not happen per the recommendation below), this gate is not
sufficient and the dashboard should move to real auth — e.g. Cloudflare
Access in front of the Pages site (free for small teams, email-based
one-time codes), or a private host with proper login.

## Recommendation

No changes are needed to what's currently displayed — everything is already
school/cohort-level aggregate data, and should stay that way while this
lightweight gate is the only access control (do not add individual student
records, names, or photos to this dashboard under the current setup).
Revisit this page once the dashboard graduates past "first pass."
