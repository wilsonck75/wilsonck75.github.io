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

## Audience decision (open — needs Charlie/Caitlin sign-off)

The dashboard is currently published at a public URL
(https://wilsonck75.github.io/projects/powering-potential-dashboard/) with
no access control, same as the rest of the portfolio site. That may be
intentional (radical transparency with donors) or may not be what's wanted
for a document that visibly disputes the org's own public 130/42K/50%/58%
figures. This needs an explicit decision, not a default:

- **Public** (current state): anyone with the link can view it, including
  donors, other board members, and search engines (unless blocked via
  `robots.txt`/`noindex`).
- **Board/staff-only**: would require either (a) moving the dashboard to a
  private repo + GitHub Pages with access restricted to an org, (b) putting
  it behind a simple shared passphrase/link-obscurity approach, or (c)
  hosting it somewhere with real auth (e.g. a Google Site restricted to the
  org, or a password-protected Netlify/Vercel deploy).

Until this is decided, treat the current public URL as the default and
avoid adding anything more sensitive than what's described above (e.g. do
not add individual student records, names, or photos to this dashboard).

## Recommendation

No changes are needed to what's currently displayed — everything is already
school/cohort-level aggregate data. The action item is the audience
decision above, which is a board decision, not something to resolve
unilaterally in a code change.
