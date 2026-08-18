#!/usr/bin/env python3
"""Assemble index.html from the original workbench plus the Program view."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
original = (ROOT / "workbench-original.html").read_text()
css = original.split("<style>", 1)[1].split("</style>", 1)[0]
script = original.split('<script src="./dashboard-data.js"></script>', 1)[1]
script = script.replace("<script>", "", 1)
# drop closing tags at end
script = script.rsplit("</script>", 1)[0].rsplit("</body>", 1)[0].rsplit("</html>", 1)[0].strip()
# original script starts workbench rendering immediately; wrap it
if "document.getElementById(\"generated-at\")" in script:
    script = script.replace(
        "document.getElementById(\"generated-at\").textContent =",
        "const workbenchStamp = document.getElementById(\"generated-at\");\n      if (workbenchStamp) workbenchStamp.textContent =",
    )

extra_css = """
      .view-toggle {
        display: inline-flex;
        gap: 4px;
        padding: 4px;
        margin-top: 18px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        position: relative;
        z-index: 2;
      }
      .view-toggle button {
        border: 0;
        border-radius: 999px;
        padding: 8px 16px;
        font: inherit;
        font-size: 0.92rem;
        color: rgba(255, 255, 255, 0.84);
        background: transparent;
        cursor: pointer;
      }
      .view-toggle button.active {
        background: #fcf8f1;
        color: var(--teal-deep);
        font-weight: 600;
      }
      .hidden { display: none !important; }
      .alert-list { display: grid; gap: 10px; margin-top: 18px; }
      .alert {
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid rgba(80, 93, 109, 0.12);
        background: #fff8ef;
      }
      .alert.high { background: #f8ece8; }
      .alert.medium { background: #fff6e8; }
      .alert.low { background: #eef4f2; }
      .alert strong { display: block; margin-bottom: 4px; }
      .alert p { margin: 0; color: var(--muted); font-size: 0.92rem; line-height: 1.45; }
      .progress-wrap { margin-top: 12px; }
      .progress-track {
        height: 16px;
        border-radius: 999px;
        background: rgba(13, 106, 99, 0.12);
        overflow: hidden;
      }
      .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--teal), #3ba894);
      }
      .school-toolbar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 12px;
      }
      .school-toolbar input,
      .school-toolbar select {
        font: inherit;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid rgba(80, 93, 109, 0.18);
        background: white;
      }
      .school-table-wrap { overflow-x: auto; }
      table.school-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
      }
      table.school-table th,
      table.school-table td {
        text-align: left;
        padding: 8px 8px;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
      }
      table.school-table th { color: var(--muted); font-weight: 600; font-size: 0.76rem; letter-spacing: 0.04em; text-transform: uppercase; }
      .status-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
      }
      .status-pill.active { background: #d8f2de; color: #1f6b36; }
      .status-pill.in_progress { background: #fff0d5; color: #8c5703; }
      .status-pill.proposed { background: #e4eef9; color: #215c94; }
      .status-pill.inactive { background: #f7dfdf; color: #9d2626; }
      .status-pill.not_started { background: #eeeae4; color: #5b6570; }
      .stacked-chart { display: flex; align-items: end; gap: 5px; min-height: 180px; }
      .stack-col { flex: 1; display: flex; flex-direction: column; align-items: stretch; justify-content: end; }
      .stack-bars { display: flex; flex-direction: column-reverse; min-height: 8px; }
      .stack-seg { width: 100%; }
      .stack-seg.deploy { background: var(--teal); }
      .stack-seg.upgrade { background: var(--blue); }
      .stack-seg.content { background: #7aa3c9; }
      .stack-seg.training { background: var(--gold); }
      .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 0.82rem; }
      .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
      .csv-btn {
        border: 0;
        border-radius: 999px;
        padding: 8px 14px;
        background: var(--teal);
        color: white;
        font: inherit;
        cursor: pointer;
      }
      .hero-health {
        margin-top: 14px;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.78);
      }
"""

body = r"""
    <div class="page">
      <section class="hero">
        <div class="hero-main">
          <p class="eyebrow">Powering Potential</p>
          <h1 id="hero-title">Where The Program Stands</h1>
          <div class="hero-copy" id="hero-copy">
            Canonical schools, Karatu 23 progress, and activity types from AJ’s Key tab.
            Public 130 / 42K / 50% / 58% figures stay in Workbench until they have a denominator.
          </div>
          <div class="view-toggle" role="tablist" aria-label="Dashboard view">
            <button type="button" id="btn-program" class="active">Program</button>
            <button type="button" id="btn-workbench">Workbench</button>
          </div>
          <div class="hero-health" id="hero-health"></div>
        </div>
        <aside class="hero-side" id="hero-side"></aside>
      </section>

      <section id="program-view">
        <section class="overview" id="program-overview"></section>
        <section class="alert-list" id="mismatch-list"></section>

        <section class="section">
          <div class="section-header">
            <div>
              <h2>Karatu 23 Tracker</h2>
              <p id="karatu-definition"></p>
            </div>
            <button class="csv-btn" id="download-csv" type="button">Download school master CSV</button>
          </div>
          <div class="panel">
            <div id="karatu-progress-label" style="font-size:1.4rem;"></div>
            <div class="progress-wrap">
              <div class="progress-track"><div class="progress-fill" id="karatu-progress-fill"></div></div>
            </div>
            <p class="panel-copy" id="karatu-note" style="margin-top:12px;"></p>
            <div class="school-table-wrap">
              <table class="school-table" id="karatu-table"></table>
            </div>
            <p class="panel-copy" id="karatu-base" style="margin-top:14px;"></p>
          </div>
        </section>

        <section class="section">
          <div class="section-header">
            <div>
              <h2>Active Network</h2>
              <p>One row per canonical school after merging name variants. Training labels are activities, not extra schools.</p>
            </div>
          </div>
          <div class="panel">
            <div class="school-toolbar">
              <input id="school-filter" placeholder="Filter by school, district, or cluster" size="36">
              <select id="status-filter">
                <option value="">All statuses</option>
                <option value="active">Active</option>
                <option value="in_progress">In progress</option>
                <option value="proposed">Proposed</option>
                <option value="inactive">Inactive</option>
              </select>
              <select id="cluster-filter"></select>
            </div>
            <div class="school-table-wrap">
              <table class="school-table" id="school-table"></table>
            </div>
          </div>
        </section>

        <section class="section">
          <div class="backbone-grid">
            <div class="panel">
              <h3>Activity By Year</h3>
              <p class="panel-copy">Deploy, upgrade, Shule Direct content, and training as separate activity types. 2024 is empty in the log.</p>
              <div class="stacked-chart" id="activity-chart"></div>
              <div class="legend">
                <span><i style="background:var(--teal)"></i>Deploy</span>
                <span><i style="background:var(--blue)"></i>Upgrade</span>
                <span><i style="background:#7aa3c9"></i>Content</span>
                <span><i style="background:var(--gold)"></i>Training</span>
              </div>
            </div>
            <div class="panel">
              <h3>Solution Generations</h3>
              <p class="panel-copy">From AJ Poole’s meeting with Albin. Later SPARC+ is not the same product as 2019 SPARC+.</p>
              <div class="kv" id="generation-key"></div>
            </div>
          </div>
        </section>

        <section class="section">
          <div class="section-header">
            <div>
              <h2>What The Archive Can Already Measure</h2>
              <p>Caitlin’s school-vs-itself questions. These are coverage counts, not yet a finished impact model.</p>
            </div>
          </div>
          <div class="outcomes-grid">
            <div class="panel">
              <h3>Exam archive</h3>
              <p class="panel-copy">Best path to before/after: original Karatu 6 vs 24 Karatu schools without PPI labs.</p>
              <div class="metrics-grid" id="program-exam-stats"></div>
              <div class="bar-chart" id="program-exam-groups"></div>
            </div>
            <div class="stack">
              <div class="panel">
                <h3>Graduate survey of Form 4 leavers</h3>
                <p class="panel-copy">270 respondents across 13 schools, 2009–2023. Shares below are of respondents, not of all graduates.</p>
                <div class="metrics-grid" id="program-survey-stats"></div>
              </div>
              <div class="panel">
                <h3>Metric contract</h3>
                <div class="kv" id="metric-contract"></div>
              </div>
            </div>
          </div>
        </section>
      </section>

      <section id="workbench-view" class="hidden">
        <section class="overview" id="overview"></section>
        <section class="section">
          <div class="section-header">
            <div>
              <h2>Extraction Quality</h2>
              <p>These ratings describe how dependable the extraction is from the current files. They do not judge whether the underlying program work happened.</p>
            </div>
          </div>
          <div class="quality-grid" id="quality-grid"></div>
        </section>
        <section class="section">
          <div class="section-header">
            <div>
              <h2>Historical Backbone And Training Interpretation</h2>
              <p>The implementation timeline remains the strongest cross-school history. Training is usually an activity attached to another implementation, not a standalone project type.</p>
            </div>
          </div>
          <div class="backbone-grid">
            <div class="stack">
              <div class="panel">
                <h3>Deployment Schools By Year <span class="doc-anchor" data-doc-key="deploymentTimelinePanel"></span></h3>
                <p class="panel-copy">Each bar shows schools with deployment or upgrade activity that year.</p>
                <div class="timeline-chart" id="deployment-timeline-chart"></div>
                <div class="timeline-footer" id="deployment-timeline-footer"></div>
              </div>
              <div class="panel">
                <h3>Training-Only School Or Entity Labels By Year <span class="doc-anchor" data-doc-key="trainingOnlyTimelinePanel"></span></h3>
                <p class="panel-copy">Training labels that do not also appear with a deployment in the same year.</p>
                <div class="timeline-chart" id="training-only-timeline-chart"></div>
                <div class="timeline-footer" id="training-only-timeline-footer"></div>
              </div>
            </div>
            <div class="stack">
              <div class="panel">
                <h3>What “Training” Means Here <span class="doc-anchor" data-doc-key="trainingPanel"></span></h3>
                <p class="panel-copy" id="training-summary"></p>
                <div class="kv" id="training-kv"></div>
                <div class="bar-chart" id="training-categories" style="margin-top: 16px;"></div>
              </div>
              <div class="panel">
                <h3>What Training Most Often Appears Beside <span class="doc-anchor" data-doc-key="trainingPairingsPanel"></span></h3>
                <p class="panel-copy">Non-training labels that show up in the same school-year as training.</p>
                <div class="bar-chart" id="training-pairings"></div>
              </div>
            </div>
          </div>
        </section>
        <section class="section">
          <div class="section-header">
            <div>
              <h2>Older Outcome Sources</h2>
              <p>Exam archive, graduate survey, and KDP tracking as extracted from the M&amp;E files.</p>
            </div>
          </div>
          <div class="outcomes-grid">
            <div class="panel">
              <h3>National Exam Archive <span class="doc-anchor" data-doc-key="examPanel"></span></h3>
              <div class="metrics-grid" id="exam-stats"></div>
              <div class="bar-chart" id="exam-groups"></div>
            </div>
            <div class="stack">
              <div class="panel">
                <h3>Graduate Survey Of Form 4 Leavers <span class="doc-anchor" data-doc-key="surveyPanel"></span></h3>
                <div class="metrics-grid" id="survey-stats"></div>
              </div>
              <div class="panel">
                <h3>Recent Graduate Tracking <span class="doc-anchor" data-doc-key="graduateTrackingPanel"></span></h3>
                <div class="bar-chart" id="graduate-years"></div>
              </div>
            </div>
          </div>
          <div class="support-grid" id="support-grid"></div>
        </section>
        <section class="section">
          <div class="section-header">
            <div>
              <h2>Questions The Archive Can Already Help Answer</h2>
            </div>
          </div>
          <div class="claims-grid" id="learning-grid"></div>
        </section>
        <section class="section">
          <div class="section-header">
            <div>
              <h2>Foundation For Cleanup And Future Collection</h2>
            </div>
          </div>
          <div class="caveat-grid" id="foundation-grid"></div>
        </section>
        <div class="foot" id="generated-at"></div>
      </section>
    </div>
"""

program_js = r"""
      const program = window.PROGRAM_DATA;
      const archive = window.DASHBOARD_DATA;

      function setView(view) {
        const programView = document.getElementById("program-view");
        const workbenchView = document.getElementById("workbench-view");
        const programBtn = document.getElementById("btn-program");
        const workbenchBtn = document.getElementById("btn-workbench");
        const isProgram = view !== "workbench";
        programView.classList.toggle("hidden", !isProgram);
        workbenchView.classList.toggle("hidden", isProgram);
        programBtn.classList.toggle("active", isProgram);
        workbenchBtn.classList.toggle("active", !isProgram);
        document.getElementById("hero-title").textContent = isProgram
          ? "Where The Program Stands"
          : "Cleaning The Archive";
        document.getElementById("hero-copy").textContent = isProgram
          ? "Canonical schools, Karatu 23 progress, and activity types from AJ’s Key tab. Public 130 / 42K / 50% / 58% figures stay in Workbench until they have a denominator."
          : "File-quality view: which sources are strongest, how training was logged, and which tables should become the backbone for future collection.";
        history.replaceState(null, "", isProgram ? "#program" : "#workbench");
      }

      function statusChip(status) {
        const label = String(status || "unknown").replaceAll("_", " ");
        return `<span class="status-pill ${status || ""}">${label}</span>`;
      }

      function renderProgramOverview() {
        const net = program.network;
        const k23 = program.karatu23;
        const items = [
          { kicker: "Karatu 23", value: k23.progressLabel, note: "Equipment definition: KDP 1 + KDP 2. Original 6 are separate." },
          { kicker: "Active full labs", value: number.format(net.activeFullLabs), note: "Canonical schools with a 5- or 20-computer lab, not inactive." },
          { kicker: "Pi-oneer only", value: number.format(net.pioneerOnly), note: "Mostly the 16 Zanzibar classroom kits, one device each." },
          { kicker: "Canonical schools", value: number.format(net.canonicalSchools), note: `${number.format(net.datedTimelineRows)} dated timeline rows after name cleanup.` },
        ];
        document.getElementById("program-overview").innerHTML = items.map((item) => `
          <div class="overview-card">
            <div class="overview-kicker">${item.kicker}</div>
            <div class="overview-value">${item.value}</div>
            <div class="overview-note">${item.note}</div>
          </div>
        `).join("");
      }

      function renderHeroSide() {
        const net = program.network;
        document.getElementById("hero-side").innerHTML = `
          <h2>How to read this</h2>
          <p>Program is for board and staff decisions. Workbench is the archive audit.</p>
          <ul class="hero-list">
            <li>Training never adds a school.</li>
            <li>Karatu 23 excludes the original six labs.</li>
            <li>Hover Workbench info dots for file provenance.</li>
          </ul>
        `;
        document.getElementById("hero-health").textContent =
          `School master generated ${new Date(program.generatedAt).toLocaleString()} · archive extract ${new Date(archive.generatedAt).toLocaleString()} · ${net.inactive} inactive sites · 2024 not logged`;
      }

      function renderMismatches() {
        document.getElementById("mismatch-list").innerHTML = program.mismatches.map((item) => `
          <div class="alert ${item.severity}">
            <strong>${item.title}</strong>
            <p>${item.detail}</p>
          </div>
        `).join("");
      }

      function renderKaratu() {
        const k = program.karatu23;
        document.getElementById("karatu-definition").textContent = k.definition;
        document.getElementById("karatu-progress-label").textContent = k.progressLabel;
        document.getElementById("karatu-progress-fill").style.width = `${(k.progressForBar / k.denominator) * 100}%`;
        document.getElementById("karatu-note").textContent = k.progressNote;
        document.getElementById("karatu-base").textContent =
          `Installed base, not in the 23: ${k.originalBase.join(", ")}.`;
        const rows = [
          `<tr><th>School</th><th>Status</th><th>Role</th><th>Year</th><th>Generation</th><th>PCs</th><th>Notes</th></tr>`,
          ...k.rows.map((row) => `<tr>
            <td>${row.name}</td>
            <td>${statusChip(row.status)}</td>
            <td>${row.role || ""}</td>
            <td>${row.year || "—"}</td>
            <td>${row.generation || "—"}</td>
            <td>${row.computers ?? "—"}</td>
            <td>${row.notes || ""}</td>
          </tr>`),
        ];
        document.getElementById("karatu-table").innerHTML = rows.join("");
      }

      function schoolRows(list) {
        return list.map((row) => `<tr>
          <td>${row.canonicalName}</td>
          <td>${row.cluster}</td>
          <td>${row.district || "—"}</td>
          <td>${statusChip(row.status)}</td>
          <td>${row.siteType.replaceAll("_", " ")}</td>
          <td>${row.currentGeneration || "—"}</td>
          <td>${row.firstYear || "—"}–${row.latestYear || "—"}</td>
          <td>${row.computers ?? "—"}</td>
        </tr>`).join("");
      }

      function filteredSchools() {
        const q = document.getElementById("school-filter").value.toLowerCase();
        const status = document.getElementById("status-filter").value;
        const cluster = document.getElementById("cluster-filter").value;
        return program.schools.filter((row) => {
          const hay = `${row.canonicalName} ${row.district} ${row.cluster} ${(row.nameVariants || []).join(" ")}`.toLowerCase();
          if (q && !hay.includes(q)) return false;
          if (status && row.status !== status) return false;
          if (cluster && row.cluster !== cluster) return false;
          return true;
        });
      }

      function renderSchoolTable() {
        const list = filteredSchools();
        document.getElementById("school-table").innerHTML = `
          <tr><th>School</th><th>Cluster</th><th>District</th><th>Status</th><th>Site</th><th>Generation</th><th>Years</th><th>PCs</th></tr>
          ${schoolRows(list)}
        `;
      }

      function renderFilters() {
        const clusters = [...new Set(program.schools.map((s) => s.cluster))].sort();
        document.getElementById("cluster-filter").innerHTML =
          `<option value="">All clusters</option>` + clusters.map((c) => `<option>${c}</option>`).join("");
        document.getElementById("school-filter").addEventListener("input", renderSchoolTable);
        document.getElementById("status-filter").addEventListener("change", renderSchoolTable);
        document.getElementById("cluster-filter").addEventListener("change", renderSchoolTable);
      }

      function renderActivity() {
        const items = program.activityByYear;
        const max = Math.max(...items.map((i) => i.deploy + i.upgrade + i.content + i.training), 1);
        document.getElementById("activity-chart").innerHTML = items.map((item) => {
          const total = item.deploy + item.upgrade + item.content + item.training;
          const h = (key) => Math.max((item[key] / max) * 150, item[key] ? 3 : 0);
          return `<div class="stack-col" title="${item.year}: ${total} activities">
            <div class="timeline-value">${total}</div>
            <div class="stack-bars">
              <div class="stack-seg deploy" style="height:${h("deploy")}px"></div>
              <div class="stack-seg upgrade" style="height:${h("upgrade")}px"></div>
              <div class="stack-seg content" style="height:${h("content")}px"></div>
              <div class="stack-seg training" style="height:${h("training")}px"></div>
            </div>
            <div class="timeline-year">${item.year}</div>
          </div>`;
        }).join("");
      }

      function renderKeyPanels() {
        document.getElementById("generation-key").innerHTML = program.generationKey.map((item) => `
          <div class="kv-row"><span>${item.name}</span><span>${item.detail}</span></div>
        `).join("");
        document.getElementById("metric-contract").innerHTML = program.metricContract.map((item) => `
          <div class="kv-row"><span>${item.metric}</span><span>${item.definition}</span></div>
        `).join("");
      }

      function renderProgramOutcomes() {
        const exam = archive.legacyOutcomes.examArchive;
        const survey = archive.legacyOutcomes.graduateSurvey;
        document.getElementById("program-exam-stats").innerHTML = [
          ["School sheets", exam.schoolSheets],
          ["Pass-rate points", exam.pctPassPoints],
          ["Years", `${exam.yearStart}–${exam.yearEnd}`],
          ["Karatu PPI sheets", exam.groups.find((g) => g.group === "Karatu PPI").schoolSheets],
        ].map(([k, v]) => `<div class="metric-tile"><div class="metric-kicker">${k}</div><div class="metric-value">${v}</div></div>`).join("");
        const maxSheets = Math.max(...exam.groups.map((g) => g.schoolSheets));
        document.getElementById("program-exam-groups").innerHTML = exam.groups.map((g) => `
          <div class="bar-row">
            <div class="bar-label">${g.group}</div>
            <div class="bar-track"><div class="bar-fill blue" style="width:${(g.schoolSheets / maxSheets) * 100}%"></div></div>
            <div class="bar-value">${g.schoolSheets}</div>
          </div>
        `).join("");
        const pct = (n) => `${Math.round((n / survey.respondents) * 100)}%`;
        document.getElementById("program-survey-stats").innerHTML = [
          ["Respondents", survey.respondents],
          ["Computer helped job", `${survey.computerHelpedJob} (${pct(survey.computerHelpedJob)})`],
          ["Has a job", `${survey.hasJob} (${pct(survey.hasJob)})`],
          ["Form 5 / high school", `${survey.wentHighSchool} (${pct(survey.wentHighSchool)})`],
          ["Vocational", `${survey.vocationalTraining} (${pct(survey.vocationalTraining)})`],
          ["College / university", `${survey.collegeOrUniversity} (${pct(survey.collegeOrUniversity)})`],
        ].map(([k, v]) => `<div class="metric-tile"><div class="metric-kicker">${k}</div><div class="metric-value">${v}</div></div>`).join("");
      }

      function downloadCsv() {
        const cols = ["schoolId", "canonicalName", "cluster", "district", "country", "status", "siteType", "currentGeneration", "firstYear", "latestYear", "computers", "inKaratu23", "karatu23Role"];
        const lines = [cols.join(",")];
        program.schools.forEach((row) => {
          lines.push(cols.map((c) => JSON.stringify(row[c] ?? "")).join(","));
        });
        const blob = new Blob([lines.join("\n")], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "ppi-school-master.csv";
        a.click();
        URL.revokeObjectURL(url);
      }

      document.getElementById("btn-program").addEventListener("click", () => setView("program"));
      document.getElementById("btn-workbench").addEventListener("click", () => setView("workbench"));
      document.getElementById("download-csv").addEventListener("click", downloadCsv);
      renderHeroSide();
      renderProgramOverview();
      renderMismatches();
      renderKaratu();
      renderFilters();
      renderSchoolTable();
      renderActivity();
      renderKeyPanels();
      renderProgramOutcomes();
      setView(location.hash === "#workbench" ? "workbench" : "program");
"""

html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Powering Potential Program Dashboard</title>
    <style>
{css}
{extra_css}
    </style>
  </head>
  <body>
{body}
    <script src="./dashboard-data.js"></script>
    <script src="./program-data.js"></script>
    <script>
{script}
{program_js}
    </script>
  </body>
</html>
"""
(ROOT / "index.html").write_text(html)
print("Wrote index.html", len(html))


if __name__ == "__main__":
    pass
