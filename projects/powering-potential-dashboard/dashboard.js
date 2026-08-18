try {
const data = window.DASHBOARD_DATA;
      const number = new Intl.NumberFormat("en-US");
      const SOURCES = {
        timeline: "Monitoring and Evaluation/Implementation Timeline for Website_2007-2023.xlsx (By Date sheet)",
        equipment: "Monitoring and Evaluation/PROGRAM EQUIPMENTS_2008-2023.xlsx",
        examArchive: "Monitoring and Evaluation/M&E_Historical Files From Janice/National Exam Results/",
        graduateSurvey: "Monitoring and Evaluation/M&E_Historical Files From Janice/Surveys-Questionnaires/Copy of Survey of Form 4 graduates.xlsx (Sheet1)",
        graduateTracking: "Monitoring and Evaluation/Completed Forms and Spreadsheets/KDP 1 SCHOOL STATUS OF GRADUATES.xlsx",
        annual2018: "Monitoring and Evaluation/M&E_Historical Files From Janice/Annual School Data/2018 Annual School Forms/School Data 2018.ods.xlsx (Student Data sheet)",
        enrollmentHistory: "Monitoring and Evaluation/M&E_Historical Files From Janice/Annual School Data/Student School #s_2006-2016.xlsx (Karatu Schools sheet)",
        quality: "Quality labels are derived from how directly and consistently the current files could be extracted.",
      };

      const DOCS = {
        deploymentTimelinePanel: {
          definition: "This chart shows the number of schools with deployment or upgrade activity in each year. It is not a raw row count, because a single year can contain multiple activity rows for the same school.",
          source: SOURCES.timeline,
        },
        trainingOnlyTimelinePanel: {
          definition: "This chart shows only training school or entity labels that do not also appear with a deployment or upgrade label in the same year. It helps separate standalone training activity from training that accompanied a rollout.",
          source: SOURCES.timeline,
        },
        trainingPanel: {
          definition: "Training is classified as a timeline activity label. In this dashboard, it is separated from deployment and upgrade labels to clarify whether it behaves like a project type or a companion activity.",
          source: SOURCES.timeline,
        },
        trainingPairingsPanel: {
          definition: "Counts of school-year combinations where at least one training row appears alongside at least one non-training implementation label.",
          source: SOURCES.timeline,
        },
        examPanel: {
          definition: "Legacy national exam workbooks grouped by cluster. School sheets are school-named tabs, and pass-rate points count usable numeric values in the percent-passed series.",
          source: SOURCES.examArchive,
        },
        surveyPanel: {
          definition: "Raw respondent counts from the older graduate survey workbook. These are yes-count tallies after filtering to valid respondent rows, not normalized percentages.",
          source: SOURCES.graduateSurvey,
        },
        graduateTrackingPanel: {
          definition: "Post-secondary placements are shown as advance school plus college or university, which is the most comparable measure across 2021-2023 given the 2023 schema change.",
          source: SOURCES.graduateTracking,
        },
      };

      function escapeHtml(value) {
        return String(value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }

      function tooltipMarkup(definition, source) {
        return `
          <span class="tooltip-wrap" tabindex="0">
            <button class="info-dot" type="button" aria-label="Show definition and source">i</button>
            <span class="tooltip-text">
              <strong>Definition</strong>
              ${escapeHtml(definition)}
              <span class="tooltip-source"><strong>Source</strong>${escapeHtml(source)}</span>
            </span>
          </span>
        `;
      }

      function injectDocAnchors() {
        document.querySelectorAll(".doc-anchor").forEach((node) => {
          const doc = DOCS[node.dataset.docKey];
          if (!doc) return;
          node.innerHTML = tooltipMarkup(doc.definition, doc.source);
        });
      }

      function renderBarChart(targetId, items, valueKey, labelKey, className = "", tooltipBuilder = null) {
        const target = document.getElementById(targetId);
        target.innerHTML = "";

        const max = Math.max(...items.map((item) => Number(item[valueKey]) || 0), 1);
        items.forEach((item) => {
          const row = document.createElement("div");
          row.className = "bar-row";
          if (tooltipBuilder) {
            row.title = tooltipBuilder(item);
          }

          const label = document.createElement("div");
          label.className = "bar-label";
          label.textContent = item[labelKey];

          const track = document.createElement("div");
          track.className = "bar-track";

          const fill = document.createElement("div");
          fill.className = `bar-fill ${className}`.trim();
          fill.style.width = `${((Number(item[valueKey]) || 0) / max) * 100}%`;
          track.appendChild(fill);

          const value = document.createElement("div");
          value.className = "bar-value";
          value.textContent = number.format(item[valueKey]);

          row.append(label, track, value);
          target.appendChild(row);
        });
      }

      function renderTimeline(targetId, items, getCount, tooltipBuilder, className = "") {
        const target = document.getElementById(targetId);
        target.innerHTML = "";
        const max = Math.max(...items.map((item) => getCount(item)), 1);

        items.forEach((item) => {
          const wrap = document.createElement("div");
          wrap.className = "timeline-bar-wrap";
          wrap.tabIndex = 0;

          const value = document.createElement("div");
          value.className = "timeline-value";
          value.textContent = getCount(item);

          const tooltip = document.createElement("div");
          tooltip.className = "timeline-tooltip";
          tooltip.innerHTML = tooltipBuilder(item);

          const bar = document.createElement("div");
          bar.className = `timeline-bar ${className}`.trim();
          bar.style.height = `${Math.max((getCount(item) / max) * 156, 8)}px`;

          const year = document.createElement("div");
          year.className = "timeline-year";
          year.textContent = item.year;

          wrap.append(value, bar, year, tooltip);
          target.appendChild(wrap);
        });
      }

      function qualityTone(label) {
        if (label === "High") return "strong";
        if (label === "Medium-High" || label === "Medium") return "medium";
        if (label === "Low-Medium") return "partial";
        return "weak";
      }

      function appendOverview() {
        const target = document.getElementById("overview");
        const items = [
          {
            kicker: "Dated timeline rows",
            value: number.format(data.timeline.totalEvents),
            note: "The strongest structured historical source in the archive.",
            definition: "Number of dated implementation rows extracted from the By Date sheet of the timeline workbook.",
            source: SOURCES.timeline,
          },
          {
            kicker: "Exam archive school sheets",
            value: number.format(data.legacyOutcomes.examArchive.schoolSheets),
            note: "Structured older student outcome coverage across seven workbooks.",
            definition: "Count of school-named worksheets across the legacy national exam workbooks after excluding average, links, and comparison tabs.",
            source: SOURCES.examArchive,
          },
          {
            kicker: "Graduate survey respondents",
            value: number.format(data.legacyOutcomes.graduateSurvey.respondents),
            note: "Older alumni responses that can be cleaned into a reusable outcomes table.",
            definition: "Count of valid respondent rows in the older Form 4 graduate survey workbook after filtering out summary lines and blanks.",
            source: SOURCES.graduateSurvey,
          },
          {
            kicker: "Equipment inventory rows",
            value: number.format(data.equipment.rowCount),
            note: "A strong base for a future installations and assets master table.",
            definition: "Count of school-level rows in the equipment deployment workbook.",
            source: SOURCES.equipment,
          },
        ];

        items.forEach((item) => {
          const card = document.createElement("div");
          card.className = `overview-card ${item.emphasis ? "emphasis" : ""}`.trim();
          card.innerHTML = `
            <div class="overview-kicker">${item.kicker}${tooltipMarkup(item.definition, item.source)}</div>
            <div class="overview-value">${item.value}</div>
            <div class="overview-note">${item.note}</div>
          `;
          target.appendChild(card);
        });
      }

      function appendQualityCards() {
        const target = document.getElementById("quality-grid");
        const sourceMap = {
          "Implementation timeline": SOURCES.timeline,
          "Equipment inventory": SOURCES.equipment,
          "National exam archive": SOURCES.examArchive,
          "Graduate survey workbook": SOURCES.graduateSurvey,
          "Historical enrollment and annual counts": `${SOURCES.enrollmentHistory}; ${SOURCES.annual2018}`,
          "Graduate tracking workbook": SOURCES.graduateTracking,
          "Active site survey workbook": "Monitoring and Evaluation/PEF School Site Surveys.xlsx",
        };
        data.quality.sources.forEach((item) => {
          const card = document.createElement("div");
          card.className = "quality-card";
          card.innerHTML = `
            <h4>${item.name}${tooltipMarkup("Extraction quality reflects how directly and consistently the current file could be summarized into the dashboard.", sourceMap[item.name] || SOURCES.quality)}</h4>
            <p><span class="chip ${qualityTone(item.quality)}">${item.quality}</span></p>
            <p>${item.detail}</p>
          `;
          target.appendChild(card);
        });
      }

      function appendTimelineFooter(targetId, items) {
        const target = document.getElementById(targetId);
        target.innerHTML = "";

        items.forEach((text) => {
          const pill = document.createElement("div");
          pill.className = "pill";
          pill.textContent = text;
          target.appendChild(pill);
        });
      }

      function appendTrainingKv() {
        const target = document.getElementById("training-kv");
        const rows = [
          ["Training-only rows", number.format(data.trainingInterpretation.categories.find((item) => item.key === "training_only").count), "Rows whose implementation label is training-related and does not also name a deployment or upgrade."],
          ["Schools with any training rows", number.format(data.trainingInterpretation.trainingSchoolCount), "Distinct school or entity labels that appear on at least one training row in the timeline."],
          ["Also have deployment rows", number.format(data.trainingInterpretation.trainingAndDeploymentSchoolCount), "Training schools that also appear on at least one deployment or upgrade row in the same workbook."],
          ["Training-only entities", `${number.format(data.trainingInterpretation.trainingOnlySchoolCount)} mostly group or district entries`, "Entity labels with training rows but no matched deployment row; many are group or district-level entries rather than individual schools."],
        ];

        rows.forEach(([label, value, definition]) => {
          const div = document.createElement("div");
          div.className = "kv-row";
          div.innerHTML = `<span>${label}${tooltipMarkup(definition, SOURCES.timeline)}</span><span>${value}</span>`;
          target.appendChild(div);
        });
      }

      function appendMetricTiles(targetId, items) {
        const target = document.getElementById(targetId);
        items.forEach((item) => {
          const tile = document.createElement("div");
          tile.className = "metric-tile";
          tile.innerHTML = `
            <div class="metric-kicker">${item.kicker}${tooltipMarkup(item.definition, item.source)}</div>
            <div class="metric-value">${item.value}</div>
          `;
          target.appendChild(tile);
        });
      }

      function appendSupportCards() {
        const target = document.getElementById("support-grid");
        const cards = [
          {
            title: "Enrollment history",
            meta: `${number.format(data.legacyOutcomes.enrollmentHistory.rows)} rows across ${data.legacyOutcomes.enrollmentHistory.schools} schools`,
            detail: `${data.legacyOutcomes.enrollmentHistory.yearStart}-${data.legacyOutcomes.enrollmentHistory.yearEnd}. ${data.legacyOutcomes.enrollmentHistory.detail}`,
            definition: "Historic annual school-count rows from the Karatu Schools sheet, useful for within-school baseline reconstruction.",
            source: SOURCES.enrollmentHistory,
          },
          {
            title: "2018 annual school compilation",
            meta: `${number.format(data.legacyOutcomes.annual2018.studentTotal)} students and ${number.format(data.legacyOutcomes.annual2018.teacherTotal)} teachers`,
            detail: `${data.legacyOutcomes.annual2018.schoolsWithStudentTotals} of ${data.legacyOutcomes.annual2018.schools} schools have student totals. ${data.legacyOutcomes.annual2018.detail}`,
            definition: "Cross-school 2018 snapshot of student and teacher counts taken from the compiled annual school workbook.",
            source: SOURCES.annual2018,
          },
        ];

        cards.forEach((item) => {
          const card = document.createElement("div");
          card.className = "support-card";
          card.innerHTML = `
            <h4>${item.title}${tooltipMarkup(item.definition, item.source)}</h4>
            <p><span class="chip medium">${item.meta}</span></p>
            <p>${item.detail}</p>
          `;
          target.appendChild(card);
        });
      }

      function appendLearningCards() {
        const target = document.getElementById("learning-grid");
        const items = [
          {
            title: "How program activity expanded over time",
            detail: "The timeline can already show when implementation activity accelerated, where activity clusters sit over time, and how training tended to accompany rollouts.",
            evidence: `${number.format(data.timeline.totalEvents)} dated rows across ${number.format(data.timeline.schoolCount)} schools`,
            definition: "Chronology of implementation activity, not a count of unique projects.",
            source: SOURCES.timeline,
          },
          {
            title: "Which school groups have exam trends we can study",
            detail: "The exam archive is already deep enough to compare pass-rate patterns by school group and to test within-school before/after questions once school IDs are normalized.",
            evidence: `${number.format(data.legacyOutcomes.examArchive.schoolSheets)} school sheets from ${data.legacyOutcomes.examArchive.yearStart}-${data.legacyOutcomes.examArchive.yearEnd}`,
            definition: "Historic school-level exam time series preserved across the legacy workbooks.",
            source: SOURCES.examArchive,
          },
          {
            title: "What alumni pathways are visible today",
            detail: "The graduate survey and recent graduate tracking already give partial visibility into employment, further education, vocational training, and perceived value of computer skills.",
            evidence: `${number.format(data.legacyOutcomes.graduateSurvey.respondents)} survey respondents and ${data.graduates.years.length} recent tracking years`,
            definition: "Mixed alumni outcome evidence drawn from one older row-level survey and one recent tracking workbook.",
            source: `${SOURCES.graduateSurvey}; ${SOURCES.graduateTracking}`,
          },
          {
            title: "Which schools have usable baseline count history",
            detail: "Historic enrollment rows and the 2018 annual compilation make it possible to reconstruct at least part of the baseline picture for school size, student counts, and teacher counts.",
            evidence: `${number.format(data.legacyOutcomes.enrollmentHistory.rows)} historic count rows plus ${number.format(data.legacyOutcomes.annual2018.studentTotal)} students in 2018 snapshot`,
            definition: "Historic school-size and staffing context, uneven but good enough to anchor future school-year master tables.",
            source: `${SOURCES.enrollmentHistory}; ${SOURCES.annual2018}`,
          },
        ];

        items.forEach((item) => {
          const card = document.createElement("div");
          card.className = "claim-card";
          card.innerHTML = `
            <h4>${item.title}${tooltipMarkup(item.definition, item.source)}</h4>
            <p><span class="chip medium">${item.evidence}</span></p>
            <p>${item.detail}</p>
          `;
          target.appendChild(card);
        });
      }

      function appendFoundationCards() {
        const target = document.getElementById("foundation-grid");
        const priorities = [
          {
            title: "School Master Table",
            detail: "Create one canonical school table with a persistent school ID, cleaned school names, district, country, installation dates, and active/inactive status. This is the glue layer across nearly every workbook.",
            source: `${SOURCES.timeline}; ${SOURCES.equipment}; ${SOURCES.annual2018}`,
            definition: "One-row-per-school reference table used to connect historical and future data sources.",
          },
          {
            title: "School-Year Outcomes Table",
            detail: "Normalize exam, enrollment, attendance, graduate progression, and any other school-year outcomes into a consistent table keyed by school and year. This is the best path to within-school before/after analysis.",
            source: `${SOURCES.examArchive}; ${SOURCES.enrollmentHistory}; ${SOURCES.annual2018}; ${SOURCES.graduateTracking}`,
            definition: "One-row-per-school-per-year analytical table for trend and pre/post analysis.",
          },
          {
            title: "Installations And Training Log",
            detail: "Split deployment, upgrade, and training into standard activity types with shared definitions. The timeline is already strong enough to seed this, but it needs a cleaner activity taxonomy.",
            source: SOURCES.timeline,
            definition: "Standard event log that distinguishes hardware deployment, software upgrade, training, and follow-up support.",
          },
          {
            title: "Future Collection Instruments",
            detail: "Design future school and teacher forms to feed the same master tables directly, with fixed definitions for students, teachers, lab access, digital literacy, exam outcomes, and alumni follow-up.",
            source: `${SOURCES.annual2018}; ${SOURCES.graduateSurvey}; ${SOURCES.graduateTracking}`,
            definition: "Repeatable survey and reporting forms aligned to the cleaned historical data model.",
          },
        ];

        priorities.forEach((item) => {
          const card = document.createElement("div");
          card.className = "caveat-card";
          card.innerHTML = `<h4>${item.title}${tooltipMarkup(item.definition, item.source)}</h4><p>${item.detail}</p>`;
          target.appendChild(card);
        });
      }

      const workbenchStamp = document.getElementById("generated-at");
      if (workbenchStamp) workbenchStamp.textContent =
        `Generated from local files on ${new Date(data.generatedAt).toLocaleString()}`;

      injectDocAnchors();
      appendOverview();
      appendQualityCards();
      renderTimeline(
        "deployment-timeline-chart",
        data.timeline.byYear,
        (item) => item.deploymentSchools || 0,
        (item) => {
          const topLabels = (item.topDeploymentLabels || [])
            .map((entry) => `${entry.label} (${entry.count})`)
            .join(", ");
          return `
            <strong>Year ${item.year}</strong>
            Deployment schools: ${item.deploymentSchools}<br>
            Deployment rows: ${item.deploymentRows}<br>
            Training rows in same year: ${item.trainingRows}<br>
            Overlap with training: ${item.overlapSchools}<br>
            All activity rows: ${item.activityRows}<br>
            All school labels: ${item.schools}${topLabels ? `<br>Top deployment labels: ${escapeHtml(topLabels)}` : "<br>Top deployment labels: none"}
          `;
        }
      );
      appendTimelineFooter("deployment-timeline-footer", [
        `Counts schools with deployment or upgrade activity`,
        `${number.format(data.timeline.districtCount)} normalized district labels`,
        `${number.format(data.timeline.totalEvents)} total dated activity rows in source`,
      ]);
      renderTimeline(
        "training-only-timeline-chart",
        data.timeline.byYear,
        (item) => item.uniqueTrainingSchools || 0,
        (item) => {
          const topLabels = (item.topUniqueTrainingLabels || [])
            .map((entry) => `${entry.label} (${entry.count})`)
            .join(", ");
          return `
            <strong>Year ${item.year}</strong>
            Training-only school/entity labels: ${item.uniqueTrainingSchools}<br>
            Training rows in year: ${item.trainingRows}<br>
            Training labels overlapping deployment: ${item.overlapSchools}<br>
            Deployment schools in same year: ${item.deploymentSchools}<br>
            All activity rows: ${item.activityRows}${topLabels ? `<br>Top training-only labels: ${escapeHtml(topLabels)}` : "<br>Top training-only labels: none"}
          `;
        },
        "training"
      );
      appendTimelineFooter("training-only-timeline-footer", [
        `Counts training labels not matched to deployment in the same year`,
        `Useful for spotting standalone training activity or messy labeling`,
        `Hover any bar for overlap and row context`,
      ]);
      document.getElementById("training-summary").textContent = data.trainingInterpretation.summary;
      appendTrainingKv();
      renderBarChart("training-categories", data.trainingInterpretation.categories, "count", "category", "", (item) => `Definition: ${item.interpretation} Source: ${SOURCES.timeline}`);
      renderBarChart("training-pairings", data.trainingInterpretation.projectPairings, "count", "type", "alt", (item) => `Definition: Count of school-year combinations where training appears with ${item.type}. Source: ${SOURCES.timeline}`);

      appendMetricTiles("exam-stats", [
        { kicker: "Workbooks", value: number.format(data.legacyOutcomes.examArchive.workbooks), definition: "Distinct legacy national exam workbook files included in the exam archive summary.", source: SOURCES.examArchive },
        { kicker: "School sheets", value: number.format(data.legacyOutcomes.examArchive.schoolSheets), definition: "School-named tabs across the exam workbooks after excluding link, average, and comparison tabs.", source: SOURCES.examArchive },
        { kicker: "% passed points", value: number.format(data.legacyOutcomes.examArchive.pctPassPoints), definition: "Usable numeric values found in the percent-passed series across school sheets.", source: SOURCES.examArchive },
        { kicker: "Year span", value: `${data.legacyOutcomes.examArchive.yearStart}-${data.legacyOutcomes.examArchive.yearEnd}`, definition: "Earliest and latest years observed in the extracted exam time series.", source: SOURCES.examArchive },
      ]);
      renderBarChart("exam-groups", data.legacyOutcomes.examArchive.groups, "schoolSheets", "group", "blue", (item) => `Definition: Number of school sheets in the ${item.group} workbook group; years ${item.yearStart}-${item.yearEnd}. Source: ${SOURCES.examArchive}`);

      appendMetricTiles("survey-stats", [
        { kicker: "Respondents", value: number.format(data.legacyOutcomes.graduateSurvey.respondents), definition: "Valid respondent rows in the graduate survey workbook after excluding summary and blank rows.", source: SOURCES.graduateSurvey },
        { kicker: "Has a job", value: number.format(data.legacyOutcomes.graduateSurvey.hasJob), definition: "Respondents marked yes to having a job in the graduate survey workbook.", source: SOURCES.graduateSurvey },
        { kicker: "Tech-related job", value: number.format(data.legacyOutcomes.graduateSurvey.techRelatedJob), definition: "Respondents marked yes to doing a computer or technology related job.", source: SOURCES.graduateSurvey },
        { kicker: "Computer helped job", value: number.format(data.legacyOutcomes.graduateSurvey.computerHelpedJob), definition: "Respondents who said computer skills helped them find a job.", source: SOURCES.graduateSurvey },
        { kicker: "Went to high school", value: number.format(data.legacyOutcomes.graduateSurvey.wentHighSchool), definition: "Respondents who marked yes to continuing to high school after Form 4.", source: SOURCES.graduateSurvey },
        { kicker: "Vocational training", value: number.format(data.legacyOutcomes.graduateSurvey.vocationalTraining), definition: "Respondents who marked yes to attending vocational training; free-text institution answers are not separately standardized here.", source: SOURCES.graduateSurvey },
        { kicker: "College or university", value: number.format(data.legacyOutcomes.graduateSurvey.collegeOrUniversity), definition: "Respondents who marked yes to attending college or university.", source: SOURCES.graduateSurvey },
        { kicker: "School coverage", value: `${number.format(data.legacyOutcomes.graduateSurvey.schools)} schools`, definition: "Normalized distinct school names represented in the graduate survey respondent rows.", source: SOURCES.graduateSurvey },
      ]);

      renderBarChart("graduate-years", data.graduates.years, "postSecondaryPlacements", "year", "", (item) => `Definition: Comparable post-secondary placements for ${item.year}, calculated as advance school plus college/university. Source: ${SOURCES.graduateTracking}`);
      appendSupportCards();
      appendLearningCards();
      appendFoundationCards();

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
        const sync = window.SYNC_METADATA;
        const syncNote = sync
          ? ` · published from ${sync.sourceRepo}@${sync.sourceCommitShort}, synced ${new Date(sync.syncedAt).toLocaleDateString()}`
          : "";
        document.getElementById("hero-health").textContent =
          `School master generated ${new Date(program.generatedAt).toLocaleString()} · archive extract ${new Date(archive.generatedAt).toLocaleString()} · ${net.inactive} inactive sites · 2024 not logged${syncNote}`;
      }

      function renderMismatches() {
        document.getElementById("mismatch-list").innerHTML = program.mismatches.map((item) => `
          <div class="alert ${item.severity}">
            <strong>${item.title}</strong>
            <p>${item.detail}</p>
          </div>
        `).join("");
      }

      function severityChip(severity) {
        const s = String(severity || "low").toLowerCase();
        return `<span class="severity-chip ${s}">${s}</span>`;
      }

      function discrepancyStatusBadge(status) {
        const s = String(status || "open").toLowerCase();
        const cls = s.startsWith("wontfix") ? "closed" : s === "resolved" ? "resolved" : "open";
        return `<span class="discrepancy-status ${cls}">${status}</span>`;
      }

      function renderKnownDiscrepancies() {
        const table = document.getElementById("discrepancy-table");
        if (!table) return;
        const items = program.knownDiscrepancies || [];
        const rows = [
          `<tr><th>Issue</th><th>Severity</th><th>Owner</th><th>Status</th></tr>`,
          ...items.map((item) => `<tr>
            <td><strong>${item.title}</strong><div class="panel-copy" style="margin:4px 0 0;font-size:0.86rem;">${item.detail}</div></td>
            <td>${severityChip(item.severity)}</td>
            <td>${item.owner || "Unassigned"}</td>
            <td>${discrepancyStatusBadge(item.status)}</td>
          </tr>`),
        ];
        table.innerHTML = rows.join("");
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

      function examCell(exam) {
        if (!exam || exam.passRatePct == null) return "—";
        return `${exam.passRatePct}% <span class="panel-copy" style="display:inline;font-size:0.82rem;">(${exam.passed}/${exam.sat}${exam.year ? `, ${exam.year}` : ""})</span>`;
      }

      function renderSchoolOutcomes() {
        const coverage = program.outcomesCoverage;
        const outcomes = program.schoolOutcomes || {};
        if (!coverage) return;

        document.getElementById("outcomes-coverage-stats").innerHTML = [
          { kicker: "Schools with a baseline survey", value: `${coverage.schoolsWithBaseline} of ${coverage.totalSchools}` },
          { kicker: "Schools with any follow-up", value: `${coverage.schoolsWithFollowup} of ${coverage.totalSchools}` },
          { kicker: "Schools with a before/after pair", value: `${coverage.schoolsWithBeforeAfter} of ${coverage.totalSchools}` },
          { kicker: "Latest follow-up school year", value: coverage.latestFollowupYear ?? "—" },
        ].map((item) => `
          <div class="metric-tile">
            <div class="metric-kicker">${item.kicker}</div>
            <div class="metric-value">${item.value}</div>
          </div>
        `).join("");

        const withData = Object.values(outcomes).filter((o) => o.baseline || o.annual.length);
        const emptyState = document.getElementById("outcomes-empty-state");
        const tableWrap = document.getElementById("outcomes-table-wrap");

        if (!withData.length) {
          emptyState.textContent = "No survey data yet. Once a school has a baseline and at least one annual follow-up survey (see SCHOOL_SURVEY_PROCESS.md), it shows up here with real before/after numbers.";
          emptyState.classList.remove("hidden");
          tableWrap.classList.add("hidden");
          return;
        }
        emptyState.classList.add("hidden");
        tableWrap.classList.remove("hidden");

        const rows = withData
          .sort((a, b) => a.school.localeCompare(b.school))
          .map((o) => {
            const latestAnnual = o.annual[o.annual.length - 1];
            const beforeEnroll = o.baseline ? o.baseline.totalEnrollment ?? "—" : "—";
            const afterEnroll = latestAnnual ? latestAnnual.totalEnrollment ?? "—" : "—";
            const beforeExam = examCell(o.baseline && o.baseline.exam);
            const afterExam = examCell(latestAnnual && latestAnnual.exam);
            const labStatus = latestAnnual && latestAnnual.labFunctional != null
              ? (latestAnnual.labFunctional ? "Functional" : "Not functional")
              : "—";
            return `<tr>
              <td>${o.school}</td>
              <td>${beforeEnroll}</td>
              <td>${afterEnroll}</td>
              <td>${beforeExam}</td>
              <td>${afterExam}</td>
              <td>${labStatus}</td>
            </tr>`;
          });
        document.getElementById("outcomes-table").innerHTML = [
          `<tr><th>School</th><th>Enrollment (baseline)</th><th>Enrollment (latest)</th><th>Exam pass rate (baseline)</th><th>Exam pass rate (latest)</th><th>Lab status (latest)</th></tr>`,
          ...rows,
        ].join("");
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
      renderKnownDiscrepancies();
      renderKaratu();
      renderFilters();
      renderSchoolTable();
      renderActivity();
      renderKeyPanels();
      renderProgramOutcomes();
      renderSchoolOutcomes();
      setView(location.hash === "#workbench" ? "workbench" : "program");
} catch (err) {
  console.error("Powering Potential dashboard failed to render:", err);
  var banner = document.createElement("div");
  banner.className = "dashboard-fatal-error";
  banner.innerHTML = "<strong>This dashboard could not load its data.</strong> " +
    "Check the browser console for details, or contact Charlie Wilson. (" + err.message + ")";
  var page = document.querySelector(".page") || document.body;
  page.insertBefore(banner, page.firstChild);
}
