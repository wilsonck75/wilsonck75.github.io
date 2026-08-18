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
      setView(location.hash === "#workbench" ? "workbench" : "program");
