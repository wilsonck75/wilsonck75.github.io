#!/usr/bin/env python3
"""Regenerate dashboard.css and dashboard.js from their sources.

index.html itself is a plain, hand-maintained HTML file (no generation
needed) that loads two static assets:

  - dashboard.css = the Workbench styling extracted from
    workbench-original.html, followed by the Program-view-only additions in
    program-extra.css.
  - dashboard.js = the Workbench rendering logic extracted from
    workbench-original.html, followed by the Program-view-only logic in
    program.js.

workbench-original.html stays as the single hand-authored source for the
"Cleaning The Archive" (Workbench) view's CSS and JS, so a change there
(e.g. a new chart, a style tweak) flows into the combined dashboard without
duplicating that view's code anywhere else. Program-view-only styling and
behavior lives directly in program-extra.css / program.js, as normal
(reviewable, lintable, diffable) source files rather than Python string
literals.

Run this after editing workbench-original.html, program-extra.css, or
program.js. It does NOT touch program-data.js (see build_program_data.py)
or index.html.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

WORKBENCH_SOURCE = ROOT / "workbench-original.html"
PROGRAM_EXTRA_CSS = ROOT / "program-extra.css"
PROGRAM_JS = ROOT / "program.js"
OUT_CSS = ROOT / "dashboard.css"
OUT_JS = ROOT / "dashboard.js"

# The exact substring markers used to slice CSS/JS out of the hand-authored
# workbench HTML file. If workbench-original.html's structure changes such
# that these markers are missing, this script fails loudly rather than
# silently producing a truncated/garbled bundle.
DASHBOARD_DATA_SCRIPT_TAG = '<script src="./dashboard-data.js"></script>'
GENERATED_AT_ASSIGNMENT = 'document.getElementById("generated-at").textContent ='
GENERATED_AT_REPLACEMENT = (
    'const workbenchStamp = document.getElementById("generated-at");\n'
    "      if (workbenchStamp) workbenchStamp.textContent ="
)


ERROR_BOUNDARY_TEMPLATE = """try {{
{body}
}} catch (err) {{
  console.error("Powering Potential dashboard failed to render:", err);
  var banner = document.createElement("div");
  banner.className = "dashboard-fatal-error";
  banner.innerHTML = "<strong>This dashboard could not load its data.</strong> " +
    "Check the browser console for details, or contact Charlie Wilson. (" + err.message + ")";
  var page = document.querySelector(".page") || document.body;
  page.insertBefore(banner, page.firstChild);
}}
"""


class AssemblyError(RuntimeError):
    """Raised when workbench-original.html doesn't match the expected shape."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssemblyError(message)


def extract_workbench_css(original_html: str) -> str:
    require("<style>" in original_html and "</style>" in original_html, "workbench-original.html is missing a <style> block")
    return original_html.split("<style>", 1)[1].split("</style>", 1)[0]


def extract_workbench_js(original_html: str) -> str:
    require(
        DASHBOARD_DATA_SCRIPT_TAG in original_html,
        f"workbench-original.html is missing the expected {DASHBOARD_DATA_SCRIPT_TAG!r} anchor",
    )
    after_data_script = original_html.split(DASHBOARD_DATA_SCRIPT_TAG, 1)[1]
    require("<script>" in after_data_script, "workbench-original.html is missing the inline <script> block after dashboard-data.js")
    script = after_data_script.replace("<script>", "", 1)
    script = script.rsplit("</script>", 1)[0].rsplit("</body>", 1)[0].rsplit("</html>", 1)[0].strip()

    # The Program view moves the "generated at" stamp out of the hero and
    # into the Workbench footer; guard the DOM lookup so the shared script
    # doesn't throw when that element isn't the first thing on the page.
    if GENERATED_AT_ASSIGNMENT in script:
        script = script.replace(GENERATED_AT_ASSIGNMENT, GENERATED_AT_REPLACEMENT)
    return script


def wrap_with_error_boundary(js: str) -> str:
    """Wrap the combined script in one top-level try/catch.

    The Workbench and Program scripts share top-level scope (Program reuses
    `number` and other bindings the Workbench script declares), so they
    can't be isolated into separate error boundaries without a larger
    rewrite. A single boundary around the whole bundle is a meaningful
    improvement over the status quo (a silent blank page on any JS error)
    without requiring that rewrite.
    """
    return ERROR_BOUNDARY_TEMPLATE.format(body=js)


def main() -> None:
    original_html = WORKBENCH_SOURCE.read_text()

    workbench_css = extract_workbench_css(original_html)
    workbench_js = extract_workbench_js(original_html)
    program_extra_css = PROGRAM_EXTRA_CSS.read_text()
    program_js = PROGRAM_JS.read_text()

    combined_js = workbench_js.rstrip("\n") + "\n\n" + program_js.rstrip("\n")

    OUT_CSS.write_text(workbench_css.rstrip("\n") + "\n\n" + program_extra_css.rstrip("\n") + "\n")
    OUT_JS.write_text(wrap_with_error_boundary(combined_js))

    print(f"Wrote {OUT_CSS} ({OUT_CSS.stat().st_size} bytes)")
    print(f"Wrote {OUT_JS} ({OUT_JS.stat().st_size} bytes)")
    print("index.html is hand-maintained and was not touched.")


if __name__ == "__main__":
    main()
