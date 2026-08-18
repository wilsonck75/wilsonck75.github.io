// Board-only access gate for the Powering Potential dashboard.
//
// IMPORTANT — read this before relying on it: GitHub Pages serves this
// entire repo as public static files. This gate hides the assembled page
// behind a shared passphrase, which stops casual/accidental visitors and
// keeps the dashboard out of search results, but it is NOT real access
// control:
//   - dashboard-data.js and program-data.js are still directly fetchable
//     by URL by anyone who knows/guesses the filename, gate or not.
//   - The passphrase hash below is visible to anyone who views source; a
//     motivated person could brute-force a weak passphrase offline.
// This is an intentional "first pass" trade-off (see PRIVACY_REVIEW.md).
// If the dashboard's content ever becomes more sensitive than the current
// school/cohort-level aggregates, replace this with real auth (e.g.
// Cloudflare Access in front of the Pages site, or a private host).
//
// To change the shared passphrase: compute a new SHA-256 hex digest (e.g.
// `python3 -c "import hashlib; print(hashlib.sha256(b'your-new-passphrase').hexdigest())"`)
// and replace PASSPHRASE_SHA256 below. Anyone with the old passphrase (or an
// old unlocked browser session) keeps access until they clear the
// `ppi-dashboard-unlocked` sessionStorage key or the hash version changes.
(function () {
  var PASSPHRASE_SHA256 = "710e3e3dae44d3e78a5d86a4285c52200f7d8cd1b6f11a0c31d7f736e07ea26e";
  var SESSION_KEY = "ppi-dashboard-unlocked-" + PASSPHRASE_SHA256.slice(0, 8);

  function unlock() {
    document.body.classList.remove("gate-locked");
    var overlay = document.getElementById("access-gate");
    if (overlay) overlay.remove();
  }

  function sha256Hex(text) {
    var encoded = new TextEncoder().encode(text);
    return crypto.subtle.digest("SHA-256", encoded).then(function (buffer) {
      return Array.prototype.map
        .call(new Uint8Array(buffer), function (b) {
          return b.toString(16).padStart(2, "0");
        })
        .join("");
    });
  }

  function showError(message) {
    var errorEl = document.getElementById("access-gate-error");
    if (errorEl) errorEl.textContent = message;
  }

  function buildOverlay() {
    var overlay = document.createElement("div");
    overlay.id = "access-gate";
    overlay.innerHTML =
      '<div class="access-gate-card">' +
      "<h1>Powering Potential dashboard</h1>" +
      "<p>This is a board/staff working view. Enter the shared passphrase to continue.</p>" +
      '<form id="access-gate-form">' +
      '<input type="password" id="access-gate-input" placeholder="Passphrase" autocomplete="off" autofocus>' +
      '<button type="submit">Unlock</button>' +
      "</form>" +
      '<p id="access-gate-error" class="access-gate-error"></p>' +
      "<p class=\"access-gate-note\">Don't have the passphrase? Ask Charlie Wilson or Caitlin Kelley.</p>" +
      "</div>";
    document.body.appendChild(overlay);

    document.getElementById("access-gate-form").addEventListener("submit", function (event) {
      event.preventDefault();
      var input = document.getElementById("access-gate-input").value;
      sha256Hex(input).then(function (hash) {
        if (hash === PASSPHRASE_SHA256) {
          try {
            sessionStorage.setItem(SESSION_KEY, "1");
          } catch (err) {
            // sessionStorage unavailable (e.g. private browsing lockdown); the
            // gate will just re-prompt on the next page load, which is fine.
          }
          unlock();
        } else {
          showError("That passphrase didn't match. Try again.");
        }
      });
    });
  }

  var alreadyUnlocked = false;
  try {
    alreadyUnlocked = sessionStorage.getItem(SESSION_KEY) === "1";
  } catch (err) {
    alreadyUnlocked = false;
  }

  if (alreadyUnlocked) {
    unlock();
  } else {
    buildOverlay();
  }
})();
