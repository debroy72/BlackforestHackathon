// KlimaCook - Registration Page Script
const AUTH_BASE = "http://localhost:9000";
const $ = (q) => document.querySelector(q);

const form = $("#regForm");
const statusEl = $("#status");
const btnSubmit = $("#btnSubmit");
const errUser = $("#errUser");
const errPw = $("#errPw");

// show/hide password
$("#togglePw").addEventListener("click", () => {
  const pw = $("#password");
  pw.type = pw.type === "password" ? "text" : "password";
});

// basic inline client-side validation
function validate() {
  let ok = true;
  errUser.textContent = "";
  errPw.textContent = "";

  const username = $("#username");
  if (!username.value.match(/^[a-zA-Z0-9_.-]{3,30}$/)) {
    errUser.textContent = "Use 3–30 letters/digits or . _ -";
    ok = false;
  }
  const pw = $("#password");
  if (!pw.value || pw.value.length < 8) {
    errPw.textContent = "Password must be at least 8 characters.";
    ok = false;
  }
  return ok;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!validate()) {
    statusEl.textContent = "Please fix the highlighted fields.";
    statusEl.classList.remove("success");
    statusEl.classList.add("error");
    return;
  }

  const username = $("#username").value.trim();
  const password = $("#password").value;
  const allergiesRaw = $("#allergies").value;

  const allergies = (allergiesRaw || "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);

  statusEl.textContent = "Creating account…";
  statusEl.classList.remove("error", "success");
  btnSubmit.disabled = true;

  try {
    const res = await fetch(`${AUTH_BASE}/auth/register_username`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, allergies }),
    });

    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        // FastAPI can return an array of validation errors or a string detail
        if (Array.isArray(err?.detail)) {
          msg = err.detail.map(e => e.msg || JSON.stringify(e)).join("; ");
        } else if (err?.detail) {
          msg = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
        }
      } catch {
        const text = await res.text();
        if (text) msg = text;
      }
      throw new Error(msg);
    }

    statusEl.textContent = "✅ Account created. Redirecting…";
    statusEl.classList.add("success");
    setTimeout(() => { window.location.href = "./login.html"; }, 900);
  } catch (err) {
    const msg = String(err?.message || "Registration failed");
    statusEl.textContent = `⚠️ ${msg}`;
    statusEl.classList.add("error");

    // highlight common issues
    if (msg.includes("Username already taken") || msg.includes("409")) {
      errUser.textContent = "Username already taken.";
    }
    if (msg.match(/Password.*(8|characters)/i)) {
      errPw.textContent = "Password must be at least 8 characters.";
    }
  } finally {
    btnSubmit.disabled = false;
  }
});