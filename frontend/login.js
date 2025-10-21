// KlimaCook – Login Page
const AUTH_BASE = "http://localhost:9000";
const $ = (q) => document.querySelector(q);

const form = $("#loginForm");
const statusEl = $("#status");
const btnLogin = $("#btnLogin");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = $("#username").value.trim();
  const password = $("#password").value;

  statusEl.textContent = "Signing in…";
  statusEl.style.color = "#6b7280";
  btnLogin.disabled = true;

  try {
    const res = await fetch(`${AUTH_BASE}/auth/login_json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        if (err && err.detail) msg = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
      } catch {
        const text = await res.text();
        if (text) msg = text;
      }
      throw new Error(msg);
    }

    const data = await res.json();
    localStorage.setItem("kc_token", data.access_token);
    statusEl.textContent = "✅ Logged in successfully!";
    statusEl.style.color = "green";
    setTimeout(() => (window.location.href = "./index.html"), 900);
  } catch (err) {
    statusEl.textContent = `⚠️ Login failed: ${err.message}`;
    statusEl.style.color = "red";
  } finally {
    btnLogin.disabled = false;
  }
});