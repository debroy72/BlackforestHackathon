// ----- Config -----
const API_BASE = "http://localhost:8000";
const el = (q) => document.querySelector(q);

// small helper
function escapeHtml(s = "") {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function api(path, opts = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

// ----- Recipes -----
async function loadRecipes() {
  const q = el("#search").value.trim();
  const max = el("#max").value.trim();
  const semantic = el("#semantic")?.checked;

  let url;
  if (semantic && q) {
    url = `/recipes/semantic?limit=100&q=${encodeURIComponent(q)}`;
    if (max) url += `&max_co2=${encodeURIComponent(max)}`;
  } else {
    url = `/recipes?limit=100`;
    if (q) url += `&q=${encodeURIComponent(q)}`;
    if (max) url += `&max_co2=${encodeURIComponent(max)}`;
  }

  try {
    const data = await api(url);
    renderRecipes(data);
  } catch (err) {
    const body = el("#recipesBody");
    body.innerHTML = `<tr><td colspan="2" class="muted">Error loading recipes: ${escapeHtml(err.message || String(err))}</td></tr>`;
  }
}

function renderRecipes(list) {
  const body = el("#recipesBody");
  body.innerHTML = "";
  if (!list.length) {
    body.innerHTML = `<tr><td colspan="2" class="muted">No recipes found.</td></tr>`;
    return;
  }
  for (const r of list) {
    const tr = document.createElement("tr");
    const tdName = document.createElement("td");
    const tdCo2 = document.createElement("td");
    tdName.innerHTML = `<button class="link" data-name="${escapeHtml(r.name)}">${escapeHtml(r.name)}</button>`;
    tdCo2.className = "co2";
    tdCo2.textContent = Number(r.carbon_kg).toFixed(3);
    tr.appendChild(tdName);
    tr.appendChild(tdCo2);
    body.appendChild(tr);
  }
  body.querySelectorAll("button.link").forEach((btn) => {
    btn.style.background = "transparent";
    btn.style.border = "none";
    btn.style.color = "#2b7a3d";
    btn.style.cursor = "pointer";
    btn.addEventListener("click", () => openDetail(btn.dataset.name));
  });
}
const seasonalBtn = document.getElementById("btnSeasonal");
if (seasonalBtn) seasonalBtn.addEventListener("click", loadSeasonal);

async function loadSeasonal() {
  try {
    // default: at least 3 ingredients in season; you can tweak here or expose UI controls
    const data = await api(`/recipes/seasonal?min_in_season=3&min_high_season=0&limit=100`);
    renderRecipes(data);
  } catch (err) {
    const body = el("#recipesBody");
    body.innerHTML = `<tr><td colspan="2" class="muted">Error loading seasonal recipes: ${escapeHtml(err.message || String(err))}</td></tr>`;
  }
}

async function openDetail(name) {
  // fetch the recipe
  let data;
  try {
    data = await api(`/recipes/${encodeURIComponent(name)}`);
  } catch (e) {
    console.error("Recipe detail fetch failed:", e);
    alert("Sorry, couldn’t load the recipe details.");
    return;
  }

  // title
  el("#modalTitle").innerText = data?.name || "Recipe";

  // steps
  const stepsHtml = (Array.isArray(data.steps) && data.steps.length)
    ? `
      <h4>Method</h4>
      <ol class="steps">
        ${data.steps.map(s => `<li>${escapeHtml(String(s))}</li>`).join("")}
      </ol>
    `
    : "";

  // image
  const imgHtml = data.image_url
  ? `<img src="${escapeHtml(data.image_url)}" 
          alt="${escapeHtml(data.name)}" 
          class="hero" 
          style="width: 100%; max-width: 500px; height: auto; display: block; margin: 0 auto; border-radius: 10px;" 
          onerror="this.style.display='none'">`
  : "";
  // ingredients — render as a 2-column table// ingredients — render as a 2-column table aligned left
const ingHtml = (Array.isArray(data.ingredients) && data.ingredients.length)
? `
  <h4 style="text-align:left; margin-top:1rem;">Ingredients</h4>
  <div class="table-wrap" style="margin-left:0; margin-right:auto; width:85%; text-align:left;">
    <table class="table small" style="width:100%; text-align:left;">
      <thead>
        <tr style="background-color:#eefaf0; color:#134e19;">
          <th style="text-align:left; padding:8px 12px;">Ingredient</th>
          <th class="co2" style="text-align:left; padding:8px 12px;">Amount</th>
        </tr>
      </thead>
      <tbody>
        ${data.ingredients.map(i => `
          <tr>
            <td style="text-align:left; padding:6px 12px;">${escapeHtml(i.name)}</td>
            <td class="co2" style="text-align:left; padding:6px 12px;">${Math.round(Number(i.quantity_g) || 0)} g</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  </div>
`
: `<div class="muted">No ingredients found.</div>`;

  // swaps (from backend heuristic)
  const swapsHtml = (Array.isArray(data.swaps) && data.swaps.length)
    ? `
      <h4>Lower-carbon, healthier swaps</h4>
      <div class="table-wrap">
        <table class="table small">
          <thead>
            <tr>
              <th>Ingredient</th>
              <th>Suggested swap</th>
              <th class="co2">Est. CO₂e saved</th>
            </tr>
          </thead>
          <tbody>
            ${data.swaps.map(sw => `
              <tr>
                <td>${escapeHtml(sw.original)}</td>
                <td title="${escapeHtml(sw.reason)}">${escapeHtml(sw.suggestion)}</td>
                <td class="co2">${Number(sw.est_saving_kg).toFixed(3)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
      <p class="micro muted">Savings are estimates based on ingredient intensities.</p>
    `
    : "";

  // base body
  el("#modalBody").innerHTML = `
    ${imgHtml}
    <div class="kv"><div class="k" style="font-weight:700; color:#0b3d1a;">Total footprint</div>
      <div><strong>${Number(data.carbon_kg || 0).toFixed(3)} kg CO₂e</strong></div>
    </div>
    ${ingHtml}
    ${swapsHtml}
    ${stepsHtml}

    <!-- AI section placeholder -->
    <div id="aiAdviceWrap" style="margin-top:18px">
      <h4>Chef Klima (AI) suggestions</h4>
      <div class="muted" id="aiAdvice">Thinking…</div>
    </div>
  `;

  // show immediately
  el("#modal").showModal();

  // fetch AI advice (non-blocking)
  try {
    const ai = await api(`/recipes/${encodeURIComponent(name)}/ai`);
    const box = el("#aiAdvice");
    if (ai && ai.advice) {
      const html = ai.advice
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/^\s*-\s+/gm, "• ")
        .replace(/\n/g, "<br>");
      box.classList.remove("muted");
      box.innerHTML = html;
    } else {
      box.classList.remove("muted");
      box.textContent = "No AI suggestions for this recipe.";
    }
  } catch (e) {
    console.warn("AI suggestions failed:", e);
    const box = el("#aiAdvice");
    if (box) {
      box.classList.remove("muted");
      box.textContent = "AI suggestions unavailable right now.";
    }
  }
}

// Hook up PDF download
const pdfBtn = document.querySelector("#btnPdf");
if (pdfBtn) {
  const rName = name; // capture the recipe name in this scope
  pdfBtn.addEventListener("click", () => {
    if (!rName) {
      alert("No recipe name to download.");
      return;
    }
    const url = `${API_BASE}/recipes/${encodeURIComponent(rName)}/shopping-list.pdf`;
    // Open in a new tab; avoids some popup blockers
    window.open(url, "_blank", "noopener");
  });
}

// ----- Chat (/ask) -----
function pushMsg(role, text){
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "you" ? "you" : "ai"}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  el("#chat").appendChild(wrap);
  el("#chat").scrollTop = el("#chat").scrollHeight;
}

async function sendQuestion(q){
  pushMsg("you", q);
  try {
    const res = await api(`/ask`, {
      method: "POST",
      body: JSON.stringify({ question: q })
    });
    pushMsg("ai", res.answer || "(no answer)");
  } catch (e) {
    pushMsg("ai", "Error contacting AI endpoint.");
  }
}

// ----- Wire up UI -----
el("#btnSearch").addEventListener("click", loadRecipes);
el("#search").addEventListener("keydown", (e)=>{ if(e.key==="Enter") loadRecipes(); });
el("#max").addEventListener("keydown", (e)=>{ if(e.key==="Enter") loadRecipes(); });
el("#modalClose").addEventListener("click", ()=> el("#modal").close());

const chatForm = el("#chatForm");
if (chatForm) {
  chatForm.addEventListener("submit", (e)=>{
    e.preventDefault();
    const q = el("#chatInput").value.trim();
    if (!q) return;
    el("#chatInput").value = "";
    sendQuestion(q);
  });
}

// Initial load
loadRecipes();