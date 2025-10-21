from __future__ import annotations
import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO
from datetime import datetime
from fastapi.responses import StreamingResponse

from dotenv import load_dotenv
load_dotenv()  # read backend/.env

import threading
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from openai import AzureOpenAI

# ---------- Optional FAISS (super-fast ANN) ----------
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

# =====================================
# FastAPI setup + CORS
# =====================================
app = FastAPI(title="KlimaCook AI – Fast (CSV + Semantic Search)")

@app.get("/debug/llm")
def debug_llm():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key  = os.getenv("AZURE_OPENAI_API_KEY")
    api_ver  = os.getenv("AZURE_OPENAI_API_VERSION")
    deploy   = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    masked = (api_key[:4] + "..." + api_key[-6:]) if api_key else None
    return {
        "endpoint": endpoint or "(missing)",
        "deployment": deploy or "(missing)",
        "api_version": api_ver or "(missing)",
        "api_key_masked": masked or "(missing)",
        "configured": bool(endpoint and api_key and deploy),
        "faiss": _FAISS_OK,
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# =====================================
# Paths / cache dirs (for embeddings)
# =====================================
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
EMB_PATH = CACHE_DIR / "recipe_emb.npy"
META_PATH = CACHE_DIR / "recipe_meta.json"
FAISS_PATH = CACHE_DIR / "recipe_faiss.index"

# FAST default (overridable via env)
SBERT_MODEL_NAME = os.getenv(
    "SBERT_MODEL_NAME",
    "paraphrase-multilingual-MiniLM-L12-v2",  # fast & multilingual
)
EMB_VERSION = "v2"  # bump when you change embedding text/model

# =====================================
# Utility helpers
# =====================================
def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s.strip().lower()

def _norm_unit(u: Optional[str]) -> Optional[str]:
    return None if u is None else u.lower().strip().replace(" ", "")

# =====================================
# Category keywords + intensities
# =====================================
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "red_meat": ["rind", "rindfleisch", "kalb", "lamm", "schwein", "hackfleisch", "fleisch", "speck", "bacon", "salami", "rippe"],
    "poultry": ["huhn", "hähnchen", "ente", "pute", "truthahn", "geflügel"],
    "fish": ["fisch", "lachs", "thunfisch", "forelle", "kabeljau", "sardine", "garnele", "krabbe", "muschel"],
    "eggs": ["ei", "eier", "eigelb", "eiweiß"],
    "dairy": ["milch", "butter", "sahne", "frischkäse", "quark", "joghurt", "schmand", "kondensmilch", "rahm", "creme fraîche"],
    "cheese": ["käse", "parmesan", "mozzarella", "gouda", "emmentaler", "cheddar"],
    "legumes": ["linsen", "bohne", "kichererbse", "erbsen"],
    "grains": ["mehl", "reis", "couscous", "quinoa", "bulgur", "hirse", "nudel", "pasta", "spaghetti", "brot", "brötchen", "semmel"],
    "vegetables": ["kartoff", "karotte", "möhre", "brokkoli", "paprika", "tomate", "zwiebel", "knoblauch", "spinat", "kohl", "salat", "gurke", "zucchini", "aubergine", "porree", "lauch", "sellerie", "pilz", "champignon", "blumenkohl", "mais", "süßkartoffel", "fenchel"],
    "fruits": ["apfel", "banane", "beere", "erdbeer", "himbeer", "kirsch", "birne", "pfirsich", "zitrone", "orange", "kiwi", "traube", "ananas", "mango", "limette", "melone", "pflaume", "aprikose"],
    "sugar": ["zucker", "honig", "sirup", "agavendicksaft", "ahornsirup"],
    "nuts": ["mandel", "nuss", "walnuss", "haselnuss", "cashew", "pistazie"],
    "oil": ["öl", "olivenöl", "sonnenblumenöl", "rapsöl", "kokosöl"],
    "spice": ["pfeffer", "salz", "paprika", "curry", "muskat", "zimt", "vanille", "rosmarin", "thymian", "basilikum", "oregano", "petersilie", "dill", "koriander", "lorbeer", "ingwer"],
}

CATEGORY_INTENSITY_FALLBACK: Dict[str, float] = {
    "red_meat": 27.0, "poultry": 7.0, "fish": 12.0, "eggs": 5.0,
    "dairy": 2.0, "cheese": 14.0, "legumes": 1.0, "grains": 3.0,
    "vegetables": 2.0, "fruits": 2.0, "sugar": 1.0, "nuts": 2.0,
    "oil": 3.0, "spice": 0.5, "other": 0.5,
}

UNIT_TO_G: Dict[str, float] = {
    "g": 1.0, "kg": 1000.0, "ml": 1.0, "l": 1000.0,
    "tl": 5.0, "teelöffel": 5.0, "el": 15.0, "esslöffel": 15.0,
    "bund": 100.0, "bd.": 100.0, "prise": 1.0, "päckchen": 10.0, "becher": 200.0,
    "stück": 50.0, "scheibe": 30.0, "dose": 400.0, "glas": 250.0, "tasse": 200.0, "messerspitze": 1.0,
}

def to_grams(qty: Optional[float], unit: Optional[str]) -> float:
    if qty is None:
        return 100.0
    return float(qty) * UNIT_TO_G.get(_norm_unit(unit), 1.0)

# =====================================
# CSV: load custom factors
# =====================================
ING_NAME_COLS = {
    "ingredient", "item", "name", "food", "product",
    "ingredient_name", "food_name", "item_name"
}
CAT_NAME_COLS = {
    "category", "group", "category_name", "group_name", "cat"
}
INTENSITY_COLS = {
    "kg_co2_per_kg", "co2_kg_per_kg", "intensity", "emission", "emissions", "co2e_kg_per_kg",
    # common alternates
    "kgco2e_per_kg", "kgco2_per_kg", "co2e_kgkg", "kg_co2e/kg", "co2e_per_kg"
}
BASIS_COLS = {
    "basis", "per", "unit_basis", "unit", "per_unit", "reference_amount", "amount", "per_amount"
}
PER100G_FLAGS = {
    "per_100g", "is_per_100g", "per100g", "per_100_g", "per100"
}

def _parse_float(x: str) -> Optional[float]:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None

def _detect_cols(header: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    ing_col = next((h for h in header if h.lower().strip() in ING_NAME_COLS), None)
    cat_col = next((h for h in header if h.lower().strip() in CAT_NAME_COLS), None)
    inten_col = next((h for h in header if h.lower().strip() in INTENSITY_COLS), None)
    basis_col = next((h for h in header if h.lower().strip() in BASIS_COLS or h.lower().strip() in PER100G_FLAGS), None)
    return ing_col, cat_col, inten_col, basis_col

@lru_cache(maxsize=1)
def load_custom_intensities() -> Tuple[Dict[str, float], Dict[str, float]]:
    path = os.getenv("CO2_CSV_PATH", "database.csv")
    p = Path(path)
    if not p.exists():
        return {}, {}
    ing_map: Dict[str, float] = {}
    cat_map: Dict[str, float] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        ing_col, cat_col, inten_col, basis_col = _detect_cols(header)
        if not inten_col:
            return {}, {}
        for row in reader:
            intensity = _parse_float(row.get(inten_col, ""))
            if intensity is None:
                continue
            per_kg = intensity
            if basis_col:
                basis_val = str(row.get(basis_col, "")).strip().lower()
                if basis_val in {"100g", "per_100g"}:
                    per_kg = intensity * 10.0
                if basis_val in {"1", "true", "yes"} and basis_col.lower() in PER100G_FLAGS:
                    per_kg = intensity * 10.0
            if ing_col:
                ing_name = _norm(row.get(ing_col))
                if ing_name:
                    ing_map[ing_name] = per_kg
            if cat_col:
                cat_name = _norm(row.get(cat_col))
                if cat_name:
                    cat_map[cat_name] = per_kg
    return ing_map, cat_map

# =====================================
# Classifier (disabled by default for speed)
# =====================================
MODEL_READY = True     # ready by default (we may skip loading)
MODEL_ERROR = None

@lru_cache(maxsize=1)
def _classifier():
    texts, labels = [], []
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            texts.append(kw)
            labels.append(cat)
    model = SentenceTransformer(SBERT_MODEL_NAME)
    X = model.encode(texts)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    return model, clf, le

def classify_category(name: str) -> str:
    use_transformer = os.getenv("KLIMA_USE_TRANSFORMER", "0") == "1"  # OFF by default
    if not use_transformer or not MODEL_READY:
        n = name.lower()
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(kw in n for kw in kws):
                return cat
        return "other"
    try:
        model, clf, le = _classifier()
        emb = model.encode([name])
        pred = clf.predict(emb)[0]
        return le.inverse_transform([pred])[0]
    except Exception:
        n = name.lower()
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(kw in n for kw in kws):
                return cat
        return "other"

# =====================================
# Pydantic models
# =====================================
class IngredientOut(BaseModel):
    name: str
    quantity_g: float
class SwapOut(BaseModel):
    original: str
    quantity_g: float
    suggestion: str
    reason: str
    est_saving_kg: float

class RecipeOut(BaseModel):
    name: str
    carbon_kg: float
    ingredients: Optional[List[IngredientOut]] = None
    image_url: Optional[str] = None
    steps: Optional[List[str]] = None
    swaps: Optional[List[SwapOut]] = None   # <-- add this

class QARequest(BaseModel):
    question: str

# =====================================
# Raw recipes & shaping
# =====================================
@lru_cache(maxsize=1)
def load_raw_recipes() -> List[Dict]:
    paths = [
        Path("recipes_data/einfachkochen_export_800_recipes_1.json"),
        Path("recipes_data/einfachbacken_export_200_recipes_1.json"),
    ]
    recs: List[Dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            recs.extend(json.load(f)["data"]["recipeExport"])
    return recs

def flatten_ingredients(rec: Dict) -> List[Dict]:
    out: List[Dict] = []
    for block in rec.get("ingredientBlocks", []):
        out.extend(block.get("ingredients", []))
    return out

import re
from html import unescape

STEP_SPLIT_RE = re.compile(r"(?:^\s*\d+[\).\:-]\s*|\n+\s*\d+[\).\:-]\s*|\n{2,}|<li>|</li>)", re.I)

def _strings_from_maybe_list(x) -> List[str]:
    """
    Normalize many shapes into a clean list of step strings:
    - list[str]
    - list[dict{text|description|step|plainText|html}]
    - single string with numbers/newlines/HTML <li>
    """
    def _clean(s: str) -> str:
        # strip basic HTML tags -> text, collapse whitespace
        s = unescape(s).replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        s = re.sub(r"<[^>]+>", " ", s)  # remove tags
        s = re.sub(r"\s+", " ", s).strip()
        return s

    if not x:
        return []
    # list form
    if isinstance(x, list):
        out: List[str] = []
        for it in x:
            if isinstance(it, str):
                c = _clean(it)
                if c:
                    out.append(c)
            elif isinstance(it, dict):
                s = (
                    it.get("text") or it.get("description") or it.get("step") or
                    it.get("plainText") or it.get("html") or it.get("content") or ""
                )
                c = _clean(str(s))
                if c:
                    out.append(c)
        return out
    # single string form
    if isinstance(x, str):
        c = _clean(x)
        # split by ordered-list patterns or blank lines
        parts = [p.strip(" .-") for p in STEP_SPLIT_RE.split(c) if p and p.strip(" .-")]
        if parts:
            return parts
        return [c] if c else []
    return []

def extract_steps(rec: Dict) -> List[str]:
    """
    Try many common fields found in Burda exports and similar data:
    - preparationSteps / instructions / steps (array of strings or dicts)
    - preparation / method / directions / zubereitung (string, html, or nested)
    - nested 'preparation': { steps | text | description | html | plainText }
    """
    # obvious arrays first
    for key in ("preparationSteps", "instructions", "steps"):
        if key in rec:
            steps = _strings_from_maybe_list(rec[key])
            if steps:
                return steps

    # common string fields (EN/DE variants)
    for key in ("preparation", "method", "directions", "howTo", "howto",
                "preparationText", "instructionsText", "zubereitung"):
        if key in rec:
            steps = _strings_from_maybe_list(rec[key])
            if steps:
                return steps

    # nested 'preparation' object
    prep = rec.get("preparation")
    if isinstance(prep, dict):
        for key in ("steps", "text", "description", "html", "plainText", "content"):
            if key in prep:
                steps = _strings_from_maybe_list(prep[key])
                if steps:
                    return steps

    # nested 'instructions' object
    inst = rec.get("instructions")
    if isinstance(inst, dict):
        for key in ("steps", "text", "description", "html", "plainText", "content"):
            if key in inst:
                steps = _strings_from_maybe_list(inst[key])
                if steps:
                    return steps

    # last resort: look inside a generic 'body'/'article' field if present
    for key in ("body", "article", "content"):
        if key in rec:
            steps = _strings_from_maybe_list(rec[key])
            if steps:
                return steps

    return []
def _find_first_url_in(obj) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("http://") or s.startswith("https://"):
            return s
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in {"image", "imageurl", "url", "src", "href"}:
                u = _find_first_url_in(v)
                if u:
                    return u
        for v in obj.values():
            u = _find_first_url_in(v);  # scan all
            if u:
                return u
        return None
    if isinstance(obj, list):
        for it in obj:
            u = _find_first_url_in(it)
            if u:
                return u
    return None

def extract_image_url(rec: Dict) -> Optional[str]:
    for k in ("imageUrl", "imageURL", "image_url", "image", "heroImage", "teaserImage", "thumbnail"):
        if k in rec:
            u = _find_first_url_in(rec[k])
            if u:
                return u
    for k in ("images", "assets", "media"):
        if k in rec:
            u = _find_first_url_in(rec[k])
            if u:
                return u
    return _find_first_url_in(rec)

def to_public_ingredients(ingredients: List[Dict]) -> List[IngredientOut]:
    out: List[IngredientOut] = []
    for ing in ingredients:
        grams = to_grams(ing.get("quantity"), ing.get("unit", {}).get("name") if ing.get("unit") else None)
        out.append(IngredientOut(name=ing["ingredient"]["name"], quantity_g=grams))
    return out
    # ---------------------------
# Heuristic "AI" swap rules
# ---------------------------
# Each rule returns: (replacement_name, replacement_category, short_reason)
# We keep it fast & offline so the UI stays snappy; it will still benefit from
# your CSV overrides and classifier for intensities.
def _swap_rule(name: str, category: str) -> Optional[Tuple[str, str, str]]:
    n = name.lower()

    # Name-specific overrides first
    if "butter" in n or "butter" in category:
        return ("Olivenöl (Raps-/Olive)", "oil", "Ungesättigte Fette; meist weniger CO₂e als Butter.")
    if any(k in n for k in ["sahne", "schlagsahne", "creme", "rahm"]):
        return ("Hafer- oder Soja-Cuisine", "dairy", "Weniger gesättigte Fette; idR. weniger CO₂e als Sahne.")
    if "milch" in n and "pflanz" not in n:
        return ("Hafer- oder Sojamilch", "dairy", "Ballaststoffe/Proteine; deutlich weniger CO₂e als Kuhmilch.")
    if any(k in n for k in ["ei ", "eier", "eidotter", "eigelb", "eiweiss", "eiweiß"]):
        return ("Leinsamen-Ei (1 EL Leinsamen + 3 EL Wasser)", "legumes", "Pflanzliche Bindung; weniger CO₂e.")
    if "thunfisch" in n or "fisch" in category:
        return ("Kichererbsen + Noriflocken", "legumes", "Proteinreich + Meeresaroma; deutlich weniger CO₂e.")

    # Category-based
    if category == "red_meat":
        return ("Braune Linsen + Champignons (50/50)", "legumes", "Weniger gesättigte Fette; massiv weniger CO₂e.")
    if category == "cheese":
        return ("Seidentofu + Hefeflocken", "dairy", "Proteinreich; reduziert gesättigte Fette & CO₂e.")
    if category == "poultry":
        return ("Sojagranulat oder Linsen", "legumes", "Hohes Protein bei viel weniger CO₂e.")
    return None

def suggest_swaps(ingredients: List[Dict]) -> List["SwapOut"]:
    swaps: List[SwapOut] = []
    for ing in ingredients:
        name = ing["ingredient"]["name"]
        qty = ing.get("quantity")
        unit_name = ing.get("unit", {}).get("name") if ing.get("unit") else None
        grams = to_grams(qty, unit_name)
        if grams <= 0:
            continue

        cat = classify_category(name)
        rule = _swap_rule(name, cat)
        if not rule:
            continue
        repl_name, repl_cat, reason = rule

        # Intensities (per kg)
        orig_i = resolve_intensity(name, cat)
        repl_i = resolve_intensity(repl_name, repl_cat)
        saved = max(0.0, (orig_i - repl_i) * (grams / 1000.0))

        # Skip tiny savings to keep UI clean
        if saved < 0.01:
            continue

        swaps.append(SwapOut(
            original=name,
            quantity_g=grams,
            suggestion=repl_name,
            reason=reason,
            est_saving_kg=round(saved, 3),
        ))
    return swaps
# ===========================
# Resolve intensity with CSV overrides
# ===========================
def resolve_intensity(ingredient_name: str, category: str) -> float:
    """
    Priority:
      1) ingredient-level match from CSV (normalized exact string)
      2) category-level match from CSV (e.g., 'red_meat')
      3) fallback constant (CATEGORY_INTENSITY_FALLBACK)
    """
    ing_map, cat_map = load_custom_intensities()

    key = _norm(ingredient_name)
    if key and key in ing_map:
        return ing_map[key]

    cat_key = _norm(category)
    if cat_key and cat_key in cat_map:
        return cat_map[cat_key]

    return CATEGORY_INTENSITY_FALLBACK.get(category, CATEGORY_INTENSITY_FALLBACK["other"])

def ingredient_footprints(ingredients: List[Dict]) -> List[Dict]:
    """Return [{name, grams, category, intensity_kg_per_kg, footprint_kg}] sorted desc by footprint."""
    rows = []
    for ing in ingredients:
        name = ing["ingredient"]["name"]
        qty = ing.get("quantity")
        unit_name = ing.get("unit", {}).get("name") if ing.get("unit") else None
        grams = to_grams(qty, unit_name)
        if grams <= 0:
            continue
        cat = classify_category(name)
        intensity = resolve_intensity(name, cat)  # kg CO2e per kg
        footprint = intensity * (grams / 1000.0)
        rows.append({
            "name": name,
            "grams": round(grams, 1),
            "category": cat,
            "intensity_kg_per_kg": round(float(intensity), 3),
            "footprint_kg": round(float(footprint), 3),
        })
    rows.sort(key=lambda r: -r["footprint_kg"])
    return rows


def estimate_recipe_carbon_kg(ingredients: List[Dict]) -> float:
    total = 0.0
    for ing in ingredients:
        name = ing["ingredient"]["name"]
        qty = ing.get("quantity")
        unit_name = ing.get("unit", {}).get("name") if ing.get("unit") else None
        grams = to_grams(qty, unit_name)
        kg = grams / 1000.0
        cat = classify_category(name)
        intensity = resolve_intensity(name, cat)
        total += kg * intensity
    return round(total, 3)


@lru_cache(maxsize=1)
def recipe_map() -> Dict[str, RecipeOut]:
    return {r.name: r for r in build_recipes()}

# =====================================
# Semantic search (vectorization + FAISS)
# =====================================
_embedder_singleton: Optional[SentenceTransformer] = None
_faiss_index = None
_faiss_normed = True  # we store normalized embeddings

@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = SentenceTransformer(SBERT_MODEL_NAME)
    return _embedder_singleton

def recipe_to_text(rec: Dict) -> str:
    name = rec.get("name", "")
    ings = flatten_ingredients(rec)
    ing_names = [i["ingredient"]["name"] for i in ings if i.get("ingredient") and i["ingredient"].get("name")]
    return f"{name}. Zutaten/Ingredients: " + ", ".join(ing_names)

def _current_catalog_signature() -> Dict:
    recs = load_raw_recipes()
    names = [r.get("name","") for r in recs]
    return {"version": EMB_VERSION, "model": SBERT_MODEL_NAME, "count": len(names), "names_head": names[:5]}

def _load_cached_embeddings() -> Optional[Tuple[np.ndarray, List[str]]]:
    try:
        if not (EMB_PATH.exists() and META_PATH.exists()):
            return None
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        want = _current_catalog_signature()
        if meta.get("version") != want["version"] or meta.get("model") != want["model"]:
            return None
        if meta.get("count") != want["count"]:
            return None
        embs = np.load(EMB_PATH).astype(np.float32)
        names: List[str] = meta.get("names", [])
        if embs.shape[0] != len(names):
            return None
        return embs, names
    except Exception:
        return None

def _save_cached_embeddings(embs: np.ndarray, names: List[str]) -> None:
    meta = _current_catalog_signature()
    meta["names"] = names
    np.save(EMB_PATH, embs)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _build_embeddings_from_scratch() -> Tuple[np.ndarray, List[str]]:
    recs = load_raw_recipes()
    texts = [recipe_to_text(r) for r in recs]
    names = [r.get("name","") for r in recs]
    model = _embedder()
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    _save_cached_embeddings(embs, names)
    return embs, names

def _ensure_embeddings() -> Tuple[np.ndarray, List[str]]:
    cached = _load_cached_embeddings()
    if cached is not None:
        return cached
    return _build_embeddings_from_scratch()

def _ensure_faiss(embs: np.ndarray):
    global _faiss_index
    if not _FAISS_OK:
        return
    if _faiss_index is not None:
        return
    d = embs.shape[1]
    # cosine sim on normalized vectors = inner product
    _faiss_index = faiss.IndexFlatIP(d)
    _faiss_index.add(embs)

def semantic_search(query: str, k: int = 50) -> List[Tuple[str, float]]:
    if not query.strip():
        return []
    embs, names = _ensure_embeddings()
    model = _embedder()
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)  # (1, d)

    if _FAISS_OK:
        _ensure_faiss(embs)
        D, I = _faiss_index.search(q, min(k, len(names)))  # distances = inner product
        idxs = I[0].tolist()
        scores = D[0].tolist()
        return [(names[i], float(scores[n])) for n, i in enumerate(idxs) if i >= 0]

    # fallback: brute-force but still fast for few thousand recipes
    sims = (embs @ q.T).reshape(-1)
    top_idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(names[i], float(sims[i])) for i in top_idx]

@app.get("/debug/embeddings")
def debug_embeddings():
    try:
        embs, names = _ensure_embeddings()
        return {"ok": True, "count": len(names), "dim": int(embs.shape[1]), "faiss": _FAISS_OK}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Manual warmup so first user never waits
@app.post("/warmup")
def warmup():
    try:
        embs, _ = _ensure_embeddings()
        if _FAISS_OK:
            _ensure_faiss(embs)
        return {"ok": True, "faiss": _FAISS_OK, "embeddings": int(embs.shape[0])}
    except Exception as e:
        return {"ok": False, "error": str(e)}
from functools import lru_cache
from typing import List

@lru_cache(maxsize=1)
def build_recipes() -> List[RecipeOut]:
    objs: List[RecipeOut] = []
    for rec in load_raw_recipes():
        ings = flatten_ingredients(rec)
        carbon = estimate_recipe_carbon_kg(ings)
        objs.append(
            RecipeOut(
                name=rec["name"],
                carbon_kg=carbon,
                ingredients=to_public_ingredients(ings),
                image_url=extract_image_url(rec),
                steps=extract_steps(rec) or None,
                swaps=None,  # compute lazily in get_recipe()
            )
        )
    return objs
    
# =====================================
# Endpoints
# =====================================
@app.get("/recipes", response_model=List[RecipeOut])
def list_recipes(
    q: Optional[str] = Query(None, description="Simple substring search in recipe name"),
    max_co2: Optional[float] = Query(None, description="Max carbon (kg CO₂e)"),
    limit: int = Query(50, ge=1, le=500)
) -> List[RecipeOut]:
    results = build_recipes()
    if q:
        ql = q.lower()
        results = [r for r in results if ql in r.name.lower()]
    if max_co2 is not None:
        results = [r for r in results if r.carbon_kg <= max_co2]
    results.sort(key=lambda r: r.carbon_kg)
    return results[:limit]

@app.get("/recipes/semantic", response_model=List[RecipeOut])
def list_recipes_semantic(
    q: str = Query(..., description="Semantic query (DE/EN)"),
    max_co2: Optional[float] = Query(None, description="Max carbon (kg CO₂e)"),
    limit: int = Query(50, ge=1, le=500)
) -> List[RecipeOut]:
    ranking = semantic_search(q, k=limit*3)
    rmap = recipe_map()
    items: List[RecipeOut] = []
    for name, _score in ranking:
        r = rmap.get(name)
        if not r:
            continue
        if max_co2 is not None and r.carbon_kg > max_co2:
            continue
        items.append(r)
        if len(items) >= limit:
            break
    if not items:
        return list_recipes(q=q, max_co2=max_co2, limit=limit)
    return items

@app.get("/recipes/seasonal", response_model=List[RecipeOut])
def list_seasonal_recipes(
    min_in_season: int = Query(3, ge=1, description="Minimum # ingredients with season score >=1"),
    min_high_season: int = Query(0, ge=0, description="Minimum # ingredients with season score ==2"),
    limit: int = Query(50, ge=1, le=500),
    max_co2: Optional[float] = Query(None, description="Optional max CO₂e filter"),
):
    """Return recipes with a bunch of in-season ingredients, prioritizing high-season."""
    out: List[Tuple[int, int, RecipeOut]] = []  # (high_count, in_count, recipe)
    # we need access to flattened ingredients per recipe, so use raw recs
    rmap = {r.name.lower(): r for r in build_recipes()}
    for rec in load_raw_recipes():
        name = rec.get("name", "")
        r = rmap.get(name.lower())
        if not r:
            continue
        ings = flatten_ingredients(rec)
        in_count, high_count = recipe_seasonality_stats(ings)
        if in_count >= min_in_season and high_count >= min_high_season:
            if max_co2 is None or r.carbon_kg <= max_co2:
                out.append((high_count, in_count, r))

    # sort: more high-season first, then more in-season, then lower CO2
    out.sort(key=lambda t: (-t[0], -t[1], t[2].carbon_kg))
    return [t[2] for t in out[:limit]]

@app.get("/recipes/{recipe_name}", response_model=RecipeOut)

def get_recipe(recipe_name: str) -> RecipeOut:
    name_l = recipe_name.lower()
    for r in build_recipes():
        if r.name.lower() == name_l:
            # compute swaps lazily so the list view stays fast
            if r.swaps is None:
                rec_dict = next((rec for rec in load_raw_recipes()
                                 if rec["name"].lower() == name_l), None)
                ings = flatten_ingredients(rec_dict) if rec_dict else []
                r.swaps = suggest_swaps(ings) or None
            return r
    raise HTTPException(status_code=404, detail="Recipe not found")

@app.get("/metrics/sources")
def metrics_sources():
    ing_map, cat_map = load_custom_intensities()
    return {
        "ingredient_overrides": len(ing_map),
        "category_overrides": len(cat_map),
        "csv_path": os.getenv("CO2_CSV_PATH", "database.csv"),
    }


# ---- seasonality support ----
SEASON_COLS = {"season", "seasonality", "in_season", "season_score"}

def _parse_int(x: str) -> Optional[int]:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

@lru_cache(maxsize=1)
def load_seasonality() -> Dict[str, int]:
    """
    Reads per-ingredient seasonality score (0/1/2) from the CSV indicated by CO2_CSV_PATH.
    Returns a dict of {normalized_ingredient_name: score}.
    """
    path = os.getenv("CO2_CSV_PATH", "database.csv")
    p = Path(path)
    if not p.exists():
        return {}

    season_map: Dict[str, int] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [h or "" for h in (reader.fieldnames or [])]

        # find ingredient & season columns
        ing_col = next((h for h in header if h.lower().strip() in ING_NAME_COLS), None)
        season_col = next((h for h in header if h.lower().strip() in SEASON_COLS), None)
        if not ing_col or not season_col:
            return {}

        for row in reader:
            name = _norm(row.get(ing_col))
            score = _parse_int(row.get(season_col, ""))
            if name is None or score is None:
                continue
            # clamp to 0/1/2 just in case
            score = max(0, min(2, score))
            season_map[name] = score

    return season_map

def ingredient_season_score(ingredient_name: str) -> int:
    """Return 0/1/2, defaulting to 0 when unknown."""
    return load_seasonality().get(_norm(ingredient_name) or "", 0)

def recipe_seasonality_stats(ingredients: List[Dict]) -> Tuple[int, int]:
    """
    Returns (count_in_season, count_high_season) for a flattened ingredient list.
    - 'in_season' means score >= 1
    - 'high_season' means score == 2
    """
    in_season = 0
    high_season = 0
    for ing in ingredients:
        nm = ing["ingredient"]["name"]
        sc = ingredient_season_score(nm)
        if sc >= 1:
            in_season += 1
        if sc >= 2:
            high_season += 1
    return in_season, high_season


    # ---- seasonality → small helpers for AI formatting ----
def _season_str(score: int) -> str:
    # 0 = out, 1 = in season, 2 = peak/ high season
    if score >= 2:
        return "peak season"
    if score >= 1:
        return "in season"
    return "out of season"

def _season_tag_for(name: str) -> str:
    sc = ingredient_season_score(name)
    if sc >= 2:
        return " — peak season"
    if sc >= 1:
        return " — in season"
    return ""

# =====================================
# LLM Q&A (/ask) — Azure
# =====================================
def generate_answer(question: str) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key  = os.getenv("AZURE_OPENAI_API_KEY")
    api_ver  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    deploy   = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g., gpt-4o

    missing = [name for name, val in [
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", api_key),
        ("AZURE_OPENAI_DEPLOYMENT", deploy),
    ] if not val]
    if missing:
        return f"LLM not configured. Missing {', '.join(missing)} in .env."

    try:
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_ver)
        resp = client.chat.completions.create(
            model=deploy,
            messages=[
                {"role": "system", "content": "You are KlimaCook, an expert on sustainable cooking and nutrition."},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"LLM call failed: {e}"

class AskIn(BaseModel):
    question: str
from typing import Any

@app.get("/recipes/{recipe_name}/ai", response_model=Dict[str, Any])
def get_recipe_ai(recipe_name: str):
    advice = generate_ai_recipe_advice(recipe_name)
    return {"advice": advice}
from functools import lru_cache

@lru_cache(maxsize=256)
def generate_ai_recipe_advice(recipe_name: str) -> str:
    # find the recipe and build a compact context
    name_l = recipe_name.lower()
    rec = next((r for r in load_raw_recipes() if r.get("name", "").lower() == name_l), None)
    if not rec:
        return "I couldn’t find that recipe."

    ings = flatten_ingredients(rec)
    rows = ingredient_footprints(ings)
    total = round(sum(r["footprint_kg"] for r in rows), 3)

    # Add season label alongside each ingredient (used in the prompt and as tags in bullets)
    season_lines = []
    for r in rows:
        nm = r["name"]
        season_lines.append(f'{nm}: {_season_str(ingredient_season_score(nm))}')

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key  = os.getenv("AZURE_OPENAI_API_KEY")
    api_ver  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    deploy   = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not (endpoint and api_key and deploy):
        return "AI suggestions are unavailable (LLM not configured). Ask in the chat panel instead."

    # Compact table for the prompt
    table_lines = ["name | grams | category | intensity_kg_per_kg | footprint_kg"]
    for r in rows[:20]:  # keep prompt small
        table_lines.append(
            f'{r["name"]} | {r["grams"]} | {r["category"]} | {r["intensity_kg_per_kg"]} | {r["footprint_kg"]}'
        )
    table = "\n".join(table_lines)
    seasons = "\n".join(season_lines[:25])

    system = (
        "You are KlimaCook, a nutritionist-chef focused on lower-carbon, healthy cooking.\n"
        "Given the ingredient footprint table and seasonality info, suggest 3–6 specific swaps or technique tweaks "
        "that lower CO₂e AND support good nutrition (e.g., fiber, unsaturated fats, lower sodium/sat fat). "
        "Prefer seasonal ingredients when proposing swaps.\n"
        "Output short markdown bullets, no preamble, no tables.\n"
        "Formatting rules:\n"
        "• When you mention an ingredient or a suggested swap that is **in season** add ' — in season'.\n"
        "• When it is **peak season** add ' — peak season'.\n"
        "• If out of season, add no tag.\n"
        "• If you estimate CO₂e savings, keep it brief (e.g., '~0.05 kg CO₂e')."
    )

    user = (
        f"Recipe: {rec.get('name','(unknown)')}\n"
        f"Estimated total footprint: {total} kg CO₂e\n\n"
        f"Top ingredients by footprint (approx):\n{table}\n\n"
        f"Seasonality (0=out,1=in,2=peak):\n{seasons}\n\n"
        f"Please propose concise, healthy, lower-CO₂e swaps and/or cooking tweaks for this recipe, "
        f"highlighting seasonality using the rules above."
    )

    try:
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_ver)
        resp = client.chat.completions.create(
            model=deploy,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"AI suggestion error: {e}"


def _shopping_list_lines(rec: RecipeOut) -> List[str]:
    lines: List[str] = []
    ings = rec.ingredients or []
    for i in ings:
        qty = f"{int(round(i.quantity_g))} g"
        lines.append(f"[ ] {i.name} — {qty}")
    return lines

def _build_pdf_bytes(title: str, subtitle: str, items: List[str]) -> bytes:
    """
    Try reportlab for a clean PDF; fall back to plain-text PDF if reportlab missing.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4

        y = height - 2.2 * cm
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, y, title)
        y -= 0.8 * cm

        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, y, subtitle)
        y -= 0.6 * cm

        c.setFont("Helvetica", 12)
        for line in items:
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 12)
            c.drawString(2 * cm, y, line)
            y -= 0.55 * cm

        c.showPage()
        c.save()
        return buf.getvalue()
    except Exception:
        # Fallback: very simple text "PDF" (still served as application/pdf)
        text = title + "\n" + subtitle + "\n\n" + "\n".join(items) + "\n"
        return text.encode("utf-8")

@app.get("/recipes/{recipe_name}/shopping-list.pdf")
def shopping_list_pdf(recipe_name: str):
    name_l = recipe_name.lower()
    rec = next((r for r in build_recipes() if r.name.lower() == name_l), None)
    if not rec:
        raise HTTPException(status_code=404, detail="Recipe not found")

    today = datetime.now().strftime("%Y-%m-%d")
    title = f"Shopping list — {rec.name}"
    subtitle = f"Generated {today} · KlimaCook"
    items = _shopping_list_lines(rec)
    pdf_bytes = _build_pdf_bytes(title, subtitle, items)

    filename = f"ShoppingList_{rec.name.replace(' ', '_')}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.post("/admin/reload-csv")
def admin_reload_csv():
    try:
        load_custom_intensities.cache_clear()
        ing_map, cat_map = load_custom_intensities()
        return {
            "ok": True,
            "ingredient_overrides": len(ing_map),
            "category_overrides": len(cat_map),
            "csv_path": os.getenv("CO2_CSV_PATH", "database.csv"),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ask")
def ask(payload: AskIn = Body(...)):
    return {"question": payload.question, "answer": generate_answer(payload.question)}


    # --- Background warmup so the first user never waits ---
@app.on_event("startup")
def _background_warmup():
    def _warm():
        try:
            # 1) build recipe list once (cached)
            build_recipes()
            # 2) make sure embeddings are built & FAISS is ready (cached)
            embs, _ = _ensure_embeddings()
            if _FAISS_OK:
                _ensure_faiss(embs)
            print("[KlimaCook] Warmup complete (recipes + embeddings).")
        except Exception as e:
            print("[KlimaCook] Warmup failed:", e)

    threading.Thread(target=_warm, daemon=True).start()