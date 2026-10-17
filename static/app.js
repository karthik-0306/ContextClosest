/**
 * app.js — ContextClosest web UI
 *
 *  - Hybrid search via GET /api/search (SigLIP 2 dense + SPLADE sparse, RRF in Qdrant)
 *  - Query trace: shows whether Redis served the result or the full model path ran
 *  - Trending searches from Redis (GET /api/trending)
 *  - "Embedding space" canvas: an illustration of a query landing among its
 *    nearest neighbours, redrawn on every search
 *  - Light/dark theme toggle (remembered per browser)
 *  - Shareable searches: /?q=... runs that search; each search updates the URL
 */

// ── DOM ─────────────────────────────────────────────────────────────────────
const form          = document.getElementById('search-form');
const input         = document.getElementById('search-input');
const button        = document.getElementById('search-btn');
const grid          = document.getElementById('results-grid');
const trace         = document.getElementById('trace');
const trending      = document.getElementById('trending');
const trendingPills = document.getElementById('trending-pills');
const fieldK        = document.getElementById('field-k');
const fieldCaption  = document.getElementById('field-caption');
const themeToggle   = document.getElementById('theme-toggle');

const EXAMPLES = ['sunglasses for men', 'white sneakers for girls', 'wooden coffee table', 'black running shoes'];
const TOP_K = 12;
let lastQuery = '';

// ── Search ──────────────────────────────────────────────────────────────────
form.addEventListener('submit', (e) => {
  e.preventDefault();
  runSearch(input.value);
});

input.addEventListener('input', () => {
  if (!input.value.trim()) showIntro();
});

document.addEventListener('keydown', (e) => {
  const typing = ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  if (e.key === '/' && !typing) {
    e.preventDefault();
    input.focus();
    input.select();
  } else if (e.key === 'Escape' && document.activeElement === input) {
    input.value = '';
    showIntro();
  }
});

document.addEventListener('click', (e) => {
  const chip = e.target.closest('[data-query], .example');
  if (!chip) return;
  const q = chip.dataset.query || chip.textContent.trim();
  input.value = q;
  runSearch(q, { force: true });
});

// Broken product image -> neutral placeholder (one listener for the whole grid)
grid.addEventListener('error', (e) => {
  const img = e.target;
  if (img.tagName !== 'IMG') return;
  const ph = document.createElement('div');
  ph.className = 'card-photo-fallback';
  ph.innerHTML = FALLBACK_ICON;
  img.replaceWith(ph);
}, true);

async function runSearch(raw, { force = false } = {}) {
  const q = raw.trim();
  if (!q || (q === lastQuery && !force)) return;
  lastQuery = q;
  history.replaceState(null, '', `?q=${encodeURIComponent(q)}`);

  setLoading(true);
  showSkeleton();
  field.setQuery(q);

  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&top_k=${TOP_K}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `The server answered ${res.status}.`);
    }
    const data = await res.json();
    if (q !== lastQuery) return;           // a newer search has started
    renderResults(data, q);
    field.showNeighbours(data.count, data.cache_hit);
    loadTrending();
  } catch (err) {
    showError(err.message || 'Could not reach the search service.');
    field.showNeighbours(0, false);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  button.disabled = on;
  button.classList.toggle('loading', on);
  button.querySelector('.btn-text').textContent = on ? 'Searching' : 'Search';
}

// ── Rendering ───────────────────────────────────────────────────────────────
function renderResults(data, q) {
  const { results = [], count = 0, elapsed_ms = 0, cache_hit = false } = data;
  renderTrace(q, count, elapsed_ms, cache_hit);

  if (!count) {
    grid.innerHTML = stateHtml('state-empty', 'Nothing close enough',
      `No product matched <strong>“${esc(q)}”</strong> above the relevance threshold. Try a broader description.`,
      true);
    return;
  }

  const top = Math.max(...results.map((r) => r.score || 0)) || 1;
  grid.innerHTML = results.map((p, i) => cardHtml(p, i, top)).join('');
}

function renderTrace(q, count, ms, hit) {
  const path = hit
    ? `<span class="trace-step hit">Redis · cache hit</span>`
    : `<span class="trace-step miss">Redis · miss</span><span class="trace-arrow">→</span>
       <span class="trace-step">SigLIP 2 + SPLADE</span><span class="trace-arrow">→</span>
       <span class="trace-step">Qdrant · RRF</span>`;
  trace.innerHTML = `
    <span class="trace-summary"><strong>${count}</strong> result${count === 1 ? '' : 's'} for <strong>“${esc(q)}”</strong></span>
    <span class="trace-path" aria-label="How this result was served">${path}</span>
    <span class="trace-time">${formatMs(ms)}</span>`;
  trace.hidden = false;
}

function cardHtml(p, i, top) {
  const rel = Math.max(0.04, Math.min(1, (p.score || 0) / top));
  const attrs = [p.color, p.material, p.category]
    .filter(Boolean)
    .slice(0, 3)
    .map((a) => `<span class="attr" title="${esc(a)}">${esc(a)}</span>`)
    .join('');
  const added = p.is_new && p.ingested_at ? new Date(p.ingested_at * 1000).toLocaleDateString() : '';
  const photo = p.image_url
    ? `<img src="${esc(p.image_url)}" alt="${esc(p.title)}" loading="lazy" />`
    : `<div class="card-photo-fallback">${FALLBACK_ICON}</div>`;

  return `
    <article class="card" style="animation-delay:${i * 35}ms">
      <div class="card-photo">
        ${photo}
        <span class="card-rank">${String(i + 1).padStart(2, '0')}</span>
        ${p.is_new ? `<span class="card-new" title="Added ${esc(added)} through the streaming pipeline">NEW</span>` : ''}
      </div>
      <div class="card-body">
        <div class="card-meta">
          <span class="card-brand" title="${esc(p.brand)}">${esc(p.brand || 'Unbranded')}</span>
          ${p.product_type ? `<span class="card-type">${esc(p.product_type.replace(/_/g, ' '))}</span>` : ''}
        </div>
        <h3 class="card-title" title="${esc(p.title)}">${esc(p.title)}</h3>
        ${attrs ? `<div class="card-attrs">${attrs}</div>` : ''}
        <div class="proximity" title="Fused relevance score, relative to the top result">
          <span>match</span>
          <span class="proximity-track"><span class="proximity-fill" style="width:${(rel * 100).toFixed(1)}%"></span></span>
          <span class="proximity-score">${(p.score || 0).toFixed(3)}</span>
        </div>
      </div>
    </article>`;
}

const FALLBACK_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 8 12 3 3 8v8l9 5 9-5V8Z"/><path d="m3 8 9 5 9-5M12 13v8"/></svg>';

function showSkeleton() {
  trace.hidden = true;
  grid.innerHTML = Array.from({ length: 8 }, () => `
    <div class="skeleton" aria-hidden="true">
      <div class="skeleton-photo"></div>
      <div class="skeleton-body">
        <div class="skeleton-line w40"></div>
        <div class="skeleton-line w90"></div>
        <div class="skeleton-line w70"></div>
      </div>
    </div>`).join('');
}

function showIntro() {
  lastQuery = '';
  if (location.search) history.replaceState(null, '', location.pathname);
  trace.hidden = true;
  grid.innerHTML = stateHtml('state-intro', 'Results appear here',
    'Describe a product in plain words: colour, material, style or use. Matches are ranked by how close they are to your meaning.');
  field.reset();
}

function showError(message) {
  trace.hidden = true;
  grid.innerHTML = stateHtml('state-error', 'The search did not go through',
    `${esc(message)} If the site was idle, the first search wakes the server and can take up to a minute; please try again.`);
}

function stateHtml(cls, title, body, withExamples = false) {
  const chips = withExamples
    ? `<div class="examples">${EXAMPLES.map((q) => `<button class="chip example" type="button">${esc(q)}</button>`).join('')}</div>`
    : '';
  return `<div class="state ${cls}"><p class="state-title">${title}</p><p class="state-sub">${body}</p>${chips}</div>`;
}

// ── Trending (Redis) ────────────────────────────────────────────────────────
async function loadTrending() {
  try {
    const res = await fetch('/api/trending?n=5');
    if (!res.ok) throw new Error();
    const items = (await res.json()).trending || [];
    if (!items.length) { trending.hidden = true; return; }
    trendingPills.innerHTML = items.map((t) => `
      <button class="chip" type="button" role="listitem" data-query="${esc(t.query)}">
        ${esc(t.query)}<span class="chip-count">${t.count}</span>
      </button>`).join('');
    trending.hidden = false;
  } catch {
    trending.hidden = true;               // Redis offline: hide quietly
  }
}

// ── Embedding-space illustration ────────────────────────────────────────────
const field = (() => {
  const canvas = document.getElementById('field');
  const ctx = canvas.getContext('2d');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let points = [];
  let centers = [];
  let query = null;          // {x, y}
  let neighbours = [];       // indices into points, nearest first
  let shown = 0;             // how many neighbour lines are visible (animated)
  let anim = 0;
  let pulse = 0;
  let searching = false;

  // Deterministic layout: the same "space" on every visit.
  function rng(seed) {
    return () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  }
  function gauss(r) {
    return Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r());
  }
  (function build() {
    const r = rng(20251017);
    for (let c = 0; c < 9; c++) {
      const cx = 0.1 + r() * 0.8, cy = 0.12 + r() * 0.76, spread = 0.07 + r() * 0.05;
      centers.push({ x: cx, y: cy });
      const n = 16 + Math.floor(r() * 12);
      for (let i = 0; i < n; i++) {
        points.push({
          x: Math.min(0.97, Math.max(0.03, cx + gauss(r) * spread)),
          y: Math.min(0.96, Math.max(0.04, cy + gauss(r) * spread)),
          s: 1.6 + r() * 1.4,
        });
      }
    }
    for (let i = 0; i < 40; i++) points.push({ x: 0.04 + r() * 0.92, y: 0.05 + r() * 0.9, s: 1.3 + r() });
  })();

  function hash(str) {
    let h = 2166136261;
    for (const ch of str.toLowerCase()) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
    return h >>> 0;
  }

  function color(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function size() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function draw() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const pad = 18;
    const X = (v) => pad + v * (w - 2 * pad);
    const Y = (v) => pad + v * (h - 2 * pad);
    const ink = color('--ink-2'), cobalt = color('--cobalt'), saffron = color('--saffron');
    const near = new Set(neighbours.slice(0, Math.ceil(shown)));

    ctx.clearRect(0, 0, w, h);

    // neighbour links
    if (query) {
      ctx.lineWidth = 1.3;
      neighbours.slice(0, Math.ceil(shown)).forEach((idx, k) => {
        const p = points[idx];
        const t = Math.min(1, shown - k);
        ctx.strokeStyle = cobalt;
        ctx.globalAlpha = 0.25 + 0.5 * (1 - k / Math.max(1, neighbours.length));
        ctx.beginPath();
        ctx.moveTo(X(query.x), Y(query.y));
        ctx.lineTo(X(query.x + (p.x - query.x) * t), Y(query.y + (p.y - query.y) * t));
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
    }

    // products
    points.forEach((p, i) => {
      const isNear = near.has(i);
      ctx.fillStyle = isNear ? cobalt : ink;
      ctx.globalAlpha = isNear ? 1 : 0.5;
      ctx.beginPath();
      ctx.arc(X(p.x), Y(p.y), isNear ? p.s + 1.6 : p.s, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;

    // query point
    if (query) {
      const qx = X(query.x), qy = Y(query.y);
      const ring = 9 + (reduceMotion ? 0 : Math.sin(pulse) * 2.5);
      ctx.strokeStyle = saffron;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.55;
      ctx.beginPath(); ctx.arc(qx, qy, ring + 4, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = saffron;
      ctx.beginPath(); ctx.arc(qx, qy, 6.5, 0, Math.PI * 2); ctx.fill();
    }
  }

  function loop() {
    cancelAnimationFrame(anim);
    if (reduceMotion) { shown = neighbours.length; draw(); return; }
    const step = () => {
      pulse += 0.06;
      if (shown < neighbours.length) shown = Math.min(neighbours.length, shown + 0.35);
      draw();
      if (query && (searching || shown < neighbours.length)) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  function setQuery(text) {
    const h = hash(text);
    const c = centers[h % centers.length];
    query = {
      x: Math.min(0.94, Math.max(0.06, c.x + (((h >> 8) & 255) / 255 - 0.5) * 0.12)),
      y: Math.min(0.92, Math.max(0.08, c.y + (((h >> 16) & 255) / 255 - 0.5) * 0.12)),
    };
    neighbours = [];
    shown = 0;
    searching = true;
    fieldK.textContent = 'searching…';
    fieldCaption.innerHTML = `Query <strong>“${esc(text)}”</strong> is being encoded…`;
    loop();
  }

  function showNeighbours(k, cached) {
    if (!query) return;
    searching = false;
    const order = points
      .map((p, i) => ({ i, d: (p.x - query.x) ** 2 + (p.y - query.y) ** 2 }))
      .sort((a, b) => a.d - b.d);
    neighbours = order.slice(0, Math.max(0, Math.min(k, TOP_K))).map((o) => o.i);
    shown = cached ? neighbours.length : 0;
    fieldK.textContent = `k = ${neighbours.length}`;
    fieldCaption.innerHTML = k
      ? `The <strong>${neighbours.length} nearest</strong> products to your query${cached ? ', served from the Redis cache' : ''}.`
      : 'No product was close enough to this query.';
    loop();
  }

  function reset() {
    query = null;
    searching = false;
    neighbours = [];
    shown = 0;
    cancelAnimationFrame(anim);
    fieldK.textContent = '56,427 products';
    fieldCaption.textContent = 'Each dot is a product. A query lands among the products closest to its meaning.';
    draw();
  }

  window.addEventListener('resize', size);
  new ResizeObserver(size).observe(canvas);
  return { setQuery, showNeighbours, reset, redraw: draw };
})();

// ── Theme ───────────────────────────────────────────────────────────────────
themeToggle.addEventListener('click', () => {
  const root = document.documentElement;
  const isDark = root.dataset.theme
    ? root.dataset.theme === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = isDark ? 'light' : 'dark';
  try { localStorage.setItem('cc-theme', root.dataset.theme); } catch { /* ignore */ }
  field.redraw();
});

// ── Utilities ───────────────────────────────────────────────────────────────
function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatMs(ms) {
  if (ms < 1) return '< 1 ms';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

// ── Start ───────────────────────────────────────────────────────────────────
const initial = new URLSearchParams(location.search).get('q');
if (initial && initial.trim()) {
  input.value = initial;
  runSearch(initial);
} else {
  showIntro();
}
loadTrending();
