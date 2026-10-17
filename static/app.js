/**
 * app.js — Client-side logic for the Hybrid Multimodal Search Engine
 *
 * Features:
 *  - High-performance hybrid search (dense SigLIP 2 + sparse SPLADE)
 *  - Redis Two-Tier Cache: instant response (<2ms) with ⚡ Cached badge
 *  - Real-Time Trending Leaderboard: live queries from Redis Sorted Sets
 *  - Skeleton loading cards while fetching
 *  - Dynamic product card rendering from API JSON
 *  - Relevance score progress bars
 *  - Graceful error and empty states
 */

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const searchInput     = document.getElementById('search-input');
const searchBtn       = document.getElementById('search-btn');
const resultsGrid     = document.getElementById('results-grid');
const statusBar       = document.getElementById('status-bar');
const statusText      = document.getElementById('status-text');
const latencyBadge    = document.getElementById('latency-badge');
const cacheBadge      = document.getElementById('cache-badge');
const trendingSection = document.getElementById('trending-section');
const trendingPills   = document.getElementById('trending-pills');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentQuery = '';

// ---------------------------------------------------------------------------
// Search entry points
// Search only fires on button click, Enter key, or trending pill click.
// ---------------------------------------------------------------------------

/** Fires when the user clicks the search button */
searchBtn.addEventListener('click', () => triggerSearch());

/** Fires on Enter key in the search input */
searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') triggerSearch();
});

/** Clear results when the user empties the input */
searchInput.addEventListener('input', () => {
  if (!searchInput.value.trim()) showIntroState();
});

// ---------------------------------------------------------------------------
// Core search flow
// ---------------------------------------------------------------------------

async function triggerSearch() {
  const q = searchInput.value.trim();
  if (!q) return;
  if (q === currentQuery) return;  // Avoid duplicate calls for the same query
  currentQuery = q;

  showSkeleton(12);
  setLoading(true);

  try {
    const url = `/api/search?q=${encodeURIComponent(q)}&top_k=12`;
    const res = await fetch(url);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data, q);

    // Refresh trending queries leaderboard in the background
    loadTrending();

  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Trending Searches (Redis Sorted Set)
// ---------------------------------------------------------------------------

async function loadTrending() {
  if (!trendingSection || !trendingPills) return;

  try {
    const res = await fetch('/api/trending?n=5');
    if (!res.ok) {
      trendingSection.style.display = 'none';
      return;
    }

    const data = await res.json();
    const items = data.trending || [];

    if (items.length === 0) {
      trendingSection.style.display = 'none';
      return;
    }

    trendingPills.innerHTML = items.map(item => `
      <button class="trending-pill" type="button" data-query="${escHtml(item.query)}">
        <span class="pill-query">${escHtml(item.query)}</span>
        <span class="pill-count">${item.count}</span>
      </button>
    `).join('');

    trendingSection.style.display = 'flex';

    // Clicking a trending pill runs the search immediately
    trendingPills.querySelectorAll('.trending-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        const query = btn.getAttribute('data-query');
        if (query) {
          searchInput.value = query;
          currentQuery = ''; // Reset so triggerSearch executes
          triggerSearch();
        }
      });
    });

  } catch {
    // If Redis is offline, cleanly hide the trending section
    trendingSection.style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

function renderResults(data, query) {
  const { results, count, elapsed_ms, cache_hit } = data;

  // Update status bar with latency and cache hit status
  updateStatus(count, query, elapsed_ms, cache_hit);

  resultsGrid.innerHTML = '';

  if (count === 0) {
    showEmptyState(query);
    return;
  }

  const fragment = document.createDocumentFragment();
  results.forEach((product, i) => {
    const card = buildCard(product, i);
    fragment.appendChild(card);
  });
  resultsGrid.appendChild(fragment);
}

function buildCard(product, index) {
  const card = document.createElement('div');
  card.className = 'product-card';
  card.style.animationDelay = `${index * 40}ms`;

  // Score as a percentage for the bar (RRF scores are 0–1)
  const scorePercent = Math.min(100, Math.round(product.score * 100));

  // Build attribute chips
  const attrs = [product.color, product.material, product.category]
    .filter(Boolean)
    .map(a => `<span class="attr-chip">${escHtml(a)}</span>`)
    .join('');

  // Image — show emoji fallback on error
  const imgHtml = product.image_url
    ? `<img
          class="card-image"
          src="${escHtml(product.image_url)}"
          alt="${escHtml(product.title)}"
          loading="lazy"
          onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
       />
       <div class="card-image-error" style="display:none;">📦</div>`
    : `<div class="card-image-error">📦</div>`;

  card.innerHTML = `
    <div class="card-image-wrap">
      ${imgHtml}
      <div class="score-badge">#${index + 1}</div>
      ${product.is_new ? `<div class="new-badge" title="Added ${new Date(product.ingested_at * 1000).toLocaleString()}">NEW</div>` : ''}
    </div>
    <div class="card-body">
      <div class="card-meta">
        ${product.brand ? `<span class="brand-badge" title="${escHtml(product.brand)}">${escHtml(product.brand)}</span>` : ''}
        ${product.product_type ? `<span class="type-badge">${escHtml(product.product_type)}</span>` : ''}
      </div>
      <p class="card-title">${escHtml(product.title)}</p>
      ${attrs ? `<div class="card-attributes">${attrs}</div>` : ''}
      <div class="score-bar-wrap">
        <span class="score-bar-label">Relevance</span>
        <div class="score-bar-track">
          <div class="score-bar-fill" style="width: ${scorePercent}%"></div>
        </div>
        <span class="score-value">${product.score.toFixed(3)}</span>
      </div>
    </div>
  `;

  return card;
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function updateStatus(count, query, elapsed_ms, cache_hit = false) {
  if (count === 0) {
    statusText.innerHTML = `No results for <strong>"${escHtml(query)}"</strong>`;
  } else {
    statusText.innerHTML = `<strong>${count}</strong> result${count !== 1 ? 's' : ''} for <strong>"${escHtml(query)}"</strong>`;
  }

  // Redis Cache indicator
  if (cacheBadge) {
    cacheBadge.style.display = cache_hit ? 'inline-flex' : 'none';
  }

  latencyBadge.textContent = `${elapsed_ms} ms`;
  statusBar.classList.add('visible');
}

// ---------------------------------------------------------------------------
// UI States
// ---------------------------------------------------------------------------

function showSkeleton(count) {
  statusBar.classList.remove('visible');
  if (cacheBadge) cacheBadge.style.display = 'none';
  resultsGrid.innerHTML = Array.from({ length: count }, () => `
    <div class="skeleton-card">
      <div class="skeleton-img"></div>
      <div class="skeleton-body">
        <div class="skeleton-line short"></div>
        <div class="skeleton-line long"></div>
        <div class="skeleton-line long"></div>
        <div class="skeleton-line short"></div>
      </div>
    </div>
  `).join('');
}

function showIntroState() {
  currentQuery = '';
  statusBar.classList.remove('visible');
  if (cacheBadge) cacheBadge.style.display = 'none';
  resultsGrid.innerHTML = `
    <div class="state-container state-intro">
      <div class="state-icon">🔍</div>
      <p class="state-title">Hybrid Semantic Search</p>
      <p class="state-sub">
        Type a query above to search across 56,427 Amazon products using
        fine-tuned SigLIP 2 dense embeddings combined with SPLADE keyword matching.
      </p>
    </div>
  `;
}

function showEmptyState(query) {
  if (cacheBadge) cacheBadge.style.display = 'none';
  resultsGrid.innerHTML = `
    <div class="state-container state-empty">
      <div class="state-icon">🕵️</div>
      <p class="state-title">No results found</p>
      <p class="state-sub">
        No products matched <strong>"${escHtml(query)}"</strong>.
        Try a broader query like "blue chair" or "running shoes".
      </p>
    </div>
  `;
}

function showError(message) {
  statusBar.classList.remove('visible');
  if (cacheBadge) cacheBadge.style.display = 'none';
  resultsGrid.innerHTML = `
    <div class="state-container state-error">
      <div class="state-icon">⚠️</div>
      <p class="state-title">Search Failed</p>
      <p class="state-sub">${escHtml(message)}</p>
    </div>
  `;
}

function setLoading(isLoading) {
  searchBtn.classList.toggle('loading', isLoading);
  searchBtn.disabled = isLoading;
  searchBtn.querySelector('.btn-text').textContent = isLoading ? 'Searching…' : 'Search';
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/** Safely escapes HTML to prevent XSS from API data */
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
showIntroState();
loadTrending();
searchInput.focus();
