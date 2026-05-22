/* ═══════════════════════════════════════════════════════════════════
   FakeGuard AI — Application Logic
   ═══════════════════════════════════════════════════════════════════ */

// ────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────
const API_BASE = window.location.origin.startsWith('file') || window.location.origin === 'null' || !window.location.hostname
  ? 'http://localhost:8000'
  : (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
      ? (window.location.port === '8000' ? window.location.origin : 'http://localhost:8000')
      : window.location.origin);

// ────────────────────────────────────────────────────────────────
// Sample Articles — realistic examples for quick testing
// ────────────────────────────────────────────────────────────────
const SAMPLES = [
  {
    title: 'BREAKING: Scientists Confirm 5G Towers Are Reprogramming Human DNA, Government Cover-Up Exposed',
    text: 'In a shocking revelation that mainstream media refuses to cover, a group of rogue scientists from an undisclosed European laboratory have confirmed that 5G cellular towers emit a specific frequency that can alter human DNA sequences. The anonymous researchers claim to have documented over 10,000 cases of genetic mutation in people living within 500 meters of 5G installations. Government officials from twelve countries have allegedly signed a secret pact to suppress this information, according to documents leaked on an encrypted messaging platform. "This is the biggest cover-up since the moon landing," said one whistleblower who spoke on condition of anonymity. Health authorities have not responded to these allegations.',
    type: 'fake',
  },
  {
    title: 'Federal Reserve Raises Interest Rates by Quarter Point, Signals Cautious Approach to Future Cuts',
    text: 'The Federal Reserve raised its benchmark interest rate by 25 basis points on Wednesday, bringing the federal funds rate to a target range of 5.25% to 5.50%, in line with market expectations. Fed Chair Jerome Powell indicated during the post-meeting press conference that future policy decisions would remain data-dependent, emphasizing the committee\'s commitment to returning inflation to its 2% target. The central bank\'s updated economic projections showed GDP growth of 2.1% for the year, while the unemployment rate is expected to edge up to 4.1% by year-end. Treasury yields were little changed following the announcement, with the 10-year note holding steady at 4.18%. Economists from Goldman Sachs and JPMorgan noted that the decision reflects a balanced approach given persistent core inflation readings.',
    type: 'real',
  },
  {
    title: 'New Study Suggests Social Media Use May Be Linked to Rising Anxiety Among Teenagers',
    text: 'A new longitudinal study published in the Journal of Adolescent Health has found a correlation between heavy social media usage and increased anxiety symptoms in teenagers aged 13 to 17. The research, conducted over three years with a sample of 8,500 participants across multiple states, found that teens who spent more than four hours daily on social media platforms were 2.3 times more likely to report moderate to severe anxiety compared to their peers with limited usage. However, the researchers cautioned that the study demonstrates correlation rather than causation and that other contributing factors such as academic pressure and family dynamics were not fully controlled. Some psychologists have questioned the methodology, noting that self-reported screen time data can be unreliable and that the definition of anxiety symptoms used in the study was broader than standard clinical criteria.',
    type: 'mixed',
  },
];

// ────────────────────────────────────────────────────────────────
// DOM References (cached for performance)
// ────────────────────────────────────────────────────────────────
const dom = {
  navbar:           document.getElementById('navbar'),
  navToggle:        document.getElementById('nav-toggle'),
  navLinks:         document.getElementById('nav-links'),
  titleInput:       document.getElementById('news-title'),
  textInput:        document.getElementById('news-text'),
  charCount:        document.getElementById('char-count'),
  analyzeBtn:       document.getElementById('analyze-btn'),
  errorContainer:   document.getElementById('error-container'),
  errorMessage:     document.getElementById('error-message'),
  skeleton:         document.getElementById('results-skeleton'),
  resultsArea:      document.getElementById('results-area'),
  verdictCard:      document.getElementById('verdict-card'),
  verdictLabel:     document.getElementById('verdict-label'),
  confidenceText:   document.getElementById('verdict-confidence-text'),
  gaugeFill:        document.getElementById('gauge-fill'),
  gaugeValue:       document.getElementById('gauge-value'),
  probFakeBar:      document.getElementById('prob-fake-bar'),
  probRealBar:      document.getElementById('prob-real-bar'),
  probFakeLabel:    document.getElementById('prob-fake-label'),
  probRealLabel:    document.getElementById('prob-real-label'),
  tabShap:          document.getElementById('tab-shap'),
  tabLime:          document.getElementById('tab-lime'),
  tabIndicator:     document.getElementById('tab-indicator'),
  tabContentShap:   document.getElementById('tab-content-shap'),
  tabContentLime:   document.getElementById('tab-content-lime'),
  shapLoading:      document.getElementById('shap-loading'),
  limeLoading:      document.getElementById('lime-loading'),
  shapHighlights:   document.getElementById('shap-highlights'),
  limeHighlights:   document.getElementById('lime-highlights'),
  metricsNote:      document.getElementById('metrics-note'),
};

// ────────────────────────────────────────────────────────────────
// Global Cache & State
// ────────────────────────────────────────────────────────────────
let currentArticle = { title: '', text: '' };
let currentExplanations = { lime: null, shap: null };

// ────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52; // r = 52 in SVG

// ────────────────────────────────────────────────────────────────
// Initialization
// ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initCharCounter();
  loadMetrics();
});

// ────────────────────────────────────────────────────────────────
// NAVBAR — scroll effect + mobile toggle
// ────────────────────────────────────────────────────────────────
function initNavbar() {
  // Scroll effect
  window.addEventListener('scroll', () => {
    dom.navbar.classList.toggle('scrolled', window.scrollY > 40);
  }, { passive: true });

  // Mobile hamburger
  dom.navToggle.addEventListener('click', () => {
    dom.navToggle.classList.toggle('open');
    dom.navLinks.classList.toggle('open');
  });

  // Close mobile menu on link click
  dom.navLinks.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      dom.navToggle.classList.remove('open');
      dom.navLinks.classList.remove('open');
    });
  });
}

// ────────────────────────────────────────────────────────────────
// CHARACTER COUNTER
// ────────────────────────────────────────────────────────────────
function initCharCounter() {
  dom.textInput.addEventListener('input', () => {
    const len = dom.textInput.value.length;
    dom.charCount.textContent = `${len.toLocaleString()} character${len !== 1 ? 's' : ''}`;
  });
}

// ════════════════════════════════════════════════════════════════
// CORE ANALYSIS FLOW
// ════════════════════════════════════════════════════════════════

/**
 * Main entry point — validates input, calls /predict, then /explain
 * for both SHAP and LIME in parallel.
 */
async function analyzeNews() {
  const title = dom.titleInput.value.trim();
  const text  = dom.textInput.value.trim();

  // ── Validation ──
  if (!text) {
    showError('Please enter or paste news article text before analyzing.');
    dom.textInput.focus();
    return;
  }
  if (text.length < 20) {
    showError('The article text is too short. Please provide at least a few sentences for accurate analysis.');
    dom.textInput.focus();
    return;
  }

  // ── Cache current article text and reset explanation cache ──
  currentArticle.title = title;
  currentArticle.text = text;
  currentExplanations.lime = null;
  currentExplanations.shap = null;

  // ── UI: enter loading state ──
  hideError();
  hideResults();
  showLoading();

  try {
    // Step 1: Prediction
    const predictionData = await apiPost('/predict', { title, text });
    hideLoading();
    displayResults(predictionData);

    // Step 2: Explanations (LIME only, SHAP is lazy-loaded on tab click)
    showExplainLoading('lime');

    const explanationPayload = { title, text, num_features: 25, method: 'lime' };
    
    try {
      const limeData = await apiPost('/explain', explanationPayload);
      hideExplainLoading('lime');
      currentExplanations.lime = limeData;
      displayExplanation(limeData, 'lime');
    } catch (limeErr) {
      hideExplainLoading('lime');
      dom.limeHighlights.innerHTML = renderExplainError('LIME analysis unavailable.');
    }

    // Scroll to results
    dom.verdictCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

  } catch (err) {
    hideLoading();
    hideResults();
    handleApiError(err);
  }
}

// ════════════════════════════════════════════════════════════════
// API HELPERS
// ════════════════════════════════════════════════════════════════

/**
 * Generic POST request to the FastAPI backend.
 */
async function apiPost(endpoint, body) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`Server responded with ${res.status}: ${detail}`);
  }
  return res.json();
}

/**
 * Generic GET request.
 */
async function apiGet(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) throw new Error(`Server responded with ${res.status}`);
  return res.json();
}

/**
 * Handle various fetch errors with user-friendly messages.
 */
function handleApiError(err) {
  if (err instanceof TypeError && err.message.includes('fetch')) {
    showError(
      'Could not connect to the backend server. Make sure the API is running:\n' +
      '<code>python -m uvicorn src.api:app --reload</code>'
    );
  } else {
    showError(`Analysis failed: ${err.message}`);
  }
}

// ════════════════════════════════════════════════════════════════
// DISPLAY RESULTS
// ════════════════════════════════════════════════════════════════

/**
 * Render the verdict card, gauge, and probability bar from /predict response.
 */
function displayResults(data) {
  const { label, confidence, probabilities } = data;
  const isFake = label === 'FAKE';
  const pct = Math.round(confidence * 100);

  // Verdict label
  dom.verdictLabel.textContent = label;
  dom.verdictLabel.className = `verdict-label ${isFake ? 'fake' : 'real'}`;
  dom.confidenceText.textContent = `${pct}% confidence`;

  // Gauge
  animateGauge(confidence, isFake);

  // Probability bar
  const fakePct = Math.round((probabilities?.FAKE ?? (isFake ? confidence : 1 - confidence)) * 100);
  const realPct = 100 - fakePct;

  dom.probFakeBar.style.width = `${fakePct}%`;
  dom.probRealBar.style.width = `${realPct}%`;
  dom.probFakeLabel.textContent = `FAKE ${fakePct}%`;
  dom.probRealLabel.textContent = `REAL ${realPct}%`;

  // Hide labels if segment is too narrow
  dom.probFakeLabel.style.display = fakePct < 12 ? 'none' : '';
  dom.probRealLabel.style.display = realPct < 12 ? 'none' : '';

  // Reset explanation tabs to LIME
  switchTab('lime');
  dom.shapHighlights.innerHTML = '';
  dom.limeHighlights.innerHTML = '';

  showResults();
}

// ────────────────────────────────────────────────────────────────
// ANIMATED GAUGE
// ────────────────────────────────────────────────────────────────

/**
 * Animate the SVG circular gauge to the given confidence (0–1).
 */
function animateGauge(confidence, isFake) {
  const fill = dom.gaugeFill;

  // Reset to full dashoffset
  fill.style.transition = 'none';
  fill.style.strokeDasharray  = GAUGE_CIRCUMFERENCE;
  fill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;

  // Apply color class
  fill.className = `gauge-fill ${isFake ? 'fake-stroke' : 'real-stroke'}`;

  // Trigger reflow so the reset takes effect
  void fill.getBoundingClientRect();

  // Animate to target
  const targetOffset = GAUGE_CIRCUMFERENCE * (1 - confidence);
  fill.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
  fill.style.strokeDashoffset = targetOffset;

  // Counter animation for the percentage number
  animateCounter(dom.gaugeValue, 0, Math.round(confidence * 100), 1000);
}

/**
 * Smoothly animate a number from `start` to `end` inside `el`.
 */
function animateCounter(el, start, end, duration) {
  const startTime = performance.now();
  function tick(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (end - start) * eased);
    el.textContent = `${current}%`;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ════════════════════════════════════════════════════════════════
// EXPLAINABILITY
// ════════════════════════════════════════════════════════════════

/**
 * Render word-importance chips from /explain response.
 * Red chips → pushes toward FAKE, Green chips → pushes toward REAL.
 * Opacity is proportional to |weight|.
 */
function displayExplanation(data, method) {
  const container = method === 'shap' ? dom.shapHighlights : dom.limeHighlights;
  const explanations = data.explanations || [];

  if (!explanations.length) {
    container.innerHTML = renderExplainError('No explanation data returned.');
    return;
  }

  // Find max absolute weight for normalization
  const maxWeight = Math.max(...explanations.map(e => Math.abs(e.weight)), 0.001);

  const chips = explanations.map((item, i) => {
    const { token, weight } = item;
    const absWeight = Math.abs(weight);
    const normalized = absWeight / maxWeight; // 0–1

    // Determine direction
    let cls = 'neutral-word';
    if (weight < -0.001)      cls = 'fake-word';
    else if (weight > 0.001)  cls = 'real-word';

    // Opacity: minimum 0.12 so chips are always visible, max 0.55
    const chipOpacity = (0.12 + normalized * 0.43).toFixed(2);

    // Stagger animation delay
    const delay = (i * 0.03).toFixed(2);

    const weightLabel = weight >= 0
      ? `+${weight.toFixed(3)}`
      : weight.toFixed(3);

    return `<span class="word-chip ${cls}"
                  style="--chip-opacity:${chipOpacity}; animation-delay:${delay}s"
                  title="Weight: ${weightLabel}">
              ${escapeHtml(token)}<span class="chip-weight">${weightLabel}</span>
            </span>`;
  });

  container.innerHTML = chips.join('');
}

/**
 * Small helper to render an inline error inside the explanation area.
 */
function renderExplainError(msg) {
  return `<p style="color:var(--text-muted);font-size:var(--fs-sm);">${msg}</p>`;
}

// ════════════════════════════════════════════════════════════════
// TAB SWITCHING (SHAP / LIME)
// ════════════════════════════════════════════════════════════════

function switchTab(method) {
  const isShap = method === 'shap';

  dom.tabShap.classList.toggle('active', isShap);
  dom.tabLime.classList.toggle('active', !isShap);
  dom.tabContentShap.classList.toggle('active', isShap);
  dom.tabContentLime.classList.toggle('active', !isShap);
  dom.tabIndicator.classList.toggle('right', !isShap);

  // Lazy-load SHAP if clicked and not already computed
  if (isShap && !currentExplanations.shap && currentArticle.text) {
    showExplainLoading('shap');
    dom.shapHighlights.innerHTML = '';
    
    const explanationPayload = {
      title: currentArticle.title,
      text: currentArticle.text,
      num_features: 25,
      method: 'shap'
    };

    apiPost('/explain', explanationPayload)
      .then(shapData => {
        hideExplainLoading('shap');
        currentExplanations.shap = shapData;
        displayExplanation(shapData, 'shap');
      })
      .catch(err => {
        hideExplainLoading('shap');
        dom.shapHighlights.innerHTML = renderExplainError('SHAP analysis unavailable: ' + err.message);
      });
  }
}

// ════════════════════════════════════════════════════════════════
// SAMPLE ARTICLE LOADER
// ════════════════════════════════════════════════════════════════

function loadSample(index) {
  const sample = SAMPLES[index];
  if (!sample) return;

  dom.titleInput.value = sample.title;
  dom.textInput.value  = sample.text;

  // Update char count
  const len = sample.text.length;
  dom.charCount.textContent = `${len.toLocaleString()} character${len !== 1 ? 's' : ''}`;

  // Visual feedback — briefly highlight the active sample button
  document.querySelectorAll('.btn-sample').forEach(btn => btn.classList.remove('active'));
  const btn = document.getElementById(`sample-btn-${index}`);
  if (btn) {
    btn.style.background = 'rgba(102, 126, 234, 0.15)';
    btn.style.borderColor = 'rgba(102, 126, 234, 0.4)';
    setTimeout(() => {
      btn.style.background = '';
      btn.style.borderColor = '';
    }, 600);
  }

  // Scroll to the analyzer card
  dom.textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ════════════════════════════════════════════════════════════════
// METRICS LOADER
// ════════════════════════════════════════════════════════════════

async function loadMetrics() {
  try {
    const data = await apiGet('/metrics');

    setMetric('accuracy',  data.accuracy);
    setMetric('precision', data.precision);
    setMetric('recall',    data.recall);
    setMetric('f1',        data.f1);

    dom.metricsNote.textContent = 'Metrics fetched live from the model evaluation API.';
  } catch {
    dom.metricsNote.textContent = 'Could not load metrics — backend may be offline.';
    // Set placeholder dashes
    ['accuracy', 'precision', 'recall', 'f1'].forEach(key => {
      const valEl = document.getElementById(`metric-val-${key}`);
      if (valEl) valEl.textContent = '—';
    });
  }
}

/**
 * Set a single metric card's value and animate the progress bar.
 */
function setMetric(key, value) {
  const valEl = document.getElementById(`metric-val-${key}`);
  const barEl = document.getElementById(`metric-bar-${key}`);
  if (!valEl) return;

  const pct = (value * 100).toFixed(1);
  valEl.textContent = `${pct}%`;

  // Animate bar fill after a short delay
  if (barEl) {
    setTimeout(() => { barEl.style.width = `${pct}%`; }, 300);
  }
}

// ════════════════════════════════════════════════════════════════
// UI STATE HELPERS
// ════════════════════════════════════════════════════════════════

function showLoading() {
  dom.analyzeBtn.classList.add('loading');
  dom.analyzeBtn.disabled = true;
  dom.skeleton.classList.add('visible');
}

function hideLoading() {
  dom.analyzeBtn.classList.remove('loading');
  dom.analyzeBtn.disabled = false;
  dom.skeleton.classList.remove('visible');
}

function showResults() {
  dom.resultsArea.classList.add('visible');
}

function hideResults() {
  dom.resultsArea.classList.remove('visible');
}

function showError(msg) {
  dom.errorMessage.innerHTML = msg;
  dom.errorContainer.classList.add('visible');
}

function hideError() {
  dom.errorContainer.classList.remove('visible');
}

function dismissError() {
  hideError();
}

function showExplainLoading(method) {
  const el = method === 'shap' ? dom.shapLoading : dom.limeLoading;
  el.classList.add('visible');
}

function hideExplainLoading(method) {
  const el = method === 'shap' ? dom.shapLoading : dom.limeLoading;
  el.classList.remove('visible');
}

// ════════════════════════════════════════════════════════════════
// UTILITY
// ════════════════════════════════════════════════════════════════

/**
 * Escape HTML entities to prevent XSS in dynamically rendered tokens.
 */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}
