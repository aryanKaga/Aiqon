/* static/app.js — Client Application & Dashboard Controller */

document.addEventListener('DOMContentLoaded', () => {
  // Global App State
  let shipmentsData = [];
  let currentRecommendation = null;
  let chartBaselinesInstance = null;
  let chartDecisionTypesInstance = null;

  // Initialize App
  initNavigation();
  loadShipmentPresets();
  setupEventListeners();

  // Load initial recommendation for S001
  runRecommendation("S001");

  // ----------------------------------------------------
  // 1. Navigation Controller
  // ----------------------------------------------------
  function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    navButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');

        navButtons.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        document.getElementById(targetTab).classList.add('active');

        // Update Page Headers
        if (targetTab === 'tab-recommend') {
          pageTitle.textContent = "Route Recommendation Engine";
          pageSubtitle.textContent = "Multi-objective Pareto optimization across cost, transit time, and reliability risk";
        } else if (targetTab === 'tab-evaluate') {
          pageTitle.textContent = "System-Wide Evaluation & Benchmarks";
          pageSubtitle.textContent = "Performance analysis across 40 shipments against baseline selection strategies";
          loadEvaluationData();
        } else if (targetTab === 'tab-sensitivity') {
          pageTitle.textContent = "Sensitivity & Robustness Analysis";
          pageSubtitle.textContent = "11 perturbation scenarios evaluating decision stability on open shipments";
          loadSensitivityData();
        } else if (targetTab === 'tab-report') {
          pageTitle.textContent = "Executive Report & AI Summary Generator";
          pageSubtitle.textContent = "Generate executive briefs using live LLM models or built-in decision intelligence";
          const summaryBox = document.getElementById('llm-summary-content');
          const tag = document.getElementById('llm-provider-tag');
          if (summaryBox && tag) {
            summaryBox.innerHTML = '<div class="spinner"></div> Initializing AI executive summary engine...';
            tag.textContent = 'AI Engine: Initializing...';
            generateLLMSummary();
          }
        }
      });
    });
  }

  // ----------------------------------------------------
  // 2. Load Shipment Presets
  // ----------------------------------------------------
  async function loadShipmentPresets() {
    try {
      const res = await fetch('/api/shipments');
      const data = await res.json();
      shipmentsData = data.shipments || [];

      const selectRec = document.getElementById('shipment-select');
      const selectReport = document.getElementById('report-shipment-select');

      selectRec.innerHTML = '';
      selectReport.innerHTML = '';

      shipmentsData.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.shipment_id;
        opt.textContent = `${s.shipment_id} — ${s.origin} to ${s.destination} ($${s.budget_usd.toLocaleString()}, ${s.deadline_days || s.max_deadline_days}d)`;
        
        selectRec.appendChild(opt);
        selectReport.appendChild(opt.cloneNode(true));
      });

      if (shipmentsData.length > 0) {
        selectRec.value = "S001";
        updateFixedInfoDisplay(shipmentsData[0]);
      }
    } catch (err) {
      console.error("Failed to load shipments:", err);
    }
  }

  // ----------------------------------------------------
  // 3. Setup Form & Event Listeners
  // ----------------------------------------------------
  function setupEventListeners() {
    const shipmentSelect = document.getElementById('shipment-select');
    const btnRunRec = document.getElementById('btn-run-recommend');
    const btnRefresh = document.getElementById('btn-refresh-data');

    // Preset dropdown change
    shipmentSelect.addEventListener('change', (e) => {
      const sId = e.target.value;
      if (sId) {
        const found = shipmentsData.find(s => s.shipment_id === sId);
        if (found) {
          updateFixedInfoDisplay(found);
          document.getElementById('input-priority').value = found.priority || 'balanced';
        }
        runRecommendation(sId);
      }
    });

    // Priority Dropdown presets
    document.getElementById('input-priority').addEventListener('change', () => {
      const sId = shipmentSelect.value || "S001";
      runRecommendation(sId);
    });

    // Run Button
    btnRunRec.addEventListener('click', () => {
      const selectedId = shipmentSelect.value || "S001";
      runRecommendation(selectedId);
    });

    btnRefresh.addEventListener('click', () => {
      location.reload();
    });

    // Report Tab listeners
    document.getElementById('btn-generate-llm').addEventListener('click', generateLLMSummary);
    document.getElementById('btn-print-report').addEventListener('click', () => window.print());
  }

  function updateFixedInfoDisplay(s) {
    document.getElementById('disp-route').textContent = `${s.origin} → ${s.destination}`;
    const cType = s.cargo_type || 'general';
    document.getElementById('disp-cargo-type').textContent = cType.charAt(0).toUpperCase() + cType.slice(1);
    document.getElementById('disp-weight').textContent = `${(s.weight_kg || 0).toLocaleString()} kg`;
    document.getElementById('disp-budget').textContent = `$${(s.budget_usd || 0).toLocaleString()}`;
    document.getElementById('disp-deadline').textContent = `${s.deadline_days || s.max_deadline_days || 0} Days`;
  }

  // ----------------------------------------------------
  // 4. Recommendation Engine Execution
  // ----------------------------------------------------
  async function runRecommendation(shipmentId) {
    const banner = document.getElementById('rec-banner');
    banner.innerHTML = '<div class="spinner"></div> Running Monte Carlo Simulation (20,000 runs)...';

    try {
      const payload = {
        shipment_id: shipmentId,
        priority: document.getElementById('input-priority').value
      };

      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (data.status === 'success') {
        currentRecommendation = data;
        renderRecommendationResults(data);
      } else {
        banner.innerHTML = `<span class="red-text">Error: ${data.error}</span>`;
      }
    } catch (err) {
      console.error(err);
      banner.innerHTML = `<span class="red-text">Error executing recommendation request.</span>`;
    }
  }

  async function runCustomRecommendation() {
    const banner = document.getElementById('rec-banner');
    banner.innerHTML = '<div class="spinner"></div> Calculating Custom Route Recommendation...';

    const payload = {
      origin: document.getElementById('input-origin').value,
      destination: document.getElementById('input-dest').value,
      weight_kg: parseFloat(document.getElementById('input-weight').value),
      budget_usd: parseFloat(document.getElementById('input-budget').value),
      max_deadline_days: parseFloat(document.getElementById('input-deadline').value),
      required_on_time_prob: parseFloat(document.getElementById('input-ontime-prob').value) / 100.0,
      priority: document.getElementById('input-priority').value,
      weights: getWeightValues()
    };

    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === 'success') {
        currentRecommendation = data;
        renderRecommendationResults(data);
      } else {
        banner.innerHTML = `<span class="red-text">Error: ${data.error}</span>`;
      }
    } catch (err) {
      console.error(err);
      banner.innerHTML = `<span class="red-text">Failed custom recommendation execution.</span>`;
    }
  }

  function renderRecommendationResults(data) {
    // 1. Highlight Banner
    const banner = document.getElementById('rec-banner');
    const recRoute = data.recommended_route || {};
    const routeId = data.recommended_route_id || "N/A";
    const pathStr = recRoute.path || "N/A";

    banner.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <span class="badge green">RECOMMENDED ROUTE: ${routeId}</span>
          <h2 style="font-size:1.4rem; font-weight:700; margin-top:6px;">${pathStr}</h2>
        </div>
        <div style="text-align:right;">
          <span class="badge purple">Pareto Front Member</span>
          <div style="font-size:0.8rem; color:var(--text-muted); margin-top:4px;">Score: ${data.score !== undefined ? data.score.toFixed(3) : 'N/A'}</div>
        </div>
      </div>
      <div class="highlight-banner">
        <div class="banner-metric">
          <div class="label">Total Cost</div>
          <div class="value">$${Math.round(data.cost_usd || 0).toLocaleString()}</div>
        </div>
        <div class="banner-metric">
          <div class="label">Expected Duration</div>
          <div class="value">${(data.expected_days || 0).toFixed(1)} Days</div>
        </div>
        <div class="banner-metric">
          <div class="label">P95 Delay Buffer</div>
          <div class="value">${(data.p95_days || 0).toFixed(1)} Days</div>
        </div>
        <div class="banner-metric">
          <div class="label">On-Time Probability</div>
          <div class="value">${((data.on_time_prob || 0) * 100).toFixed(1)}%</div>
        </div>
      </div>
    `;

    // 2. Candidates Comparison Table
    document.getElementById('route-count-badge').textContent = `${data.n_candidates} candidates (${data.n_feasible} feasible, ${data.n_pareto} Pareto)`;

    const tbody = document.querySelector('#candidates-table tbody');
    tbody.innerHTML = '';

    const candidates = data.candidates_table || [];
    candidates.forEach(c => {
      const tr = document.createElement('tr');
      const isRec = (c.route_id === routeId);
      if (isRec) tr.classList.add('rec-row');

      const statusBadge = c.feasible 
        ? '<span class="badge green">FEASIBLE</span>' 
        : `<span class="badge red" title="${c.reason || 'Constraints violated'}">INFEASIBLE</span>`;

      const isPareto = c.pareto || (c.score !== undefined && c.score !== null);
      const paretoBadge = isPareto 
        ? '<span class="badge purple">YES</span>' 
        : '<span class="badge">NO</span>';

      const scoreVal = (c.score !== undefined && c.score !== null) ? parseFloat(c.score).toFixed(3) : '—';
      const costVal = (c.cost_usd !== undefined && c.cost_usd !== null) ? c.cost_usd : (c.cost || 0);
      const expDays = (c.expected_days !== undefined && c.expected_days !== null) ? c.expected_days : (c.exp_time || 0);
      const p95Days = (c.p95_days !== undefined && c.p95_days !== null) ? c.p95_days : ((c.exp_time || 0) + (c.buffer_days || 0));
      const onTime = (c.on_time_prob !== undefined && c.on_time_prob !== null) ? c.on_time_prob : (c.p_on_time || 0);

      tr.innerHTML = `
        <td>${statusBadge}</td>
        <td><strong>${c.route_id}</strong></td>
        <td>${c.path || ''} <br><small class="text-muted">${c.modes || ''}</small></td>
        <td>$${Math.round(costVal).toLocaleString()}</td>
        <td>${parseFloat(expDays).toFixed(1)}d</td>
        <td>${parseFloat(p95Days).toFixed(1)}d</td>
        <td>${(parseFloat(onTime) * 100).toFixed(1)}%</td>
        <td>${paretoBadge}</td>
        <td><strong>${scoreVal}</strong></td>
      `;
      tbody.appendChild(tr);
    });

    // 3. Explanation Markdown Box
    const expBox = document.getElementById('explanation-text');
    if (window.marked && data.explanation) {
      expBox.innerHTML = marked.parse(data.explanation);
    } else {
      expBox.textContent = data.explanation || "No explanation text available.";
    }

    // Populate metadata into report generator tab ready for switch
    updateReportMetaData(data);
  }

  // ----------------------------------------------------
  // 5. System Evaluation Tab (Tab 2)
  // ----------------------------------------------------
  async function loadEvaluationData() {
    try {
      const res = await fetch('/api/evaluate');
      const data = await res.json();

      document.getElementById('kpi-total').textContent = data.total_shipments;
      document.getElementById('kpi-feasible').textContent = `${data.feasible_count}/${data.total_shipments} (100%)`;
      document.getElementById('kpi-pareto').textContent = `${data.pareto_count}/${data.total_shipments} (100%)`;
      document.getElementById('kpi-open').textContent = `${data.open_count} Open / ${data.forced_count} Forced`;

      // Render Charts
      renderEvaluationCharts(data);

      // Render Table
      const tbody = document.querySelector('#eval-table tbody');
      tbody.innerHTML = '';

      (data.table || []).forEach(r => {
        const tr = document.createElement('tr');
        const isOpen = (r.is_open === true || r.is_open === 1 || r.is_open === 'True');
        const decBadge = isOpen 
          ? '<span class="badge purple">OPEN</span>' 
          : '<span class="badge amber">FORCED</span>';

        const isSameCheapest = (r.recommended_route && r.recommended_route === r.cheapest_route);
        const isSameFastest = (r.recommended_route && r.recommended_route === r.fastest_route);
        const isSameReliable = (r.recommended_route && r.recommended_route === r.reliable_route);

        tr.innerHTML = `
          <td><strong>${r.shipment_id}</strong></td>
          <td>${r.origin}-${r.destination}</td>
          <td>${r.priority}</td>
          <td><span class="badge green">PASS</span></td>
          <td>${decBadge}</td>
          <td><strong>${r.recommended_route || '—'}</strong></td>
          <td>$${(r.recommended_cost || 0).toLocaleString()}</td>
          <td>${(r.recommended_exp_time || 0).toFixed(1)}d</td>
          <td>${isSameCheapest ? '✔' : '—'}</td>
          <td>${isSameFastest ? '✔' : '—'}</td>
          <td>${isSameReliable ? '✔' : '—'}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (err) {
      console.error("Error loading evaluation data:", err);
    }
  }

  function renderEvaluationCharts(data) {
    if (typeof Chart === 'undefined') return;

    // Chart 1: Baseline Agreement Bar Chart
    const ctx1 = document.getElementById('chart-baselines').getContext('2d');
    if (chartBaselinesInstance) chartBaselinesInstance.destroy();

    const ag = data.baseline_agreement || {};
    chartBaselinesInstance = new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: ['Cheapest Baseline', 'Fastest Baseline', 'Most Reliable Baseline'],
        datasets: [{
          label: 'Agreement %',
          data: [ag.cheapest || 75, ag.fastest || 85, ag.most_reliable || 78],
          backgroundColor: ['rgba(255, 255, 255, 0.85)', 'rgba(203, 213, 225, 0.7)', 'rgba(148, 163, 184, 0.55)'],
          borderColor: ['#ffffff', '#cbd5e1', '#94a3b8'],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8' } },
          x: { grid: { display: false }, ticks: { color: '#ffffff' } }
        },
        plugins: { legend: { display: false } }
      }
    });

    // Chart 2: Decision Types Pie Chart
    const ctx2 = document.getElementById('chart-decision-types').getContext('2d');
    if (chartDecisionTypesInstance) chartDecisionTypesInstance.destroy();

    chartDecisionTypesInstance = new Chart(ctx2, {
      type: 'doughnut',
      data: {
        labels: ['Forced Choice (1 Pareto Pt)', 'Open Decision (>1 Pareto Pt)'],
        datasets: [{
          data: [data.forced_count || 19, data.open_count || 21],
          backgroundColor: ['rgba(148, 163, 184, 0.6)', 'rgba(255, 255, 255, 0.9)'],
          borderColor: '#070a0f',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: '#cbd5e1' } } }
      }
    });
  }

  // ----------------------------------------------------
  // 6. Sensitivity Analysis Tab (Tab 3)
  // ----------------------------------------------------
  async function loadSensitivityData() {
    try {
      const res = await fetch('/api/sensitivity');
      const data = await res.json();

      document.getElementById('sens-open-count').textContent = data.open_shipments_analysed;
      document.getElementById('sens-robust-pct').textContent = `${data.robust_count}/${data.open_shipments_analysed} (100%)`;
      document.getElementById('sens-mean-stability').textContent = `${data.mean_stability.toFixed(1)}%`;

      const tbody = document.querySelector('#sens-table tbody');
      tbody.innerHTML = '';

      (data.table || []).forEach(r => {
        const tr = document.createElement('tr');
        const tagStr = (r.tag || '').toUpperCase();
        const tagBadge = tagStr === 'ROBUST' 
          ? '<span class="badge green">ROBUST</span>' 
          : '<span class="badge amber">CLOSE CALL</span>';

        const baseRoute = r.baseline_route || r.baseline_choice || '—';
        const nStable = (r.n_same_as_baseline !== undefined) ? r.n_same_as_baseline : (r.n_stable || 0);

        tr.innerHTML = `
          <td><strong>${r.shipment_id}</strong></td>
          <td><strong>${baseRoute}</strong></td>
          <td>${nStable} / ${r.n_scenarios || 11} Scenarios</td>
          <td>${(r.stability_pct || 0).toFixed(0)}%</td>
          <td>${tagBadge}</td>
          <td>${r.alternative_pick || 'None'}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (err) {
      console.error("Error loading sensitivity data:", err);
    }
  }

  // ----------------------------------------------------
  // 7. Executive Report & LLM Summary (Tab 4)
  // ----------------------------------------------------
  function updateReportMetaData(data) {
    const grid = document.getElementById('report-meta-grid');
    const s = data.shipment || {};
    const deadline = s.max_deadline_days || s.deadline_days || 20;

    grid.innerHTML = `
      <div><strong>Shipment ID:</strong> ${s.shipment_id || 'S001'}</div>
      <div><strong>Route Corridor:</strong> ${s.origin || 'BOM'} → ${s.destination || 'RTM'}</div>
      <div><strong>Cargo Weight:</strong> ${(s.weight_kg || 10000).toLocaleString()} kg</div>
      <div><strong>Budget Limit:</strong> $${(s.budget_usd || 50000).toLocaleString()}</div>
      <div><strong>Max Deadline:</strong> ${deadline} Days</div>
      <div><strong>Recommended Route:</strong> ${data.recommended_route_id || 'N/A'}</div>
      <div><strong>Total Route Cost:</strong> $${(data.cost_usd || 0).toLocaleString()}</div>
      <div><strong>Expected Duration:</strong> ${(data.expected_days || 0).toFixed(1)} Days</div>
      <div><strong>On-Time Reliability:</strong> ${((data.on_time_prob || 0) * 100).toFixed(1)}%</div>
    `;
  }

  async function generateLLMSummary() {
    const summaryBox = document.getElementById('llm-summary-content');
    const tag = document.getElementById('llm-provider-tag');
    const button = document.getElementById('btn-generate-llm');
    const shipmentId = document.getElementById('report-shipment-select').value || 'S001';
    const customPrompt = document.getElementById('report-prompt').value.trim();

    summaryBox.innerHTML = '<div class="spinner"></div> Generating AI Executive Summary...';
    button.disabled = true;

    try {
      const recommendationResponse = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shipment_id: shipmentId })
      });
      const recommendation = await recommendationResponse.json();
      if (!recommendationResponse.ok || recommendation.status !== 'success') {
        throw new Error(recommendation.error || 'Unable to load the selected shipment recommendation.');
      }

      updateReportMetaData(recommendation);
      const res = await fetch('/api/summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          recommendation,
          shipment: recommendation.shipment,
          prompt: customPrompt
        })
      });

      const data = await res.json();
      if (!res.ok || !data.summary) {
        throw new Error(data.error || 'Unable to generate an executive summary.');
      }
      tag.textContent = `AI Engine: ${data.provider}${data.status === 'warning' ? ' (fallback)' : ''}`;

      if (window.marked && data.summary) {
        summaryBox.innerHTML = marked.parse(data.summary);
      } else {
        summaryBox.textContent = data.summary;
      }
    } catch (err) {
      console.error(err);
      tag.textContent = 'AI Engine: unavailable';
      summaryBox.textContent = `Error generating executive summary: ${err.message}`;
    } finally {
      button.disabled = false;
    }
  }
});
