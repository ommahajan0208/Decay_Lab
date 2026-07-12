/* ═══════════════════════════════════════════════════════════
   DECAY LAB — NEUROSCIENCE SIMULATOR
   Client-Side Logic: All 12 Features
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    // ── State ────────────────────────────────────────────────
    let decayChart     = null;
    let simChart       = null;
    let banditChart    = null;
    let autoPlayTimer  = null;
    let banditWeightHistory = [];  // for bandit chart

    const COLORS = [
        '#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6',
        '#ec4899','#14b8a6','#f97316','#6366f1','#84cc16',
    ];

    // ── Tab Switching ─────────────────────────────────────────
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tab = btn.getAttribute('data-tab');
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
            document.getElementById(`tab-${tab}`).classList.remove('hidden');
            // Auto-refresh the selected tab's content
            if (tab === 'live')     { refreshAll(); }
            if (tab === 'race')     { refreshRace(); }
            if (tab === 'bandit')   { refreshBandit(); }
        });
    });

    // ── Chart Options Factory ──────────────────────────────────
    function makeChartOpts(yLabel = 'Strength', yMax = 1.05) {
        return {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 600, easing: 'easeOutQuart' },
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', labels: { color: '#94a3b8', usePointStyle: true, padding: 16,
                    font: { family: "'Plus Jakarta Sans', sans-serif", size: 11, weight: '600' } } },
                tooltip: {
                    backgroundColor: 'rgba(11, 15, 25, 0.95)', titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1', borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1,
                    padding: 12, boxPadding: 6, cornerRadius: 8,
                    titleFont: { family: "'Plus Jakarta Sans', sans-serif", weight: '700' },
                    bodyFont: { family: "'Space Grotesk', sans-serif" },
                    callbacks: { label: ctx => ` ${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}%` }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#7c8fa6', font: { family: "'Space Grotesk', sans-serif", size: 11 } } },
                y: {
                    beginAtZero: true, max: yMax,
                    grid: { color: 'rgba(255,255,255,0.03)', drawBorder: false },
                    ticks: { color: '#7c8fa6', callback: v => (v * 100).toFixed(0) + '%', font: { family: "'Space Grotesk', sans-serif", size: 11 } },
                    title: { display: true, text: yLabel, color: '#7c8fa6', font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: '600' } }
                }
            }
        };
    }

    // ── Event Feed ────────────────────────────────────────────
    function addEvent(msg, type = 'info') {
        const feed = document.getElementById('event-feed');
        const el = document.createElement('div');
        el.className = `event-item event-${type}`;
        el.textContent = msg;
        feed.insertBefore(el, feed.firstChild);
        // Keep max 50 events
        while (feed.children.length > 50) feed.removeChild(feed.lastChild);
    }

    // ── Simulated Clock ───────────────────────────────────────
    async function refreshClock() {
        const data = await api('/api/time');
        document.getElementById('sim-clock-display').textContent = data.display;
        document.getElementById('sim-hours-display').textContent = `(+${data.offset_hours.toFixed(1)}h)`;
    }

    // ── API Helper ────────────────────────────────────────────
    async function api(url, method = 'GET', body = null) {
        const opts = { method, headers: { 'Content-Type': 'application/json' } };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(url, opts);
        return res.json();
    }

    // ── Decay Chart with Ebbinghaus ───────────────────────────
    async function refreshDecayChart() {
        const data = await api('/api/series');
        const ctx = document.getElementById('decayChart').getContext('2d');
        const showEbb = document.getElementById('toggle-ebbinghaus').checked;

        const datasets = data.datasets.map((ds, i) => {
            const gradient = ctx.createLinearGradient(0, 0, 0, 350);
            gradient.addColorStop(0, COLORS[i % COLORS.length] + '33');
            gradient.addColorStop(1, 'rgba(11, 15, 25, 0)');
            return {
                ...ds,
                borderColor: COLORS[i % COLORS.length],
                backgroundColor: gradient,
                borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 6, fill: true, tension: 0.4,
            };
        });

        if (showEbb && data.ebbinghaus) {
            datasets.push({
                label: '📉 Ebbinghaus (1885)',
                data: data.ebbinghaus,
                borderColor: 'rgba(251,191,36,0.6)',
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: [8, 5],
                pointRadius: 0, fill: false,
            });
        }

        // Threshold line
        datasets.push({
            label: '⚠️ Prune Threshold',
            data: Array(data.labels.length).fill(0.15),
            borderColor: 'rgba(239,68,68,0.4)',
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0, fill: false,
        });

        if (decayChart) decayChart.destroy();
        decayChart = new Chart(ctx, {
            type: 'line',
            data: { labels: data.labels, datasets },
            options: makeChartOpts('Effective Strength'),
        });
    }

    // ── Health Grid ───────────────────────────────────────────
    async function refreshHealthGrid() {
        const data = await api('/api/memories');
        const grid = document.getElementById('health-grid');
        const badge = document.getElementById('memory-count-badge');
        const memories = data.memories || [];
        const alive = memories.filter(m => m.current_strength > 0.15).length;

        badge.textContent = `${alive}/${memories.length} alive`;
        grid.innerHTML = '';

        memories.forEach(m => {
            const pct = Math.round((m.current_strength || 0) * 100);
            const critical = pct < 25;
            const good = pct > 60;
            const color = pct > 60 ? '#10b981' : pct > 30 ? '#f59e0b' : '#ef4444';

            const card = document.createElement('div');
            card.className = `health-card ${critical ? 'critical' : good ? 'good' : ''}`;
            card.innerHTML = `
                <div class="health-card-label" title="${m.content}">${m.content}</div>
                <div class="health-bar-bg">
                    <div class="health-bar-fill" style="width:${pct}%;background:${color}"></div>
                </div>
                <div class="health-pct">${pct}% · ${(m.metadata?.recall_count || 0)}× recalled</div>
            `;
            grid.appendChild(card);
        });
    }

    // ── Graveyard ─────────────────────────────────────────────
    async function refreshGraveyard() {
        const data = await api('/api/graveyard');
        const list = document.getElementById('graveyard-list');
        const badge = document.getElementById('graveyard-count');
        const items = data.graveyard || [];

        badge.textContent = `${items.length} pruned`;
        if (items.length === 0) {
            list.innerHTML = '<div class="empty-state">No memories pruned yet. Fast-forward time to watch the Dumb Brain forget!</div>';
            return;
        }

        list.innerHTML = '';
        // Show newest first
        [...items].reverse().forEach(g => {
            const card = document.createElement('div');
            card.className = 'grave-card';
            const strength = (g.final_strength * 100).toFixed(1);
            const thresh = (g.threshold * 100).toFixed(0);
            card.innerHTML = `
                <div class="grave-header">
                    <span>🪦 ${g.content.substring(0, 50)}${g.content.length > 50 ? '…' : ''}</span>
                    <span style="font-size:0.72rem">📉 ${strength}% &lt; ${thresh}% threshold</span>
                </div>
                <div class="grave-meta">
                    Brain: ${g.profile} · Recalled ${g.recall_count}× · 
                    ${g.recall_count === 0 ? 'Never studied' : `Studied ${g.recall_count} time${g.recall_count > 1 ? 's' : ''}`}
                </div>
            `;
            list.appendChild(card);
        });
    }

    // ── Time Control ──────────────────────────────────────────
    document.querySelectorAll('.time-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const hours = parseFloat(btn.getAttribute('data-hours'));
            const data = await api('/api/time/advance', 'POST', { hours });
            await refreshClock();
            addEvent(`⏩ Jumped +${hours}h → ${data.display}`, 'info');
            if (data.pruned > 0) {
                addEvent(`☠️ ${data.pruned} memory/memories pruned by garbage collector!`, 'death');
                data.new_graveyard?.forEach(g =>
                    addEvent(`🪦 "${g.content.substring(0,40)}…" died at ${(g.final_strength*100).toFixed(1)}% strength`, 'death')
                );
            }
            await refreshAll();
        });
    });

    document.getElementById('reset-time-btn').addEventListener('click', async () => {
        await api('/api/time/reset', 'POST');
        addEvent('⏮ Time reset to now (real time)', 'info');
        await refreshClock();
        await refreshAll();
        await refreshGraveyard();
    });

    // ── Sleep Cycle ───────────────────────────────────────────
    document.getElementById('sleep-btn').addEventListener('click', async () => {
        const data = await api('/api/sleep', 'POST');
        addEvent(`🌙 Sleep cycle! +8h → ${data.display}. Consolidated: ${data.consolidated.length} memories. Pruned: ${data.pruned}.`, 'sleep');
        data.consolidated.forEach(id => addEvent(`✨ "${id}" consolidated during sleep (+10% strength)`, 'study'));
        data.new_graveyard?.forEach(g =>
            addEvent(`🪦 "${g.content.substring(0,40)}…" forgotten during sleep (${(g.final_strength*100).toFixed(1)}%)`, 'death')
        );
        await refreshClock();
        await refreshAll();
        await refreshGraveyard();
    });

    // ── Auto-Play ─────────────────────────────────────────────
    const autoplayBtn = document.getElementById('autoplay-btn');
    autoplayBtn.addEventListener('click', () => {
        if (autoPlayTimer) {
            clearInterval(autoPlayTimer);
            autoPlayTimer = null;
            autoplayBtn.textContent = '▶ Play';
            autoplayBtn.classList.remove('playing');
            addEvent('⏸ Auto-play paused', 'info');
        } else {
            autoplayBtn.textContent = '⏸ Pause';
            autoplayBtn.classList.add('playing');
            addEvent('▶ Auto-play started', 'info');

            const tick = async () => {
                const speed = parseFloat(document.getElementById('autoplay-speed').value);
                const data = await api('/api/time/advance', 'POST', { hours: speed });
                await refreshClock();
                if (data.pruned > 0) {
                    addEvent(`☠️ ${data.pruned} pruned at ${data.display}`, 'death');
                    data.new_graveyard?.forEach(g =>
                        addEvent(`🪦 "${g.content.substring(0,35)}…" died (${(g.final_strength*100).toFixed(1)}%)`, 'death')
                    );
                }
                await refreshAll();
                await refreshGraveyard();
            };
            autoPlayTimer = setInterval(tick, 1500);
        }
    });

    // ── Ebbinghaus Toggle ─────────────────────────────────────
    document.getElementById('toggle-ebbinghaus').addEventListener('change', () => refreshDecayChart());

    // ── Refresh Chart Button ───────────────────────────────────
    document.getElementById('refresh-chart-btn').addEventListener('click', refreshDecayChart);

    // ── Profile Switch ─────────────────────────────────────────
    document.getElementById('brain-profile').addEventListener('change', async (e) => {
        const prof = e.target.value;
        const data = await api('/api/profile', 'POST', { profile: prof });
        addEvent(`🔄 Switched to ${prof} brain profile. Pruned: ${data.pruned}`, 'info');
        await refreshAll();
        await refreshGraveyard();
    });

    // ── Search ─────────────────────────────────────────────────
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const resultsContainer = document.getElementById('search-results');

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;
        searchBtn.disabled = true; searchBtn.textContent = 'Searching...';
        try {
            const data = await api('/api/search', 'POST', { query });
            resultsContainer.innerHTML = '';
            if (!data.results.length) {
                resultsContainer.innerHTML = '<div class="empty-state">No results found.</div>';
                return;
            }
            data.results.forEach(r => {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `
                    <div class="result-header">
                        <span class="result-id">#${r.id}</span>
                        <span class="result-score">Score: ${r.score.toFixed(3)}</span>
                    </div>
                    <div class="result-content">${r.content}</div>
                    <div class="result-footer">
                        <div class="metrics">
                            <span>Rel: ${(r.relevance*100).toFixed(0)}%</span>
                            <span>Str: ${(r.strength*100).toFixed(1)}%</span>
                        </div>
                        <div class="feedback-buttons">
                            <button class="feedback-btn" data-reward="1">👍</button>
                            <button class="feedback-btn" data-reward="-1">👎</button>
                        </div>
                    </div>`;
                resultsContainer.appendChild(card);
            });
            addEvent(`🔍 Searched "${query}" → ${data.results.length} results retrieved & reinforced`, 'recall');
            if (data.pruned > 0) {
                addEvent(`☠️ ${data.pruned} weak memories pruned after search`, 'death');
                data.new_graveyard?.forEach(g =>
                    addEvent(`🪦 "${g.content.substring(0,35)}…" pruned`, 'death')
                );
                await refreshGraveyard();
            }

            // Feedback buttons - LinUCB bandit
            document.querySelectorAll('.feedback-btn').forEach(btn => {
                btn.addEventListener('click', async e => {
                    const reward = parseFloat(e.target.getAttribute('data-reward'));
                    const fb = await api('/api/feedback', 'POST', { reward });
                    e.target.parentElement.innerHTML = '<span style="font-size:0.76rem;color:#aaa">✓ Recorded</span>';
                    const armName = fb.arm_updated || 'unknown';
                    addEvent(`${reward > 0 ? '👍' : '👎'} Feedback → reward ${reward > 0 ? '+1' : '-1'} applied to arm: ${armName}`, reward > 0 ? 'study' : 'death');
                    await refreshBandit();
                });
            });

            await refreshDecayChart();
            await refreshHealthGrid();
        } catch (err) {
            console.error(err);
            resultsContainer.innerHTML = '<div class="empty-state" style="color:#ef4444">Search error.</div>';
        } finally {
            searchBtn.disabled = false; searchBtn.textContent = 'Search';
        }
    }

    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', e => e.key === 'Enter' && performSearch());

    // ── Add Memory + Interference Detection ────────────────────
    document.getElementById('add-memory-btn').addEventListener('click', async () => {
        const input = document.getElementById('add-memory-input');
        const content = input.value.trim();
        if (!content) return;
        const data = await api('/api/memories/add', 'POST', { content });
        input.value = '';
        addEvent(`🆕 Added memory: "${content.substring(0,40)}"`, 'study');
        if (data.interference?.length > 0) {
            data.interference.forEach(evt => {
                addEvent(`⚡ Interference! "${evt.content.substring(0,35)}…" weakened by ${(evt.penalty*100).toFixed(0)}% (${(evt.similarity*100).toFixed(0)}% similar)`, 'interf');
            });
        }
        await refreshAll();
    });

    // ── All Live Tab Refresh ───────────────────────────────────
    async function refreshAll() {
        await Promise.all([
            refreshDecayChart(),
            refreshHealthGrid(),
            refreshGraveyard(),
        ]);
    }

    // ── Brain Race ────────────────────────────────────────────
    document.getElementById('refresh-race-btn')?.addEventListener('click', refreshRace);

    async function refreshRace() {
        const data = await api('/api/brain_race');
        const profiles = [
            { key: 'smart',    color: '#3b82f6', fillFn: pct => `hsl(217,91%,${20+pct*0.4}%)` },
            { key: 'dumb',     color: '#ef4444', fillFn: pct => `hsl(0,84%,${20+pct*0.4}%)`   },
            { key: 'adaptive', color: '#8b5cf6', fillFn: pct => `hsl(258,90%,${20+pct*0.4}%)` },
        ];

        profiles.forEach(({ key, color }) => {
            const prof = data[key];
            const statsEl = document.getElementById(`race-${key}-stats`);
            const memsEl  = document.getElementById(`race-${key}-memories`);

            statsEl.innerHTML = `
                <span>Alive: ${prof.alive}/${prof.total}</span>
                <span>Avg: ${(prof.avg_strength*100).toFixed(1)}%</span>
            `;

            memsEl.innerHTML = '';
            prof.memories.forEach(m => {
                const pct = Math.round(m.strength * 100);
                const row = document.createElement('div');
                row.className = `race-mem${m.alive ? '' : ' pruned'}`;
                row.innerHTML = `
                    <span style="font-size:0.75rem;flex-shrink:0;width:14px">${m.alive ? '🟢' : '💀'}</span>
                    <span style="flex:0 0 auto;font-size:0.72rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:80px" title="${m.content}">${m.content}</span>
                    <div class="race-mem-bar">
                        <div class="race-mem-fill" style="width:${pct}%;background:${color}"></div>
                    </div>
                    <span style="font-size:0.7rem;color:#7c8fa6;flex-shrink:0">${pct}%</span>
                `;
                memsEl.appendChild(row);
            });
        });
    }

    // ── LinUCB Bandit Dashboard ───────────────────────────────
    const ARM_COLORS = ['#3b82f6', '#10b981', '#f59e0b'];
    let banditUCBHistory = []; // [{hlr, power, reinf}] for the UCB chart

    async function refreshBandit() {
        const data = await api('/api/bandit');
        const names = data.arm_names || ['HLR', 'Power-Law', 'Reinforcement'];
        const featureNames = data.feature_names || [];

        // -- Arm cards with active highlight --
        for (let i = 0; i < 3; i++) {
            const card = document.getElementById(`arm-card-${i}`);
            const ucbEl = document.getElementById(`arm-ucb-${i}`);
            if (!card) continue;

            const isActive = data.last_arm_selected === i;
            card.classList.toggle('arm-selected', isActive);
            card.style.borderColor = isActive ? ARM_COLORS[i] : 'rgba(255,255,255,0.08)';
            card.style.boxShadow = isActive ? `0 0 18px ${ARM_COLORS[i]}55` : 'none';

            const ucbScore = data.ucb_scores?.[i];
            ucbEl.textContent = ucbScore !== undefined ? `UCB: ${ucbScore.toFixed(3)}` : 'UCB: —';
            ucbEl.style.color = isActive ? ARM_COLORS[i] : '#7c8fa6';
        }

        // -- Context vector feature table --
        const ctxEl = document.getElementById('bandit-context');
        if (ctxEl && data.last_context && data.last_context.some(v => v !== 0)) {
            ctxEl.innerHTML = '';
            data.last_context.forEach((val, i) => {
                const fname = featureNames[i] || `Feature ${i}`;
                const pct = Math.min(100, Math.round(val * 100));
                ctxEl.innerHTML += `
                    <div class="ctx-feature">
                        <div class="ctx-feature-name">${fname}</div>
                        <div class="ctx-bar-bg">
                            <div class="ctx-bar-fill" style="width:${pct}%;background:${ARM_COLORS[i % 3]}"></div>
                        </div>
                        <span class="ctx-feature-val">${val.toFixed(4)}</span>
                    </div>`;
            });
        }

        // -- Theta vectors table (per arm) --
        const thetaEl = document.getElementById('bandit-theta');
        if (thetaEl && data.theta) {
            let html = '<table class="theta-table"><thead><tr><th>Feature</th>';
            names.forEach((n, i) => {
                html += `<th style="color:${ARM_COLORS[i]}">${n}</th>`;
            });
            html += '</tr></thead><tbody>';
            featureNames.forEach((fname, fi) => {
                html += `<tr><td class="theta-fname">${fname}</td>`;
                names.forEach((_, ai) => {
                    const val = data.theta[ai]?.[fi] ?? 0;
                    const valStr = val.toFixed(4);
                    const color = val > 0 ? '#10b981' : val < 0 ? '#ef4444' : '#7c8fa6';
                    html += `<td style="color:${color};font-family:monospace">${valStr}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            thetaEl.innerHTML = html;
        }

        // -- Feedback History --
        const histEl = document.getElementById('bandit-history');
        if (histEl) {
            histEl.innerHTML = '';
            if (!data.history || !data.history.length) {
                histEl.innerHTML = '<div class="empty-state">No feedback yet. Switch to Adaptive mode, search memories, then click thumbs-up or thumbs-down.</div>';
            } else {
                [...data.history].reverse().forEach((h, idx) => {
                    const el = document.createElement('div');
                    el.className = `bandit-entry ${h.reward > 0 ? 'positive' : 'negative'}`;
                    const ctx = h.context?.map(v => v.toFixed(3)).join(', ') || '—';
                    el.innerHTML = `
                        <strong>Round ${data.history.length - idx}:</strong> ${h.reward > 0 ? '👍' : '👎'}
                        → arm: <span style="color:${ARM_COLORS[names.indexOf(h.arm)]||'#7c8fa6'}">${h.arm}</span>
                        <span style="font-size:0.73rem;color:#7c8fa6;display:block;margin-top:2px">ctx: [${ctx}]</span>
                    `;
                    histEl.appendChild(el);
                });
            }
        }

        // -- UCB score evolution chart --
        if (data.ucb_scores) {
            banditUCBHistory.push({
                hlr:   data.ucb_scores[0],
                power: data.ucb_scores[1],
                reinf: data.ucb_scores[2],
            });
            if (banditUCBHistory.length > 20) banditUCBHistory.shift();
        }
        if (banditUCBHistory.length > 1) {
            const bCtx = document.getElementById('banditChart').getContext('2d');
            const labels = banditUCBHistory.map((_, i) => `Q${i+1}`);
            const bDatasets = [
                { label: 'HLR UCB',           data: banditUCBHistory.map(h => h.hlr),   borderColor: ARM_COLORS[0], backgroundColor: ARM_COLORS[0]+'20', borderWidth: 2, pointRadius: 3, fill: false },
                { label: 'Power-Law UCB',      data: banditUCBHistory.map(h => h.power), borderColor: ARM_COLORS[1], backgroundColor: ARM_COLORS[1]+'20', borderWidth: 2, pointRadius: 3, fill: false },
                { label: 'Reinforcement UCB',  data: banditUCBHistory.map(h => h.reinf), borderColor: ARM_COLORS[2], backgroundColor: ARM_COLORS[2]+'20', borderWidth: 2, pointRadius: 3, fill: false },
            ];
            if (banditChart) banditChart.destroy();
            banditChart = new Chart(bCtx, {
                type: 'line',
                data: { labels, datasets: bDatasets },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    animation: { duration: 400 },
                    plugins: { legend: { labels: { color: '#94a3b8', usePointStyle: true, font: { size: 11 } } } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#7c8fa6' } },
                        y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#7c8fa6', font: { size: 11 } },
                            title: { display: true, text: 'UCB Score', color: '#7c8fa6' } }
                    }
                },
            });
        }
    }

    // ── Alpha Slider (LinUCB exploration parameter) ───────────
    document.getElementById('alpha-slider').addEventListener('input', async e => {
        const val = parseFloat(e.target.value);
        document.getElementById('alpha-value').textContent = val.toFixed(2);
        await api('/api/bandit/tune', 'POST', { alpha: val });
        addEvent(`LinUCB alpha → ${val.toFixed(2)} (${val > 2 ? 'high exploration' : 'exploitation mode'})`, 'info');
        await refreshBandit();
    });

    // ── Student Simulation ────────────────────────────────────
    document.getElementById('run-simulation-btn').addEventListener('click', async () => {
        const resultsEl = document.getElementById('sim-results');
        const loadingEl = document.getElementById('sim-loading');
        const runBtn    = document.getElementById('run-simulation-btn');

        resultsEl.classList.add('hidden');
        loadingEl.classList.remove('hidden');
        runBtn.disabled = true; runBtn.textContent = 'Running...';

        try {
            const data = await api('/api/simulation/student');
            renderSimulation(data);
        } catch (e) {
            console.error(e);
            alert('Simulation failed — is the server running?');
        } finally {
            loadingEl.classList.add('hidden');
            runBtn.disabled = false; runBtn.textContent = '▶ Run Simulation';
        }
    });

    function renderSimulation(data) {
        const { smart, dumb } = data;
        document.getElementById('smart-score').textContent = `${smart.remembered_count}/${smart.total}`;
        document.getElementById('dumb-score').textContent  = `${dumb.remembered_count}/${dumb.total}`;

        document.getElementById('schedule-note').innerHTML = `
            📚 Study Schedule: Day 1 — ${data.study_schedule['1'].join(', ')} &nbsp;|&nbsp;
            Day 3 — ${data.study_schedule['3'].join(', ')} &nbsp;|&nbsp;
            📝 Exam Day 7 (≥20% threshold)
        `;

        // Timeline chart
        const days = smart.timeline.map(t => `Day ${t.day}`);
        const topicIds = Object.keys(smart.timeline[0].strengths);
        const datasets = [];
        topicIds.forEach((tid, i) => {
            const hue = (i * 36) % 360;
            datasets.push({
                label: `Smart – ${tid}`,
                data: smart.timeline.map(t => t.strengths[tid]),
                borderColor: `hsl(${hue},70%,60%)`, borderWidth: 2,
                pointRadius: 3, fill: false, tension: 0.4,
            });
            datasets.push({
                label: `Dumb – ${tid}`,
                data: dumb.timeline.map(t => t.strengths[tid]),
                borderColor: `hsl(${hue},70%,60%)`, borderWidth: 2,
                borderDash: [5, 3], pointRadius: 2, fill: false, tension: 0.4,
            });
        });

        const sCtx = document.getElementById('simChart').getContext('2d');
        if (simChart) simChart.destroy();
        simChart = new Chart(sCtx, {
            type: 'line',
            data: { labels: days, datasets },
            options: makeChartOpts('Memory Strength'),
        });

        // Table
        const tbody = document.getElementById('sim-table-body');
        tbody.innerHTML = '';
        smart.snapshots.forEach(ss => {
            const ds = dumb.snapshots.find(d => d.id === ss.id) || {};
            const recalled = ss.recalled_on_days.length
                ? ss.recalled_on_days.map(d => `Day ${d}`).join(', ') : '—';
            const sPct = (ss.strength_at_exam * 100).toFixed(1);
            const dPct = ((ds.strength_at_exam || 0) * 100).toFixed(1);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td title="${ss.id}">${ss.content}</td>
                <td>${recalled}</td>
                <td><div class="strength-bar-wrap">
                    <div class="strength-bar smart-bar" style="width:${Math.min(100, parseFloat(sPct))}px"></div>
                    <span>${sPct}%</span></div></td>
                <td><div class="strength-bar-wrap">
                    <div class="strength-bar dumb-bar" style="width:${Math.min(100, parseFloat(dPct))}px"></div>
                    <span>${dPct}%</span></div></td>
                <td><span class="badge ${ss.remembered ? 'badge-yes' : 'badge-no'}">${ss.remembered ? '✓ Yes' : '✗ No'}</span></td>
                <td><span class="badge ${ds.remembered ? 'badge-yes' : 'badge-no'}">${ds.remembered ? '✓ Yes' : '✗ No'}</span></td>
            `;
            tbody.appendChild(row);
        });

        document.getElementById('sim-results').classList.remove('hidden');
    }

    // ── Initial Boot Sequence ─────────────────────────────────
    async function boot() {
        await refreshClock();
        await refreshAll();
        addEvent('🚀 Decay Lab initialized. All systems online.', 'info');
        addEvent('🎯 Tip: Use +1h/+1d buttons or ▶ Play to see memories decay in real time!', 'info');
        addEvent('💡 Tip: Switch to Dumb Brain and add a memory, then fast-forward to watch it die!', 'info');
    }

    boot();
});
