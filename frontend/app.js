/* =============================================
   Wine Quality Predictor — Frontend Logic
   ============================================= */

const API = 'http://localhost:8000';

// ── State ─────────────────────────────────────
let token    = localStorage.getItem('wq_token');
let username = localStorage.getItem('wq_user');
let chartInstance = null;

// ── DOM refs ──────────────────────────────────
const authView     = document.getElementById('auth-view');
const dashView     = document.getElementById('dashboard-view');
const tabLogin     = document.getElementById('tab-login');
const tabSignup    = document.getElementById('tab-signup');
const authForm     = document.getElementById('auth-form');
const authBtnText  = document.getElementById('auth-btn-text');
const authMsg      = document.getElementById('auth-message');
const userNameEl   = document.getElementById('user-name');
const logoutBtn    = document.getElementById('logout-btn');

let isLogin = true;

// ── Boot ──────────────────────────────────────
(function init() {
    if (token) enterDashboard();
    else showAuth();
})();

// ── Toast ─────────────────────────────────────
function toast(msg, type = 'success') {
    const el = document.getElementById('toast');
    el.textContent  = msg;
    el.className    = `show ${type}`;
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.className = ''; }, 3000);
}

// ── Auth view ─────────────────────────────────
function showAuth() {
    authView.classList.remove('hidden');
    dashView.classList.add('hidden');
}
function enterDashboard() {
    authView.classList.add('hidden');
    dashView.classList.remove('hidden');
    userNameEl.textContent = username;
    // activate default tab
    activateTab('tab-predict');
}

tabLogin.addEventListener('click', () => {
    isLogin = true;
    tabLogin.classList.add('active');
    tabSignup.classList.remove('active');
    authBtnText.textContent = 'Enter';
    authMsg.className = '';
    authMsg.textContent = '';
});
tabSignup.addEventListener('click', () => {
    isLogin = false;
    tabSignup.classList.add('active');
    tabLogin.classList.remove('active');
    authBtnText.textContent = 'Create Account';
    authMsg.className = '';
    authMsg.textContent = '';
});

// ── Auth submit ───────────────────────────────
authForm.addEventListener('submit', async e => {
    e.preventDefault();
    const u = document.getElementById('auth-username').value.trim();
    const p = document.getElementById('auth-password').value;
    if (!u || !p) return;

    const endpoint = isLogin ? '/auth/login' : '/auth/signup';
    try {
        const res  = await fetch(API + endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: u, password: p })
        });
        const data = await res.json();

        if (res.ok) {
            if (isLogin) {
                token    = data.access_token;
                username = u;
                localStorage.setItem('wq_token', token);
                localStorage.setItem('wq_user',  username);
                enterDashboard();
                toast('Welcome back, ' + u + ' 🍷');
            } else {
                authMsg.className   = 'success';
                authMsg.textContent = 'Account created — please log in.';
                tabLogin.click();
            }
        } else {
            authMsg.className   = 'error';
            authMsg.textContent = data.detail || 'Authentication failed.';
        }
    } catch {
        authMsg.className   = 'error';
        authMsg.textContent = 'Cannot reach API server.';
    }
});

// ── Logout ────────────────────────────────────
logoutBtn.addEventListener('click', () => {
    token = username = null;
    localStorage.removeItem('wq_token');
    localStorage.removeItem('wq_user');
    showAuth();
});

// ── Tab navigation ────────────────────────────
function activateTab(id) {
    document.querySelectorAll('.nav-item').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === id);
    });
    document.querySelectorAll('.tab-section').forEach(s => {
        s.classList.toggle('active', s.id === id);
    });
    if (id === 'tab-history')  loadHistory();
    if (id === 'tab-insights') loadChart();
}
document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
});

// ── Predict ───────────────────────────────────
document.getElementById('predict-form').addEventListener('submit', async e => {
    e.preventDefault();
    const payload = {
        type:               document.getElementById('p-type').value,
        fixed_acidity:      parseFloat(document.getElementById('p-fixed-acidity').value),
        volatile_acidity:   parseFloat(document.getElementById('p-volatile-acidity').value),
        citric_acid:        parseFloat(document.getElementById('p-citric-acid').value),
        residual_sugar:     parseFloat(document.getElementById('p-residual-sugar').value),
        chlorides:          parseFloat(document.getElementById('p-chlorides').value),
        free_sulfur_dioxide:parseFloat(document.getElementById('p-free-sulfur').value),
        total_sulfur_dioxide:parseFloat(document.getElementById('p-total-sulfur').value),
        density:            parseFloat(document.getElementById('p-density').value),
        pH:                 parseFloat(document.getElementById('p-ph').value),
        sulphates:          parseFloat(document.getElementById('p-sulphates').value),
        alcohol:            parseFloat(document.getElementById('p-alcohol').value),
    };

    try {
        const res  = await fetch(API + '/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
            body: JSON.stringify(payload)
        });
        if (res.status === 401) { showAuth(); return; }
        const data = await res.json();
        if (!res.ok) { toast(data.detail || 'Prediction failed', 'error'); return; }

        const rounded = data.rounded_prediction;
        const exact   = data.prediction;
        document.getElementById('pred-score').textContent = rounded;
        document.getElementById('pred-exact').textContent = `Exact score: ${exact.toFixed(4)}`;
        document.getElementById('quality-bar').style.width = `${(exact / 10) * 100}%`;
        toast('Prediction complete ✓');
    } catch {
        toast('API connection failed', 'error');
    }
});

// ── Batch upload ──────────────────────────────
const dropZone  = document.getElementById('drop-zone');
const csvInput  = document.getElementById('csv-input');
let   blobUrl   = null;

document.getElementById('browse-link').addEventListener('click', () => csvInput.click());
csvInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
    if (!file.name.endsWith('.csv')) { toast('Please upload a .csv file', 'error'); return; }
    dropZone.querySelector('p').textContent = `Processing: ${file.name}…`;

    const form = new FormData();
    form.append('file', file);
    try {
        const res = await fetch(API + '/predict_batch', {
            method: 'POST',
            headers: { 'Authorization': 'Bearer ' + token },
            body: form
        });
        if (res.status === 401) { showAuth(); return; }
        if (!res.ok) { const d = await res.json(); toast(d.detail || 'Batch error', 'error'); return; }

        const csv = await res.text();
        renderBatchTable(csv);

        if (blobUrl) URL.revokeObjectURL(blobUrl);
        blobUrl = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
        document.getElementById('download-btn').onclick = () => {
            const a = Object.assign(document.createElement('a'), { href: blobUrl, download: 'predictions.csv' });
            a.click();
        };
        document.getElementById('batch-output').classList.remove('hidden');
        dropZone.querySelector('p').textContent = `Done: ${file.name}`;
        toast('Batch complete — ' + (csv.split('\n').length - 2) + ' rows predicted');
    } catch {
        toast('Upload failed', 'error');
    }
}

function renderBatchTable(csv) {
    const rows = csv.trim().split('\n');
    const thead = document.querySelector('#batch-table thead');
    const tbody = document.querySelector('#batch-table tbody');
    thead.innerHTML = tbody.innerHTML = '';
    const headers = rows[0].split(',');
    thead.innerHTML = '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>';
    rows.slice(1, 11).forEach(row => {
        const cols = row.split(',');
        tbody.innerHTML += '<tr>' + cols.map((c, i) => {
            const val = parseFloat(c);
            const disp = isNaN(val) ? c : (i === cols.length - 1 ? `<span class="score-pill">${val.toFixed(2)}</span>` : val.toFixed(3));
            return `<td>${disp}</td>`;
        }).join('') + '</tr>';
    });
}

// ── History ───────────────────────────────────
async function loadHistory() {
    try {
        const res  = await fetch(API + '/predictions_history', { headers: { 'Authorization': 'Bearer ' + token } });
        if (res.status === 401) { showAuth(); return; }
        const data = await res.json();
        const tbody = document.getElementById('history-body');
        tbody.innerHTML = '';
        data.slice(0, 20).forEach(log => {
            const date = new Date(log.timestamp).toLocaleString();
            tbody.innerHTML += `
            <tr>
                <td>${log.id}</td>
                <td>${date}</td>
                <td>${log.wine_type}</td>
                <td>${log.alcohol}</td>
                <td>${log.pH}</td>
                <td><span class="score-pill">${log.predicted_quality.toFixed(2)}</span></td>
            </tr>`;
        });
        if (!data.length) tbody.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center;padding:2rem">No records yet</td></tr>';
    } catch {}
}
document.getElementById('refresh-btn').addEventListener('click', loadHistory);

// ── Insights chart ────────────────────────────
function loadChart() {
    if (chartInstance) return;
    const features    = ['alcohol','volatile acidity','free SO₂','sulphates','density','citric acid','total SO₂','residual sugar','chlorides','pH','fixed acidity','is_red'];
    const importances = [0.35,0.15,0.08,0.07,0.06,0.05,0.05,0.05,0.04,0.04,0.03,0.03];
    const ctx = document.getElementById('importance-chart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Feature Importance',
                data: importances,
                backgroundColor: features.map((_, i) => `hsla(${30 + i * 12}, 70%, 55%, 0.5)`),
                borderColor:     features.map((_, i) => `hsla(${30 + i * 12}, 70%, 65%, 1)`),
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive:true, maintainAspectRatio:true,
            plugins: {
                legend: { labels: { color:'#a09080', font:{ family:'Inter',size:13 } } }
            },
            scales: {
                x: { ticks:{color:'#a09080'}, grid:{color:'rgba(255,255,255,0.04)'} },
                y: { ticks:{color:'#a09080'}, grid:{color:'rgba(255,255,255,0.06)'} }
            }
        }
    });
}
