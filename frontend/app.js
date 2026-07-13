const API_URL = "http://localhost:8000";

// DOM Elements
const authView = document.getElementById('auth-view');
const dashboardView = document.getElementById('dashboard-view');
const toggleLoginBtn = document.getElementById('toggle-login');
const toggleSignupBtn = document.getElementById('toggle-signup');
const authForm = document.getElementById('auth-form');
const authSubmitBtn = document.getElementById('auth-submit-btn');
const authMessage = document.getElementById('auth-message');
const userDisplay = document.getElementById('user-display');
const logoutBtn = document.getElementById('logout-btn');

// State
let token = localStorage.getItem('token');
let username = localStorage.getItem('username');
let isLoginMode = true;

// Initialization
function init() {
    if (token) {
        showDashboard();
    } else {
        showAuth();
    }
}

// Toast
function showToast(msg, type='success') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = `toast show ${type}`;
    setTimeout(() => { toast.classList.remove('show'); }, 3000);
}

// Auth Toggle
toggleLoginBtn.addEventListener('click', () => {
    isLoginMode = true;
    toggleLoginBtn.classList.add('active');
    toggleSignupBtn.classList.remove('active');
    authSubmitBtn.textContent = 'Login';
    authMessage.textContent = '';
});

toggleSignupBtn.addEventListener('click', () => {
    isLoginMode = false;
    toggleSignupBtn.classList.add('active');
    toggleLoginBtn.classList.remove('active');
    authSubmitBtn.textContent = 'Sign Up';
    authMessage.textContent = '';
});

// Auth Submit
authForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const u = document.getElementById('username').value;
    const p = document.getElementById('password').value;
    
    try {
        const endpoint = isLoginMode ? '/auth/login' : '/auth/signup';
        const res = await fetch(`${API_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: u, password: p })
        });
        const data = await res.json();
        
        if (res.ok) {
            if (isLoginMode) {
                token = data.access_token;
                username = u;
                localStorage.setItem('token', token);
                localStorage.setItem('username', username);
                showDashboard();
                showToast('Logged in successfully');
            } else {
                authMessage.textContent = 'Account created. Please login.';
                authMessage.style.color = 'var(--success)';
                toggleLoginBtn.click(); // Switch to login mode
            }
        } else {
            authMessage.textContent = data.detail || 'Authentication failed';
            authMessage.style.color = 'var(--danger)';
        }
    } catch (err) {
        authMessage.textContent = 'Server error. Is the API running?';
        authMessage.style.color = 'var(--danger)';
    }
});

// Navigation
function showAuth() {
    authView.classList.remove('hidden');
    dashboardView.classList.add('hidden');
}

function showDashboard() {
    authView.classList.add('hidden');
    dashboardView.classList.remove('hidden');
    userDisplay.textContent = username;
    document.querySelector('.nav-btn[data-target="tab-predict"]').click();
}

logoutBtn.addEventListener('click', () => {
    token = null;
    username = null;
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    showAuth();
});

// Tabs Logic
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
        
        const targetBtn = e.currentTarget;
        targetBtn.classList.add('active');
        document.getElementById(targetBtn.dataset.target).classList.remove('hidden');
        
        if(targetBtn.dataset.target === 'tab-history') {
            loadHistory();
        } else if(targetBtn.dataset.target === 'tab-insights') {
            loadChart();
        }
    });
});

// Predict API
document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        type: document.getElementById('p_type').value,
        fixed_acidity: parseFloat(document.getElementById('p_fixed_acidity').value),
        volatile_acidity: parseFloat(document.getElementById('p_volatile_acidity').value),
        citric_acid: parseFloat(document.getElementById('p_citric_acid').value),
        residual_sugar: parseFloat(document.getElementById('p_residual_sugar').value),
        chlorides: parseFloat(document.getElementById('p_chlorides').value),
        free_sulfur_dioxide: parseFloat(document.getElementById('p_free_sulfur').value),
        total_sulfur_dioxide: parseFloat(document.getElementById('p_total_sulfur').value),
        density: parseFloat(document.getElementById('p_density').value),
        pH: parseFloat(document.getElementById('p_ph').value),
        sulphates: parseFloat(document.getElementById('p_sulphates').value),
        alcohol: parseFloat(document.getElementById('p_alcohol').value)
    };

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(payload)
        });
        if (res.status === 401) { showAuth(); return; }
        
        const data = await res.json();
        if (res.ok) {
            document.getElementById('pred-score').textContent = data.rounded_prediction;
            document.getElementById('pred-exact').textContent = `Exact: ${data.prediction.toFixed(4)}`;
            const pct = (data.prediction / 10) * 100;
            document.getElementById('meter-bar').style.width = `${pct}%`;
            showToast('Prediction successful!');
        } else {
            showToast(data.detail || 'Error', 'error');
        }
    } catch (err) {
        showToast('API Connection failed', 'error');
    }
});

// Batch Upload Logic
const dropZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('csv-file');
let uploadedFile = null;

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

async function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }
    dropZone.querySelector('p').textContent = `Selected: ${file.name}`;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const res = await fetch(`${API_URL}/predict_batch`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` },
            body: formData
        });
        if (res.status === 401) { showAuth(); return; }
        
        if (res.ok) {
            const csvText = await res.text();
            renderBatchTable(csvText);
            document.getElementById('batch-results').classList.remove('hidden');
            
            // Setup download button
            const blob = new Blob([csvText], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.getElementById('download-csv-btn');
            a.onclick = () => {
                const link = document.createElement('a');
                link.href = url;
                link.download = 'predictions.csv';
                link.click();
            };
            showToast('Batch processed successfully');
        } else {
            const err = await res.json();
            showToast(err.detail || 'Batch error', 'error');
        }
    } catch (e) {
        showToast('Upload failed', 'error');
    }
}

function renderBatchTable(csvStr) {
    const lines = csvStr.trim().split('\n');
    const thead = document.querySelector('#batch-table thead');
    const tbody = document.querySelector('#batch-table tbody');
    thead.innerHTML = ''; tbody.innerHTML = '';
    
    const headers = lines[0].split(',');
    const trH = document.createElement('tr');
    headers.forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        trH.appendChild(th);
    });
    thead.appendChild(trH);
    
    for(let i=1; i<Math.min(lines.length, 6); i++) { // show max 5 rows
        const cols = lines[i].split(',');
        const tr = document.createElement('tr');
        cols.forEach(c => {
            const td = document.createElement('td');
            td.textContent = parseFloat(c) ? parseFloat(c).toFixed(2) : c;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    }
}

// History
async function loadHistory() {
    try {
        const res = await fetch(`${API_URL}/predictions_history`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.status === 401) { showAuth(); return; }
        
        if (res.ok) {
            const data = await res.json();
            const tbody = document.querySelector('#history-table tbody');
            tbody.innerHTML = '';
            data.reverse().slice(0, 15).forEach(log => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${log.id}</td>
                    <td>${new Date(log.timestamp).toLocaleString()}</td>
                    <td>${log.wine_type}</td>
                    <td>${log.alcohol}</td>
                    <td>${log.pH}</td>
                    <td><span class="badge" style="color:var(--text-color); background:var(--primary-color)">${log.predicted_quality.toFixed(2)}</span></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch(e) {}
}
document.getElementById('refresh-history').addEventListener('click', loadHistory);

// Insights Chart
let chartInstance = null;
function loadChart() {
    if (chartInstance) return; // already loaded
    
    const ctx = document.getElementById('importanceChart').getContext('2d');
    
    // Hardcoded for demo matching streamlit
    const features = ["alcohol", "volatile acidity", "free sulfur dioxide", "sulphates", "density", "citric acid", "total sulfur dioxide", "residual sugar", "chlorides", "pH", "fixed acidity", "is_red"];
    const importances = [0.35, 0.15, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03];
    
    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Feature Importance',
                data: importances,
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            },
            scales: {
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
            }
        }
    });
}

// Start
init();
