// ════════════════════════════════════════════════════════════════════════════
// Confidence-Aware Adaptive RAG - Frontend JavaScript
// ════════════════════════════════════════════════════════════════════════════

const API_BASE = '';

// ── DOM Elements ────────────────────────────────────────────────────────────

const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const systemStatus = document.getElementById('systemStatus');
const examplesList = document.getElementById('examplesList');
const resultsSection = document.getElementById('resultsSection');
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');
const errorMessage = document.getElementById('errorMessage');

// Results elements
const displayedQuery = document.getElementById('displayedQuery');
const finalAnswerText = document.getElementById('finalAnswerText');
const finalConfidenceValue = document.getElementById('finalConfidenceValue');
const finalConfidenceBadge = document.getElementById('finalConfidenceBadge');
const roundsInfo = document.getElementById('roundsInfo');
const roundsContainer = document.getElementById('roundsContainer');
const signalsChart = document.getElementById('signalsChart');
const systemNotice = document.getElementById('systemNotice');
const modelBackendSelect = document.getElementById('modelBackend');
const modelIdSelect = document.getElementById('modelId');
const switchModelBtn = document.getElementById('switchModelBtn');

// Active backend (from /api/status)
let llmBackend = null;
let availableModels = {};

// ── Initialization ──────────────────────────────────────────────────────────

async function init() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        // Capture backend type from server
        llmBackend = (data && data.config && data.config.llm_backend) ? data.config.llm_backend : null;
        const modelId = (data && data.config && data.config.model_id) ? data.config.model_id : null;
        
        if (data.status === 'ready') {
            systemStatus.classList.add('ready');
            const st = systemStatus.querySelector('.status-text');
            st.textContent = llmBackend ? `Ready — ${llmBackend}` : 'System Ready';
            submitBtn.disabled = false;
        } else {
            systemStatus.querySelector('.status-text').textContent = 'System Not Ready';
        }
        
        // Set current backend in selector
        if (modelBackendSelect && llmBackend) {
            modelBackendSelect.value = llmBackend;
        }
        
        // Show/hide notice based on backend
        if (systemNotice) {
            systemNotice.style.display = (llmBackend === 'mock') ? 'flex' : 'none';
        }

        // Load example queries and available models
        loadExamples();
    } catch (error) {
        console.error('Failed to check system status:', error);
        systemStatus.querySelector('.status-text').textContent = 'Connection Error';
    }
}

async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE}/api/examples`);
        const data = await response.json();
        
        // Store available models
        if (data.available_models) {
            availableModels = data.available_models;
        }
        
        // Populate examples
        examplesList.innerHTML = '';
        data.examples.forEach(example => {
            const chip = document.createElement('div');
            chip.className = 'example-chip';
            chip.textContent = example;
            chip.onclick = () => {
                queryInput.value = example;
                queryInput.focus();
            };
            examplesList.appendChild(chip);
        });
    } catch (error) {
        console.error('Failed to load examples:', error);
    }
}

// ── Event Handlers ──────────────────────────────────────────────────────────

submitBtn.addEventListener('click', handleSubmit);

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        handleSubmit();
    }
});

async function handleSubmit() {
    const query = queryInput.value.trim();
    
    if (!query) {
        showError('Please enter a question');
        return;
    }
    
    // Reset UI
    hideError();
    resultsSection.style.display = 'none';
    loadingState.style.display = 'block';
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').style.display = 'none';
    submitBtn.querySelector('.btn-loader').style.display = 'inline-block';
    
    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });
        
        if (!response.ok) {
            throw new Error('Failed to process query');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error processing query:', error);
        showError('Failed to process query. Please try again.');
    } finally {
        loadingState.style.display = 'none';
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').style.display = 'inline-block';
        submitBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

// ── Display Functions ───────────────────────────────────────────────────────

function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Display query
    if (displayedQuery) {
        displayedQuery.textContent = data.query;
    }
    
    // Display final answer
    if (finalAnswerText) {
        finalAnswerText.textContent = data.final_answer;
    }
    if (finalConfidenceValue) {
        finalConfidenceValue.textContent = data.final_confidence.toFixed(3);
    }
    
    // Show mock LLM notice if applicable
    if (systemNotice && data.final_answer && data.final_answer.includes('mock')) {
        systemNotice.style.display = 'flex';
    } else if (systemNotice) {
        systemNotice.style.display = 'none';
    }
    
    // Set confidence badge color
    finalConfidenceBadge.classList.remove('high', 'medium', 'low');
    if (data.final_confidence >= 0.7) {
        finalConfidenceBadge.classList.add('high');
    } else if (data.final_confidence >= 0.4) {
        finalConfidenceBadge.classList.add('medium');
    } else {
        finalConfidenceBadge.classList.add('low');
    }
    
    // Display rounds info
    roundsInfo.textContent = `${data.total_rounds} round${data.total_rounds !== 1 ? 's' : ''}`;
    
    // Display rounds
    displayRounds(data.rounds);
    
    // Display signals chart
    displaySignalsChart(data.rounds);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayRounds(rounds) {
    roundsContainer.innerHTML = '';
    
    rounds.forEach((round, index) => {
        const roundDiv = document.createElement('div');
        roundDiv.className = 'round-item';
        
        const signals = round.signals;
        const confidence = round.confidence;
        
        roundDiv.innerHTML = `
            <div class="round-header">
                <div class="round-title">Round ${round.round}</div>
                <div class="round-k-badge">k = ${round.k}</div>
            </div>
            
            <div class="signals-grid">
                <div class="signal-item">
                    <span class="signal-label">Retrieval (Sr)</span>
                    <div class="signal-value">${signals.Sr.toFixed(3)}</div>
                    <div class="signal-bar">
                        <div class="signal-bar-fill" style="width: ${signals.Sr * 100}%"></div>
                    </div>
                </div>
                
                <div class="signal-item">
                    <span class="signal-label">LLM Confidence (Sl)</span>
                    <div class="signal-value">${signals.Sl.toFixed(3)}</div>
                    <div class="signal-bar">
                        <div class="signal-bar-fill" style="width: ${signals.Sl * 100}%"></div>
                    </div>
                </div>
                
                <div class="signal-item">
                    <span class="signal-label">Consistency (Sc)</span>
                    <div class="signal-value">${signals.Sc.toFixed(3)}</div>
                    <div class="signal-bar">
                        <div class="signal-bar-fill" style="width: ${signals.Sc * 100}%"></div>
                    </div>
                </div>
                
                <div class="signal-item">
                    <span class="signal-label">Final Confidence (C)</span>
                    <div class="signal-value" style="color: ${getConfidenceColor(confidence)}">
                        ${confidence.toFixed(3)}
                    </div>
                    <div class="signal-bar">
                        <div class="signal-bar-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                </div>
            </div>
            
            ${round.current_query !== rounds[0].current_query ? 
                `<div class="round-answer">
                    <strong>Reformulated Query:</strong> ${round.current_query}
                </div>` : ''}
            
            <div class="round-answer">
                <strong>Answer:</strong> ${round.answer.substring(0, 200)}${round.answer.length > 200 ? '...' : ''}
            </div>
        `;
        
        roundsContainer.appendChild(roundDiv);
    });
}

function displaySignalsChart(rounds) {
    signalsChart.innerHTML = '';
    
    rounds.forEach((round, index) => {
        const chartRound = document.createElement('div');
        chartRound.className = 'chart-round';
        
        const signals = round.signals;
        const maxHeight = 200;
        
        chartRound.innerHTML = `
            <div class="chart-bars">
                <div class="chart-bar sr" style="height: ${signals.Sr * maxHeight}px">
                    <div class="chart-bar-label">${signals.Sr.toFixed(2)}</div>
                </div>
                <div class="chart-bar sl" style="height: ${signals.Sl * maxHeight}px">
                    <div class="chart-bar-label">${signals.Sl.toFixed(2)}</div>
                </div>
                <div class="chart-bar sc" style="height: ${signals.Sc * maxHeight}px">
                    <div class="chart-bar-label">${signals.Sc.toFixed(2)}</div>
                </div>
            </div>
            <div class="chart-round-label">Round ${round.round}</div>
        `;
        
        signalsChart.appendChild(chartRound);
    });
}

function getConfidenceColor(confidence) {
    if (confidence >= 0.7) return '#10b981';
    if (confidence >= 0.4) return '#f59e0b';
    return '#ef4444';
}

function showError(message) {
    errorMessage.textContent = message;
    errorState.style.display = 'block';
    setTimeout(() => {
        errorState.style.display = 'none';
    }, 5000);
}

function hideError() {
    errorState.style.display = 'none';
}

// ── Model Switching ─────────────────────────────────────────────────────────

if (modelBackendSelect) {
    modelBackendSelect.addEventListener('change', (e) => {
        const backend = e.target.value;
        
        // Show/hide model ID selector based on backend
        if (backend === 'mock') {
            modelIdSelect.style.display = 'none';
        } else if (availableModels[backend] && availableModels[backend].models) {
            modelIdSelect.style.display = 'inline-block';
            modelIdSelect.innerHTML = '';
            availableModels[backend].models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelIdSelect.appendChild(option);
            });
        }
    });
}

if (switchModelBtn) {
    switchModelBtn.addEventListener('click', async () => {
        const backend = modelBackendSelect.value;
        const modelId = modelIdSelect.style.display === 'none' ? '' : modelIdSelect.value;
        
        switchModelBtn.disabled = true;
        switchModelBtn.textContent = 'Switching...';
        
        try {
            const response = await fetch(`${API_BASE}/api/switch-model`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ backend, model_id: modelId })
            });
            
            if (!response.ok) {
                throw new Error('Failed to switch model');
            }
            
            const data = await response.json();
            llmBackend = data.backend;
            
            // Update status
            const st = systemStatus.querySelector('.status-text');
            st.textContent = `Ready — ${data.backend}`;
            
            // Update notice
            if (systemNotice) {
                systemNotice.style.display = (data.backend === 'mock') ? 'flex' : 'none';
            }
            
            // Show success message
            showError(`Switched to ${data.backend} backend successfully!`);
            
        } catch (error) {
            console.error('Error switching model:', error);
            showError('Failed to switch model. Please try again.');
        } finally {
            switchModelBtn.disabled = false;
            switchModelBtn.textContent = 'Switch';
        }
    });
}

// ── Initialize on page load ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
