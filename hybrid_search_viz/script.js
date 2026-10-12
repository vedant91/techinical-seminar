/* ── All real API calls go through this base URL ──────────────────────────── */
const API = '';   // same-origin (FastAPI serves both /api and the static files)

/* ── Utilities ────────────────────────────────────────────────────────────── */
const $  = id => document.getElementById(id);
const fmt = obj => JSON.stringify(obj, null, 2);

function setBadge(el, state, label) {
    el.className = `badge badge-${state}`;
    el.textContent = label;
}

/* ── Store Status Banner ──────────────────────────────────────────────────── */
let storeReady = false;  // tracks whether the backend store has any indexed chunks

async function checkStoreStatus() {
    try {
        const res  = await fetch(`${API}/api/status`);
        const data = await res.json();
        storeReady = data.ready;
        const banner = $('store-warning');
        if (!storeReady) {
            banner.innerHTML = `
                <strong>⚠ Store is empty.</strong> The server restarted and the in-memory index was cleared.
                Go to <strong>Step 1 — Document Upload</strong> and re-ingest your document before querying.
                <em style="color:var(--txt3)">(Store: ${data.store_type}, Chunks: ${data.chunk_count})</em>
            `;
            banner.classList.remove('hidden');
        } else {
            banner.classList.add('hidden');
        }
    } catch (e) {
        // Server not ready yet — silently ignore
    }
}

// Poll on load and every 10s so the banner self-clears after ingestion
checkStoreStatus();
setInterval(checkStoreStatus, 10000);

/* ── Panel / Nav Logic ─────────────────────────────────────────────────────── */
const navItems = document.querySelectorAll('.nav-item');
const panels   = document.querySelectorAll('.panel');
const pageTitle = $('page-title');

function showPanel(id) {
    panels.forEach(p => p.classList.remove('active'));
    navItems.forEach(n => {
        n.classList.remove('active');
        if (n.dataset.panel === id) n.classList.add('active');
    });
    const panel = $(id);
    if (panel) panel.classList.add('active');
    // update page title from nav label
    const nav = document.querySelector(`[data-panel="${id}"]`);
    if (nav) pageTitle.textContent = nav.querySelector('span:last-child').textContent;
}

// Sidebar nav clicks (only unlocked items)
navItems.forEach(n => {
    n.addEventListener('click', () => {
        if (!n.classList.contains('locked')) showPanel(n.dataset.panel);
    });
});

// "data-nav" buttons inside panels
document.addEventListener('click', e => {
    const btn = e.target.closest('[data-nav]');
    if (btn) showPanel(btn.dataset.nav);
});

function unlockNav(panelId) {
    const nav = document.querySelector(`[data-panel="${panelId}"]`);
    if (nav) nav.classList.remove('locked');
}

/* ── File Upload ────────────────────────────────────────────────────────────── */
const dropZone    = $('drop-zone');
const fileInput   = $('file-input');
const previewArea = $('file-preview-area');
const docTextarea = $('doc-textarea');
const btnIngest   = $('btn-ingest');

let uploadedFile = null;   // hold the actual File object for FormData

dropZone.addEventListener('click', () => fileInput.click());

['dragover', 'dragenter'].forEach(ev => {
    dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add('over'); });
});
['dragleave', 'dragend', 'drop'].forEach(ev => {
    dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove('over'); });
});
dropZone.addEventListener('drop', e => {
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => {
    if (e.target.files[0]) loadFile(e.target.files[0]);
});

function loadFile(file) {
    if (!/\.(md|txt|markdown|pdf)$/i.test(file.name)) {
        alert('Please upload a .md, .txt, or .pdf file.');
        return;
    }
    uploadedFile = file;
    $('preview-name').textContent = file.name;
    $('preview-size').textContent = (file.size / 1024).toFixed(1) + ' KB';
    dropZone.style.display = 'none';
    previewArea.classList.remove('hidden');
    btnIngest.disabled = false;

    if (/\.pdf$/i.test(file.name)) {
        // PDFs are binary — show a placeholder in the textarea
        docTextarea.value = `[PDF binary — "${file.name}" (${(file.size/1024).toFixed(1)} KB)]\n\nThe file will be sent to the server where pymupdf extracts text spans and infers headings by font size.`;
        docTextarea.readOnly = true;
    } else {
        docTextarea.readOnly = false;
        const reader = new FileReader();
        reader.onload = e => { docTextarea.value = e.target.result; };
        reader.readAsText(file);
    }
}

/* ── Stage 1 → 2: Ingest ─────────────────────────────────────────────────── */
const pipelineBadge = $('pipeline-badge');

btnIngest.addEventListener('click', async () => {
    if (!uploadedFile) return;

    btnIngest.disabled = true;
    $('btn-ingest-label').textContent = 'Processing…';
    $('btn-ingest-spinner').classList.remove('hidden');
    setBadge(pipelineBadge, 'running', 'INGESTING');

    try {
        const form = new FormData();
        form.append('file', uploadedFile);

        const res  = await fetch(`${API}/api/ingest`, { method: 'POST', body: form });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || 'Ingestion failed');

        renderIngestPanel(data);
        renderChunkPanel(data);

        // Refresh store status immediately so the warning banner clears
        await checkStoreStatus();

        setBadge(pipelineBadge, 'pass', 'INDEXED');
        unlockNav('panel-ingest');
        unlockNav('panel-chunk');
        unlockNav('panel-retrieve');
        showPanel('panel-ingest');

    } catch (err) {
        setBadge(pipelineBadge, 'fail', 'ERROR');
        alert(`❌ Ingestion error: ${err.message}`);
    } finally {
        btnIngest.disabled = false;
        $('btn-ingest-label').textContent = 'Process Document';
        $('btn-ingest-spinner').classList.add('hidden');
    }
});

/* ── Render: Ingest Panel ─────────────────────────────────────────────────── */
function renderIngestPanel(data) {
    // Stage banner
    const status = data.final_status === 'success' ? 'pass' : 'fail';
    const ingestStage = (data.stages || []).find(s => s.stage === 'ingestion') || {};
    const banner = $('stage-ingest-summary');
    banner.className = `stage-result ${status}`;
    banner.textContent = `Agent 1 — ingestion: ${status.toUpperCase()} | confidence: ${ingestStage.confidence ?? '—'} | ${data.escalation_reason || 'Parsed successfully'}`;

    // Outline = vectorless JSON tree
    const outline = data.result?.document?.doc_id
        ? (data.result.document.outline ?? [])
        : [];
    $('outline-json').textContent = fmt(outline);

    // Full document contract (strip block_tree for readability)
    const docObj = { ...(data.result?.document ?? {}) };
    delete docObj.block_tree;
    $('doc-contract-json').textContent = fmt(docObj);
}

/* ── Render: Chunk Panel ─────────────────────────────────────────────────── */
function renderChunkPanel(data) {
    const stages = data.stages || [];

    // Stage banner
    const chunkStage = stages.find(s => s.stage === 'chunk_validation') || {};
    const valStatus  = chunkStage.status === 'pass' ? 'pass' : 'fail';
    const banner = $('stage-chunk-summary');
    banner.className = `stage-result ${valStatus}`;
    banner.textContent = `Agent 3 — chunk_validation: ${valStatus.toUpperCase()} | confidence: ${chunkStage.confidence ?? '—'} | Chunks stored: ${data.result?.chunks_stored ?? '?'}, Quarantined: ${data.result?.chunks_quarantined ?? 0}`;

    // Chunk validation JSON (from stage log)
    const chunkStages = stages.filter(s => s.stage === 'chunking' || s.stage === 'chunk_validation');
    $('chunk-val-json').textContent = fmt(chunkStages);

    // NOTE: The pipeline doesn't return the raw chunks array in the ingest result
    // (they're stored in the vector store). We derive display info from stage metadata.
    const chunkList = $('chunk-list');
    const chunksStored = data.result?.chunks_stored ?? 0;
    if (chunksStored === 0) {
        chunkList.innerHTML = '<p style="color:var(--txt3);padding:8px">No chunks stored.</p>';
        return;
    }
    chunkList.innerHTML = '';
    // Show placeholder cards based on count (actual text is inside vector store)
    for (let i = 0; i < chunksStored; i++) {
        const card = document.createElement('div');
        card.className = 'chunk-card';
        card.innerHTML = `
            <div class="chunk-card-id">Chunk ${String(i).padStart(3,'0')}</div>
            <div class="chunk-card-meta">
                <span class="meta-pill">stored</span>
                <span class="meta-pill">embedded</span>
            </div>
        `;
        chunkList.appendChild(card);
    }
}

/* ── Stage 4: Hybrid Search ──────────────────────────────────────────────── */
$('btn-ask').addEventListener('click', async () => {
    // Refresh store status first
    await checkStoreStatus();

    if (!storeReady) {
        // Don't let the user hit a confusing LLM validator error
        showPanel('panel-upload');
        return;
    }

    const query = $('query-input').value.trim();
    if (!query) return;

    $('btn-ask').disabled = true;
    $('btn-ask-label').textContent = 'Searching…';
    $('btn-ask-spinner').classList.remove('hidden');
    setBadge(pipelineBadge, 'running', 'RETRIEVING');
    setBadge($('val2-badge'), 'running', 'RUNNING');
    $('final-answer').textContent = '';
    $('final-answer').className = 'answer-body dim';
    $('citations-area').classList.add('hidden');
    $('vector-results').textContent = 'Running dense vector search…';
    $('struct-results').textContent  = 'Running vectorless outline search…';

    try {
        const form = new FormData();
        form.append('query', query);

        const res  = await fetch(`${API}/api/ask`, { method: 'POST', body: form });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || 'Query failed');

        renderRetrievePanel(data);
        setBadge(pipelineBadge, 'pass', 'DONE');

    } catch (err) {
        setBadge(pipelineBadge, 'fail', 'ERROR');
        setBadge($('val2-badge'), 'fail', 'FAILED');
        $('final-answer').textContent = `Error: ${err.message}`;
        $('final-answer').className = 'answer-body';
    } finally {
        $('btn-ask').disabled = false;
        $('btn-ask-label').textContent = 'Run Search';
        $('btn-ask-spinner').classList.add('hidden');
    }
});

/* ── Render: Retrieve Panel ──────────────────────────────────────────────── */
function renderRetrievePanel(data) {
    const stages = data.stages || [];
    const valStage = stages.find(s => s.stage === 'retrieval_validation') || {};

    // Split retrieved chunks by source path
    const retrieved = data.result?.retrieval?.retrieved_chunks ?? [];
    const vChunks   = retrieved.filter(c => c.source_path === 'vector' || c.source_path === 'both');
    const sChunks   = retrieved.filter(c => c.source_path === 'vectorless' || c.source_path === 'both');

    $('vector-results').textContent  = vChunks.length ? fmt(vChunks) : 'No vector matches returned.';
    $('struct-results').textContent  = sChunks.length ? fmt(sChunks) : 'No vectorless matches returned.';

    // Validation badge + answer
    if (data.final_status === 'success' && data.result?.answer) {
        setBadge($('val2-badge'), 'pass', `PASS — conf: ${valStage.confidence ?? '?'}`);

        const ans = data.result.answer;
        $('final-answer').textContent = ans.answer;
        $('final-answer').className = 'answer-body';

        if (ans.citations && ans.citations.length) {
            const area = $('citations-area');
            area.innerHTML = '<div class="citations-hd">Citations</div>';
            ans.citations.forEach(c => {
                area.innerHTML += `
                    <div class="cit-row">
                        <span class="cit-ids">${(c.chunk_ids||[]).join(', ')}</span>
                        <span class="cit-claim">${c.claim}</span>
                    </div>`;
            });
            area.classList.remove('hidden');
        }

        if (ans.unanswered_parts && ans.unanswered_parts.length) {
            $('final-answer').textContent += '\n\nUnanswered gaps: ' + ans.unanswered_parts.join('; ');
        }
    } else {
        // Insufficient context or failed
        const reason = data.escalation_reason || 'Insufficient context — query refused to prevent hallucination.';
        setBadge($('val2-badge'), 'fail', 'FAIL');
        $('final-answer').textContent = reason;
        $('final-answer').className = 'answer-body';
    }
}
