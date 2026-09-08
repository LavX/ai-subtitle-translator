import { GuiSession } from './session.mjs';
import { safeName, uniqueName, routeModel, zip, parseCues, latestFirst } from './archive.mjs';

const $ = (id) => document.getElementById(id);
const sessionUrl = new URL('./session', window.location.href);
sessionUrl.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
const storageKey = 'subtitle-translator.openrouter-key.v1';
const rows = [];
const terminal = new Set(['completed', 'partial', 'failed', 'cancelled']);
let connected = false, submitting = false, reading = false, token = '', epoch = 0, rowSequence = 0, session;
let ownerFingerprint = '', selectedRow = null, savedKeyAvailable = false, restoring = false;
let modelCatalog = [];
const forgotten = new Set();

function notify(message, error = false) {
 $('notice').textContent = message;
 $('notice').classList.toggle('error', error);
}
function storageButtons() {
 $('forget-saved-key').hidden = !savedKeyAvailable;
 $('use-saved-key').hidden = !savedKeyAvailable || connected;
}
function forgetStoredKey() {
 try { localStorage.removeItem(storageKey); savedKeyAvailable = false; storageButtons(); return true; }
 catch { notify('Browser storage is unavailable. Clear this site’s storage in your browser to remove any previously saved key.', true); return false; }
}
function rememberValidatedKey(value) {
 if (!$('remember-key').checked) return forgetStoredKey();
 try { localStorage.setItem(storageKey, value); savedKeyAvailable = true; storageButtons(); return true; }
 catch { notify('Connected for this session. Your browser could not save the key.', true); return false; }
}
function disconnect(message = 'Disconnected. Server jobs keep running. Reconnect with the same key to resume tracking.') {
 connected = false; submitting = false; restoring = false; token = ''; epoch++;
 $('api-key').value = '';
 session?.stop();
 $('connection-panel').classList.remove('connected');
 $('connection-heading').textContent = 'OpenRouter API key';
 $('connection-state').textContent = 'Not connected';
 $('connection-state').classList.remove('connected');
 $('connect').hidden = false; $('connect').disabled = false; $('disconnect').hidden = true;
 $('live-state').textContent = ''; $('api-key').disabled = false;
 storageButtons(); notify(message); render();
}
async function connect() {
 if (submitting || $('connect').disabled || connected) return;
 const candidate = $('api-key').value.trim();
 if (!candidate) { notify('Enter your OpenRouter API key.', true); return; }
 const version = ++epoch;
 const current = () => version === epoch;
 session?.stop();
 $('connect').disabled = true;
 session = new GuiSession(sessionUrl, {
  state: state => {
   if (!current()) return;
   $('live-state').textContent = {connecting: 'Connecting…', live: 'Live updates', reconnecting: 'Reconnecting… Jobs keep running.', disconnected: ''}[state];
   $('live-state').dataset.state = state; render();
  },
  ready: connection => {
   if (!current()) return;
   if (ownerFingerprint && ownerFingerprint !== connection.ownerScope) {
    rows.splice(0, rows.length, ...rows.filter(row => row.state === 'ready'));
    forgotten.clear(); selectedRow = null;
   }
   ownerFingerprint = connection.ownerScope; token = candidate; connected = true;
   $('api-key').value = ''; $('api-key').disabled = true;
   $('connect').hidden = true; $('disconnect').hidden = false;
   $('connection-panel').classList.add('connected'); $('connection-heading').textContent = 'OpenRouter key';
   $('connection-state').textContent = 'Key accepted'; $('connection-state').classList.add('connected');
   storageButtons();
  },
  snapshot: snapshot => { if (current()) mergeSnapshot(snapshot); },
  rejected: () => {
   if (!current()) return;
   forgetStoredKey(); disconnect('OpenRouter rejected this key. Enter a valid key to reconnect.');
  },
 });
 try {
  await session.start(candidate);
  if (!current()) return;
  notify('Jobs restore automatically. Add files to translate.');
  rememberValidatedKey(candidate); storageButtons(); render();
  const generation = session.generation;
  try {
   const catalog = await session.request('models');
   if (!current() || generation !== session.generation) return;
   modelCatalog = catalog.models || []; renderModels(); updateModelDetails();
  } catch { if (current()) notify('The model list is unavailable. You can enter a model ID.'); }
 } catch (error) {
  if (current()) notify(error.status === 503 ? 'OpenRouter key validation is temporarily unavailable. Try again shortly.' : error.status === 429 ? 'Key validation is rate limited. Wait a moment before connecting again.' : 'Could not connect. Check your key and try again.', true);
 } finally { if (current()) $('connect').disabled = false; }
}
$('connection-form').addEventListener('submit', (event) => { event.preventDefault(); void connect(); });
$('disconnect').addEventListener('click', () => disconnect(savedKeyAvailable ? 'Disconnected. Your saved key remains on this device. Choose Use saved key to reconnect, or Forget saved key to remove it.' : undefined));
$('forget-saved-key').addEventListener('click', () => {
 const removed = forgetStoredKey(); $('remember-key').checked = false;
 disconnect(removed ? 'Saved key removed. Enter a key to connect again.' : 'Disconnected. Clear this site’s storage in your browser to remove the saved key.');
});
$('remember-key').addEventListener('change', () => {
 if (!$('remember-key').checked) forgetStoredKey();
 else if (connected) rememberValidatedKey(token);
});
$('api-key').addEventListener('input', () => {
 if (savedKeyAvailable) forgetStoredKey();
 if ($('connect').disabled) { epoch++; session?.stop(); $('connect').disabled = false; }
});
function reuseSavedKey() {
 try {
  const saved = localStorage.getItem(storageKey);
  savedKeyAvailable = !!saved; storageButtons();
  if (saved) { $('api-key').value = saved; $('remember-key').checked = true; void connect(); }
 } catch { notify('Browser storage is unavailable. Enter a key for this session.', true); }
}
$('use-saved-key').addEventListener('click', reuseSavedKey);

function jobTiming(row) {
 const started = Date.parse((row.startedAt || '').replace(/([+-]\d\d:\d\d)Z$/, '$1'));
 if (!Number.isFinite(started)) return '';
 const seconds = Math.max(0, Math.floor((Date.now() - started) / 1000));
 return `${Math.floor(seconds / 60)}m ${seconds % 60}s elapsed`;
}
function noBatchFeedback(row) {
 const started = Date.parse((row.startedAt || '').replace(/([+-]\d\d:\d\d)Z$/, '$1'));
 const prolonged = Number.isFinite(started) && Date.now() - started >= 120000;
 const reportedActivity = /\b(request|retry|backoff|recover|timeout|rate.limit)/i.test(row.message || '');
 return `${prolonged ? 'No batch has completed for at least 2 minutes. Provider output is still unconfirmed.' : 'No completed batch yet.'}${reportedActivity ? '' : ' The server has not reported request activity.'}`;
}
function ordered() { return latestFirst(rows); }
function render() {
 const focusedAction = document.activeElement?.dataset?.action;
 $('files').replaceChildren();
 let cost = 0;
 for (const row of ordered()) {
  const li = document.createElement('li'); li.className = `file ${row.state}${previewRow() === row ? ' selected' : ''}`;
  if (previewRow() === row) li.setAttribute('aria-current', 'true');
  const text = document.createElement('div');
  const heading = document.createElement('h3'); heading.textContent = row.name; text.append(heading);
  const state = document.createElement('p'); state.className = 'state';
  state.textContent = row.state === 'partial' ? 'Partial: some cues remain untranslated' : row.state === 'ready' ? 'Awaiting submission' : row.state[0].toUpperCase() + row.state.slice(1);
  text.append(state);
  if (row.note) { const note = document.createElement('p'); note.textContent = row.note; text.append(note); }
  const duplicateError = terminal.has(row.state) && row.error && row.message && (row.message.includes(row.error) || row.error.includes(row.message));
  if (row.message && !duplicateError) { const message = document.createElement('p'); message.className = 'job-message'; message.textContent = row.message; text.append(message); }
  if (terminal.has(row.state) && row.error) { const error = document.createElement('p'); error.className = 'job-error'; error.textContent = row.error; text.append(error); }
  if (row.jobId) {
   const meta = document.createElement('p');
   const waiting = row.state === 'processing' && !(row.completedBatches > 0);
   const coverage = translatedCoverage(row);
   const detail = row.state === 'queued' ? 'Waiting for a worker' : row.state === 'partial' ? coverage ? `${coverage.done}/${coverage.total} cues translated (${Math.round(coverage.done / coverage.total * 100)}%)` : 'Translation incomplete' : row.state === 'failed' ? 'No complete translation' : waiting ? 'Waiting for first batch' : row.state === 'completed' ? coverage ? `${coverage.done}/${coverage.total} cues translated` : 'Translation complete' : `${Math.round(row.progress || 0)}% of batches processed`;
   const batches = !terminal.has(row.state) && row.totalBatches ? ` · ${row.completedBatches || 0}/${row.totalBatches} batches` : '';
   const elapsed = row.state === 'processing' ? jobTiming(row) : '';
   meta.textContent = `${detail}${batches}`;
   const clock = document.createElement('span'); clock.dataset.elapsed = row.key; clock.textContent = elapsed ? ` · ${elapsed}` : ''; meta.append(clock);
   if (Number.isFinite(row.cost)) meta.append(` · $${row.cost.toFixed(5)}`);
   if (waiting) {
    const explanation = document.createElement('p');
    explanation.dataset.waiting = row.key;
    explanation.textContent = noBatchFeedback(row);
    text.append(explanation);
   }
   text.append(meta);
   if (!terminal.has(row.state)) { const progress = document.createElement('progress'); progress.max = 100; if (row.state !== 'processing' || row.completedBatches > 0) progress.value = row.progress || 0; progress.setAttribute('aria-label', `${row.name} translation progress`); text.append(progress); }
  }
  if (Number.isFinite(row.cost)) cost += row.cost;
  const actions = document.createElement('div'); actions.className = 'file-actions';
  function action(label, handler, disabled = false) {
   const button = document.createElement('button'); button.type = 'button'; button.className = 'secondary'; button.textContent = label; button.dataset.action = `${row.key}:${label}`; button.disabled = disabled; button.addEventListener('click', handler); actions.append(button);
  }
  action('Preview', () => { selectedRow = row; $('cue-search').value = ''; render(); void hydrate(row); $('caption-preview').scrollIntoView({block: 'nearest', behavior: 'smooth'}); });
  if (downloadable(row)) action(row.state === 'partial' ? 'Download partial SRT' : 'Download SRT', () => downloadOne(row));
  if (row.state === 'queued' || row.state === 'processing') action(row.state === 'queued' ? 'Cancel queued job' : 'Cancel', () => cancel(row), !session?.live || row.cancelling);
  if (!['submitting', 'processing', 'queued', 'finishing'].includes(row.state)) action(row.state === 'ready' ? 'Remove' : 'Forget', async () => {
   if (row.jobId && connected && session?.live) {
    try { await session.request('forget', {jobId: row.jobId}); }
    catch (error) { if (error?.status !== 404) { notify('The job could not be removed from the service and stays in your history.', true); return; } }
   }
   if (row.jobId) forgotten.add(row.jobId);
   rows.splice(rows.indexOf(row), 1); if (selectedRow === row) selectedRow = null; render();
  });
  li.append(text, actions); $('files').append(li);
 }
 if (focusedAction) {
  const replacement = [...$('files').querySelectorAll('button')].find(button => button.dataset.action === focusedAction);
  replacement?.focus({preventScroll: true});
 }
 $('empty-queue').hidden = rows.length > 0;
 $('upload-panel').classList.toggle('has-files', rows.length > 0);
 const states = ['processing', 'queued', 'completed', 'partial', 'failed'].map(state => { const count = rows.filter(row => row.state === state).length; return count ? `${count} ${state === 'processing' ? 'running' : state}` : ''; }).filter(Boolean);
 $('queue-summary').textContent = states.join(' · ');
 updatePreview();
 const awaiting = rows.filter(row => row.state === 'ready').length;
 const active = rows.filter(row => ['queued', 'submitting', 'processing', 'finishing'].includes(row.state)).length;
 const downloads = rows.filter(downloadable), partialDownloads = downloads.filter(row => row.state === 'partial').length;
 $('file-count').textContent = rows.length ? `${rows.length} file${rows.length === 1 ? '' : 's'} · ${awaiting} awaiting submission · ${active} active · ${downloads.length} downloadable${partialDownloads ? ` (${partialDownloads} partial)` : ''}` : 'No files added';
 $('total-cost').textContent = `Reported API cost: $${cost.toFixed(4)}`;
 $('translate').disabled = !session?.live || submitting || reading || !!reasoningIssue() || !rows.some(r => r.state === 'ready');
 $('translate-label').textContent = submitting ? 'Submitting files…' : 'Translate files';
 $('download-all').disabled = !rows.some(downloadable);
 $('restore-jobs').disabled = !session?.live || restoring;
 $('dropzone').disabled = reading;
}
async function addFiles(files) {
 if (reading) return;
 reading = true; render();
 const errors = [];
 try {
  for (const file of files) {
   const signature = `${file.name}\0${file.size}\0${file.lastModified}`;
   if (rows.some(row => row.signature === signature)) continue;
   if (rows.length >= 100) { errors.push('The queue is limited to 100 files.'); break; }
   if (rows.reduce((total, row) => total + row.size, 0) + file.size > 20000000) { errors.push('The queue is limited to 20 MB.'); break; }
   if (!/\.srt$/i.test(file.name) || !file.size) { errors.push(`${safeName(file.name)}: choose a nonempty .srt file.`); continue; }
   if (file.size > 8000000) { errors.push(`${safeName(file.name)}: file is too large.`); continue; }
   try {
    const content = new TextDecoder('utf-8', {fatal: true}).decode(await file.arrayBuffer());
    if (!content.trim() || content.length > 2000000) { errors.push(`${safeName(file.name)}: file must contain 1 to 2,000,000 characters.`); continue; }
    rows.push({key: ++rowSequence, name: safeName(file.name), size: file.size, signature, content, state: 'ready', addedAt: Date.now()});
   } catch { errors.push(`${safeName(file.name)}: could not read UTF-8 text.`); }
  }
 } finally { reading = false; render(); }
 notify(errors.length ? errors.join(' ') : 'Files added. Check the settings, then translate.', errors.length > 0);
}
$('upload-panel').addEventListener('click', (event) => {
 // Use the dispatch-time path: a click inside an interactive region may re-render that region
 // before bubbling reaches the panel, leaving the target detached and closest() blind.
 const insideInteractive = event.composedPath().some(node => node instanceof Element && node.hasAttribute('data-preview-interactive'));
 if (!reading && event.target !== $('file-input') && !insideInteractive) $('file-input').click();
});
$('file-input').addEventListener('change', () => { const files = [...$('file-input').files]; $('file-input').value = ''; void addFiles(files); });
let dragDepth = 0;
const uploadPanel = $('upload-panel');
uploadPanel.addEventListener('dragenter', (event) => {
 if (!event.dataTransfer.types.includes('Files')) return;
 event.preventDefault(); dragDepth++;
 if (!reading) uploadPanel.classList.add('dragging');
});
uploadPanel.addEventListener('dragover', (event) => {
 if (!event.dataTransfer.types.includes('Files')) return;
 event.preventDefault(); event.dataTransfer.dropEffect = reading ? 'none' : 'copy';
});
uploadPanel.addEventListener('dragleave', () => {
 dragDepth = Math.max(0, dragDepth - 1);
 if (!dragDepth) uploadPanel.classList.remove('dragging');
});
uploadPanel.addEventListener('drop', (event) => {
 event.preventDefault(); dragDepth = 0; uploadPanel.classList.remove('dragging');
 void addFiles([...event.dataTransfer.files]);
});
window.addEventListener('dragover', (e) => { if (e.dataTransfer.types.includes('Files')) e.preventDefault(); });
window.addEventListener('drop', (e) => { if (e.dataTransfer.types.includes('Files')) e.preventDefault(); });

$('translate').addEventListener('click', async () => {
 if (!session?.live || submitting || reading) return;
 const sourceLanguage = $('source-language').value.trim(), targetLanguage = $('target-language').value.trim(), model = $('model').value.trim();
 if (!sourceLanguage || !targetLanguage || sourceLanguage.length > 50 || targetLanguage.length > 50 || !model) { notify('Enter source and target language codes and a model ID.', true); return; }
 const issue = reasoningIssue();
 if (issue) { notify(issue, true); return; }
 const provider = $('provider-only').value.trim().toLowerCase();
 if (provider && !/^[a-z0-9][a-z0-9/._-]{0,99}$/.test(provider)) { notify('Enter one OpenRouter provider ID, such as azure, or leave Provider empty.', true); return; }
 $('model').value = routeModel(model, $('routing').value);
 const settings = {sourceLanguage, targetLanguage, title: $('media-title').value.trim(), config: {model: routeModel(model, $('routing').value), requestTimeout: Number($('request-timeout').value), provider: {sort: $('routing').value}}};
 if ($('service-tier').value !== 'auto') settings.config.serviceTier = $('service-tier').value;
 if (provider) { settings.config.provider.only = [provider]; settings.config.provider.allowFallbacks = false; }
 if ($('reasoning').value !== 'default') settings.config.reasoning = {effort: $('reasoning').value};

 const batch = rows.filter(row => row.state === 'ready');
 const version = epoch; let submittedCount = 0, submissionFailed = false;
 submitting = true; notify('Submitting your subtitle files…'); render();
 try {
  for (const row of batch) {
   if (!session?.live || version !== epoch) break;
   if (!rows.includes(row)) continue;
   row.submissionId ||= crypto.randomUUID?.() || `file-${Date.now()}-${++rowSequence}`;
   row.state = 'submitting'; row.target = targetLanguage; render();
   try {
    const result = await session.request('submit', {submissionId: row.submissionId, request: {...settings, title: settings.title || row.name.replace(/\.srt$/i, ''), content: row.content, fileName: row.name, jobName: `${row.name} (${targetLanguage})`}});
    if (version !== epoch || !rows.includes(row)) return;
    if (!result.jobId) throw new Error('Missing job ID');
    submittedCount++;
    row.jobId = result.jobId;
    if (row.state === 'submitting' || row.state === 'unknown') { row.state = result.status; row.progress = 0; row.note = ''; }
   } catch (error) {
    if (version !== epoch || !rows.includes(row)) return;
    if (row.jobId) { if (!session.live) break; continue; }
    submissionFailed = true;
    if (error.status && error.status < 500) {
     row.state = 'rejected';
     row.note = error.status === 429 ? 'Queue full. No automatic retry.' : error.status === 401 ? 'OpenRouter rejected the key. No automatic retry.' : `Request rejected (${error.status}). Check file, settings and API key requirements.`;
    } else {
     row.state = 'unknown'; row.note = 'Submission status unknown. A server job may exist. Reconnecting will check it automatically. No automatic retry.';
    }
    notify(row.note, true); break;
   } finally { if (version === epoch) render(); }
  }
 } finally {
  if (version !== epoch) return;
  submitting = false;
  if (connected && version === epoch && submittedCount && !submissionFailed && !rows.some(row => ['partial', 'failed', 'unknown', 'rejected'].includes(row.state))) notify(`Submitted ${submittedCount} file${submittedCount === 1 ? '' : 's'}. Tracking translation progress below.`);
  render();
 }
});
function translatedCoverage(row) {
 if (row.state === 'partial') {
  const matched = /^(\d+)\/(\d+) lines translated/.exec(row.error || '');
  if (matched && Number(matched[2]) > 0) return {done: Number(matched[1]), total: Number(matched[2])};
  return null;
 }
 if (row.totalLines > 0) return {done: row.state === 'completed' ? row.totalLines : row.completedLines || 0, total: row.totalLines};
 return null;
}
function failureReason(row) {
 const error = row.error || '';
 if (/timed? out|timeout/i.test(error)) return 'Provider requests timed out. For a new attempt, try smaller batches or a faster route.';
 if (/429|rate.limit/i.test(error)) return 'OpenRouter rate limit reached. Wait before starting another attempt.';
 if (/402|credit|balance/i.test(error)) return 'OpenRouter reported insufficient credit.';
 if (/401|403|api.key/i.test(error)) return 'OpenRouter rejected the translation request. Check your key permissions.';
 return row.state === 'partial' ? 'Some batches failed. Untranslated cues remain in the original language.' : 'The provider could not finish this translation. Try another model or route.';
}
function applyJob(row, result) {
 row.revision = (row.revision || 0) + 1;
 row.state = result.status; row.progress = result.progress;
 row.startedAt = result.startedAt; row.createdAt = result.createdAt; row.completedAt = result.completedAt; row.totalBatches = result.totalBatches; row.completedBatches = result.completedBatches;
 row.totalLines = result.totalLines; row.completedLines = result.completedLines; row.error = result.error;
 row.message = typeof result.message === 'string' ? result.message : '';
 row.cost = result.totalCost;
 if ('result' in result) row.result = result.result;
 if ('hasResult' in result) row.hasResult = result.hasResult;
 row.note = ['failed', 'partial'].includes(row.state) ? failureReason(row) : '';
}
function mergeSnapshot(snapshot) {
 for (const job of snapshot.jobs || []) {
  if (forgotten.has(job.jobId)) continue;
  let row = rows.find(item => item.jobId === job.jobId || (job.submissionId && item.submissionId === job.submissionId));
  if (!row) {
   row = {key: ++rowSequence, name: safeName(job.fileName || job.jobName || 'subtitle.srt'), size: 0, signature: `remote:${job.jobId}`, content: '', restored: true};
   rows.push(row);
  }
  const previous = row.state;
  row.reconciledGeneration = null;
  row.jobId = job.jobId; row.submissionId = job.submissionId; row.target = job.targetLanguage || 'translated';
  applyJob(row, job);
  if (previous && previous !== row.state && terminal.has(row.state)) {
   notify(row.state === 'completed' ? `${row.name} is ready to download.` : `${row.name}: ${failureReason(row)}`, row.state !== 'completed');
  }
 }
 render();
 const row = previewRow();
 if (row && !row.hydrationFailed && ((!row.content && row.jobId) || (row.hasResult && !row.result))) void hydrate(row);
 void reconcileMissing(snapshot);
}
async function reconcileMissing(snapshot) {
 const version = epoch, generation = session.generation;
 // A snapshot covers loaded history. Resolve missing rows explicitly, without polling.
 for (const row of rows.filter(row => row.jobId && !snapshot.jobs.some(job => job.jobId === row.jobId))) {
  if (version !== epoch || generation !== session.generation) return;
  if (!row.reconciledGeneration || row.reconciledGeneration !== session.generation) {
   row.reconciledGeneration = session.generation;
   await hydrate(row, false, true);
  }
 }
}
async function hydrate(row, source = true, statusOnly = false) {
 if (!session?.live || !row.jobId || !rows.includes(row)) return;
 if (row.hydrating) { await row.hydrating; if (statusOnly) return hydrate(row, false, true); return; }
 const version = epoch, generation = session.generation;
 const current = () => version === epoch && generation === session.generation && rows.includes(row);
 row.hydrating = (async () => {
  try {
   if (source && !row.content) {
    const original = await session.request('source', {jobId: row.jobId});
    if (!current()) return;
    row.content = original.content || '';
   }
   if (statusOnly || (row.hasResult && !row.result?.content)) {
    const revision = row.revision;
    const result = await session.request(statusOnly ? 'status' : 'job', {jobId: row.jobId});
    if (!current()) return;
    if (statusOnly && row.revision === revision) applyJob(row, result);
    if (result.result) row.result = result.result;
   }
   row.hydrationFailed = false;
  } catch (error) {
   if (!current()) return;
   row.hydrationFailed = true;
   if (error.status === 404) { row.state = 'unavailable'; row.note = 'Job no longer available on this service.'; row.hasResult = false; }
   else notify('Could not load this file. Select Preview or Download to try again.', true);
  } finally { row.hydrating = null; if (current()) render(); }
 })();
 return row.hydrating;
}
async function cancel(row) {
 if (!connected || !['queued', 'processing'].includes(row.state) || row.cancelling) return;
 const version = epoch;
 const generation = session.generation;
 const current = () => session.live && epoch === version && generation === session.generation && rows.includes(row);
 row.cancelling = true; render();
 try {
  const result = await session.request('cancel', {jobId: row.jobId});
  if (!current()) return;
  if (result.status === 'cancelled') {
   if (!terminal.has(row.state)) row.state = 'cancelled';
   row.note = ''; notify('Job cancelled.');
  } else if (result.status === 'cancelling') {
   row.note = 'Cancelling. The job stops after its current request.';
   notify('Cancellation requested. The job stops after its current request.');
  } else if (result.status === 'processing') {
   if (!terminal.has(row.state)) row.state = 'processing';
   row.note = 'Already processing. This job was not cancelled.';
   notify('This job started processing and cannot be cancelled.');
  } else if (terminal.has(result.status)) {
   if (!terminal.has(row.state)) row.state = 'finishing';
   row.note = 'The job already finished. Retrieving its result.';
   const latest = await session.request('job', {jobId: row.jobId});
   if (!current()) return;
   applyJob(row, latest);
   notify(`The job is already ${latest.status}. Its server record was preserved.`);
  } else {
   notify('Cancellation was not confirmed. Progress tracking will check the job.', true);
  }
 } catch { if (current()) notify('Cancellation could not be confirmed. Progress tracking will check the job.', true); }
 finally { row.cancelling = false; if (current()) render(); }
}
function downloadable(row) { return (row.state === 'completed' || row.state === 'partial') && (row.hasResult || typeof row.result?.content === 'string' && row.result.content.length > 0); }
function outputName(row) {
 return safeName(`${row.name.replace(/\.srt$/i, '').slice(0, 100)}${row.state === 'partial' ? '.partial' : ''}.${safeName(row.target)}.srt`);
}
function save(blob, name) {
 const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = name;
 document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
}
async function downloadOne(row) {
 const version = epoch;
 await hydrate(row, false);
 if (version === epoch && rows.includes(row) && row.result?.content) save(new Blob([row.result.content], {type: 'application/x-subrip;charset=utf-8'}), outputName(row));
}
$('download-all').addEventListener('click', async () => {
 const version = epoch, used = new Set(), files = [];
 $('download-all').disabled = true;
 for (const row of rows.filter(downloadable)) {
  await hydrate(row, false);
  if (version !== epoch) return;
  if (!rows.includes(row) || !row.result?.content) { notify('ZIP could not be completed. Try downloading again.', true); render(); return; }
  files.push({name: uniqueName(outputName(row), used), content: row.result.content});
 }
 if (files.length) save(new Blob([zip(files)], {type: 'application/zip'}), 'translated-subtitles.zip');
 render();
});
function previewRow() { return selectedRow && rows.includes(selectedRow) ? selectedRow : rows[0]; }
function updatePreview() {
 const row = previewRow();
 const sourceContent = row?.content || '', translatedContent = row?.result?.content || '';
 if (row && (!row.preview || row.preview.sourceContent !== sourceContent || row.preview.translatedContent !== translatedContent)) {
  const source = parseCues(sourceContent), translated = parseCues(translatedContent);
  row.preview = {sourceContent, translatedContent, source, translated, byTime: new Map(translated.map(cue => [cue.key, cue]))};
 }
 const cues = row ? (row.preview.source.length ? row.preview.source : row.preview.translated) : [];
 const index = Math.max(0, Math.min(row?.cueIndex || 0, cues.length - 1));
 if (row) row.cueIndex = index;
 const cue = cues[index], hasSource = Boolean(row?.preview.source.length);
 const translated = cue && row.preview.byTime.get(cue.key);
 $('preview-label').textContent = row ? row.name : 'Subtitle preview';
 $('source-caption').hidden = !cue || !hasSource;
 $('preview-source').hidden = Boolean(cue && !hasSource);
 renderSubtitle($('preview-source'), cue && hasSource ? cue.text : row ? row.jobId && !row.content ? 'Select Preview to load this file.' : 'No readable subtitle cues found.' : 'Drop your subtitles here');
 $('translated-caption').hidden = !translated;
 $('translated-caption').textContent = row?.target ? `Translation · ${row.target}` : 'Translation';
 $('preview-translated').hidden = !translated;
 renderSubtitle($('preview-translated'), translated?.text || '');
 $('original-pane').hidden = Boolean(cue && !hasSource);
 $('translated-pane').hidden = !cue;
 $('translated-caption').hidden = false;
 $('translation-placeholder').hidden = Boolean(translated);
 $('translation-placeholder').textContent = row?.state === 'failed' ? 'Translation failed.' : row?.state === 'processing' ? 'Translation in progress…' : 'Translate this file to compare the result.';
 $('preview-caption').textContent = !row ? 'Add an SRT file to inspect its subtitles.' : row.state === 'partial' ? `Partial result: ${translatedCoverage(row) ? `${translatedCoverage(row).done}/${translatedCoverage(row).total} cues translated. ` : ''}Some cues still contain original text.` : !hasSource && cue ? 'Translated file. Original text is unavailable for restored jobs.' : translatedContent && cue && !translated ? 'No translation found for this timestamp.' : !translatedContent && cue ? 'Original subtitles. Translation will appear here when ready.' : '';
 $('preview-caption').hidden = !$('preview-caption').textContent;
 $('preview-controls').hidden = !cue;
 $('previous-cue').disabled = index === 0;
 $('next-cue').disabled = index >= cues.length - 1;
 $('cue-position').textContent = cue ? `Cue ${index + 1} of ${cues.length}` : '';
 $('cue-time').textContent = cue?.time || '';
 $('cue-slider').max = Math.max(1, cues.length); $('cue-slider').value = index + 1;
 $('cue-slider').disabled = cues.length < 2;
 $('cue-slider').setAttribute('aria-valuetext', `Cue ${index + 1} of ${cues.length}`);
 $('cue-jump').max = Math.max(1, cues.length); $('cue-jump').value = index + 1;
 renderCueBrowser(row, cues, index);
}
function changeCue(index) { const row = previewRow(); if (row) { row.cueIndex = index; updatePreview(); } }
$('previous-cue').addEventListener('click', () => changeCue((previewRow()?.cueIndex || 0) - 1));
$('next-cue').addEventListener('click', () => changeCue((previewRow()?.cueIndex || 0) + 1));
$('cue-slider').addEventListener('input', (event) => changeCue(Number(event.target.value) - 1));

$('restore-jobs').addEventListener('click', async () => {
 if (!session?.live || restoring) return;
 const version = epoch; restoring = true; forgotten.clear(); render();
 try {
  await session.request('restore');
  if (version === epoch) notify('Job list refreshed. No translations were resubmitted.');
 } catch { if (version === epoch) notify('Could not refresh jobs. Reconnect to try again.', true); }
 finally { if (version === epoch) { restoring = false; render(); } }
});

function renderModels() {
 $('models').replaceChildren();
 for (const model of modelCatalog) {
  const option = document.createElement('option');
  option.value = routeModel(model.id, $('routing').value); option.label = model.name;
  $('models').append(option);
 }
}
$('routing').addEventListener('change', () => {
 $('model').value = routeModel($('model').value, $('routing').value); renderModels(); updateModelDetails();
 updateTierHint();
});

function updateTierHint() {
 $('service-tier-hint').textContent = $('service-tier').value === 'default'
  ? 'Standard keeps the provider on its normal processing queue even with lowest-price routing. Follow routing lets a :floor route use a provider\u2019s cheaper Flex queue, which can wait minutes per request.'
  : $('routing').value === 'floor'
   ? 'Allows discounted Flex capacity. Requests can be much slower or time out.'
   : $('routing').value === 'nitro'
    ? 'Allows priority capacity, which can cost more than standard pricing.'
    : 'Uses OpenRouter’s normal capacity selection.';
}
$('service-tier').addEventListener('change', updateTierHint);

function selectedModel() {
 const id = $('model').value.trim().replace(/:(floor|nitro)$/, '');
 return modelCatalog.find(item => item.id === id);
}
function reasoningEfforts() {
 return (selectedModel()?.reasoning?.supportedEfforts || []).filter(effort => ['minimal', 'low', 'medium', 'high', 'xhigh'].includes(effort));
}
function reasoningIssue() {
 const effort = $('reasoning').value;
 if (effort === 'none' && (selectedModel()?.reasoning?.mandatory === true || /:thinking(?::(?:floor|nitro))?$/.test($('model').value.trim()))) return 'This model requires reasoning. Choose Model default or a supported effort, or select another model.';
 if (!['none', 'default'].includes(effort) && !reasoningEfforts().includes(effort)) return 'The selected reasoning effort is not supported by this model’s catalog metadata. Choose another option.';
 return '';
}
function updateReasoningOptions() {
 const selected = $('reasoning').value || 'none';
 const choices = [['none', 'Off'], ['default', 'Model default'], ...reasoningEfforts().map(effort => [effort, effort[0].toUpperCase() + effort.slice(1)])];
 if (!choices.some(([value]) => value === selected)) choices.push([selected, `${selected} (unavailable)`]);
 $('reasoning').replaceChildren();
 for (const [value, label] of choices) {
  const option = document.createElement('option'); option.value = value; option.textContent = label;
  $('reasoning').append(option);
 }
 $('reasoning').value = selected;
 const issue = reasoningIssue();
 if (issue) document.querySelector('.request-options').open = true;
 $('reasoning').setAttribute('aria-invalid', String(!!issue));
 $('reasoning-hint').textContent = issue || (selected === 'default' ? 'Model default leaves reasoning settings to the provider.' : selected === 'none' ? 'Off explicitly disables reasoning for subtitle translation.' : `Requests ${selected} reasoning effort.`);
}
$('reasoning').addEventListener('change', () => { updateReasoningOptions(); render(); });

function renderSubtitle(element, value) {
 element.replaceChildren();
 const stack = [element];
 for (const part of value.split(/(<\/?(?:i|b|u)>)/gi)) {
  const tag = /^<(i|b|u)>$/i.exec(part), closing = /^<\/(i|b|u)>$/i.exec(part);
  if (tag) { const node = document.createElement(tag[1].toLowerCase()); stack.at(-1).append(node); stack.push(node); }
  else if (closing && stack.length > 1 && stack.at(-1).tagName.toLowerCase() === closing[1].toLowerCase()) stack.pop();
  else stack.at(-1).append(document.createTextNode(part));
 }
}
function renderCueBrowser(row, cues, index) {
 const query = $('cue-search').value.trim().toLocaleLowerCase();
 const matches = cues.map((cue, position) => ({cue, position})).filter(({cue}) => !query || `${cue.text} ${row?.preview.byTime.get(cue.key)?.text || ''}`.toLocaleLowerCase().includes(query));
 const visible = query ? matches.slice(0, 50) : matches.slice(Math.max(0, index - 2), index + 4);
 $('cue-results').replaceChildren();
 for (const {cue, position} of visible) {
  const item = document.createElement('li'), button = document.createElement('button'); button.type = 'button';
  const number = document.createElement('span'), text = document.createElement('span');
  number.textContent = String(position + 1).padStart(2, '0'); text.textContent = cue.text.replace(/<\/?(?:i|b|u)>/gi, '');
  button.append(number, text); button.setAttribute('aria-current', String(position === index));
  button.addEventListener('click', () => changeCue(position)); item.append(button); $('cue-results').append(item);
 }
 $('cue-browser-count').textContent = query ? `${matches.length} matches` : `${cues.length} cue${cues.length === 1 ? '' : 's'}`;
 $('cue-search-status').textContent = query && !matches.length ? 'No matching cues.' : query && matches.length > 50 ? 'Showing the first 50 matches. Narrow your search to see more.' : '';
}
$('cue-search').addEventListener('input', updatePreview);
$('cue-jump').addEventListener('change', (event) => { const index = Number(event.target.value); if (Number.isInteger(index) && index > 0) changeCue(index - 1); else updatePreview(); });
$('swap-languages').addEventListener('click', () => { const source = $('source-language').value; $('source-language').value = $('target-language').value; $('target-language').value = source; });
$('caption-preview').addEventListener('keydown', (event) => {
 if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); changeCue((previewRow()?.cueIndex || 0) + (event.key === 'ArrowRight' ? 1 : -1)); }
});
$('caption-preview').tabIndex = 0;
$('caption-preview').setAttribute('aria-label', 'Subtitle comparison. Use left and right arrow keys to browse cues.');

function updateModelDetails() {
 const model = selectedModel();
 updateReasoningOptions(); render();
 $('model-details').hidden = !model; $('model-details').replaceChildren();
 if (!model) return;
 const name = document.createElement('strong'); name.textContent = model.name; $('model-details').append(name);
 const input = Number(model.pricing?.prompt), output = Number(model.pricing?.completion);
 if (Number.isFinite(input) && Number.isFinite(output)) { const price = document.createElement('span'); price.textContent = `Catalog price / 1M tokens: $${(input * 1e6).toFixed(3)} in · $${(output * 1e6).toFixed(3)} out`; $('model-details').append(price); }
}
$('model').addEventListener('input', updateModelDetails);

render();
reuseSavedKey();
setInterval(() => {
 for (const row of rows.filter(row => row.state === 'processing')) {
  const clock = document.querySelector(`[data-elapsed="${row.key}"]`);
  if (clock) clock.textContent = ` · ${jobTiming(row)}`;
  const waiting = document.querySelector(`[data-waiting="${row.key}"]`);
  if (waiting) waiting.textContent = noBatchFeedback(row);
 }
}, 1000);
