// One app session. Commands are never replayed after a lost connection.
export class GuiSession {
 constructor(url, callbacks = {}) {
  this.url = url; this.callbacks = callbacks; this.pending = new Map();
  this.generation = 0; this.sequence = 0; this.attempt = 0; this.live = false;
 }
 start(key) {
  this.key = key; this.stopped = false;
  return new Promise((resolve, reject) => { this.initial = {resolve, reject}; this.open(); });
 }
 open() {
  if (this.stopped) return;
  this.authenticated = false;
  const generation = ++this.generation;
  const socket = this.socket = new WebSocket(this.url);
  const current = () => !this.stopped && generation === this.generation && socket === this.socket;
  this.callbacks.state?.(this.accepted ? 'reconnecting' : 'connecting');
  this.armWatchdog(generation, 18000);
  socket.onopen = () => { if (current()) socket.send(JSON.stringify({type: 'auth', apiKey: this.key})); };
  socket.onmessage = (event) => {
   if (!current()) return;
   let message;
   try { message = JSON.parse(event.data); } catch { this.lost(generation); return; }
   this.armWatchdog(generation, 55000);
   if (message.type === 'ready') {
    this.authenticated = true; this.accepted = true; this.readyMessage = message;
    this.callbacks.ready?.(message);

   } else if (message.type === 'heartbeat') {
    socket.send(JSON.stringify({type: 'pong'}));
   } else if (message.type === 'snapshot' && this.authenticated) {
    this.live = true; this.attempt = 0;
    this.callbacks.snapshot?.(message);
    this.callbacks.state?.('live');
    this.initial?.resolve(this.readyMessage); this.initial = null;
   } else if (message.type === 'reply' && this.live) {
    const request = this.pending.get(message.id);
    if (!request || request.generation !== generation) return;
    this.pending.delete(message.id); clearTimeout(request.timer);
    if (message.error) request.reject(Object.assign(new Error(message.error.message), {status: message.error.status}));
    else request.resolve(message.value);
   } else if (message.type === 'error') {
    const error = Object.assign(new Error(message.message || 'Session unavailable'), {status: message.status});
    if (message.status === 401 || message.status === 403) {
     this.stop(error); this.callbacks.rejected?.();
    } else if (!this.accepted) this.stop(error);
    else this.lost(generation);
   }
  };
  socket.onerror = () => { if (current()) this.lost(generation); };
  socket.onclose = () => { if (current()) this.lost(generation); };
 }
 armWatchdog(generation, delay) {
  clearTimeout(this.watchdog);
  this.watchdog = setTimeout(() => this.lost(generation), delay);
 }
 drain(error) {
  for (const request of this.pending.values()) { clearTimeout(request.timer); request.reject(error); }
  this.pending.clear();
 }
 lost(generation) {
  if (this.stopped || generation !== this.generation) return;
  this.generation++; this.live = false; clearTimeout(this.watchdog);
  this.socket?.close();
  const error = new Error('Connection lost. Command outcome may be unknown.');
  this.drain(error);
  if (!this.accepted) { this.stop(error); return; }
  this.callbacks.state?.('reconnecting');
  clearTimeout(this.retry);
  this.retry = setTimeout(() => this.open(), Math.min(15000, 500 * 2 ** this.attempt++));
 }
 request(type, payload = {}) {
  if (!this.live || this.stopped) return Promise.reject(new Error('The app connection is unavailable.'));
  if (this.pending.size >= 4) return Promise.reject(Object.assign(new Error('Wait for the current command to finish.'), {status: 429}));
  const id = `cmd-${++this.sequence}`, generation = this.generation;
  return new Promise((resolve, reject) => {
   const timer = setTimeout(() => this.lost(generation), 35000);
   this.pending.set(id, {resolve, reject, timer, generation});
   try { this.socket.send(JSON.stringify({id, type, payload})); }
   catch { this.lost(generation); }
  });
 }
 stop(error = new Error('Disconnected')) {
  this.stopped = true; this.live = false; this.generation++;
  clearTimeout(this.watchdog); clearTimeout(this.retry);
  this.socket?.close(); this.key = '';
  this.drain(error);
  this.initial?.reject(error); this.initial = null;
  this.callbacks.state?.('disconnected');
 }
}
