import assert from 'node:assert/strict';
import { test } from 'node:test';
import { GuiSession } from '../../src/subtitle_translator/static/session.mjs';

class Socket {
 static all = [];
 constructor(url) { this.url = url; this.sent = []; Socket.all.push(this); }
 send(data) { this.sent.push(JSON.parse(data)); }
 close() { this.closed = true; }
 receive(value) { this.onmessage({data: JSON.stringify(value)}); }
 ready() { this.onopen(); this.receive({type: 'ready', ownerScope: 'test-owner'}); }
 snapshot() { this.receive({type: 'snapshot', jobs: []}); }
}
function setup(t, callbacks = {}) {
 Socket.all = [];
 t.mock.method(globalThis, 'WebSocket', function(url) { return new Socket(url); });
 t.mock.timers.enable({apis: ['setTimeout']});
 const session = new GuiSession('ws://test/ui/session', callbacks);
 t.after(() => session.stop());
 return session;
}

test('authentication alone does not claim fresh progress', async t => {
 const states = [], session = setup(t, {state: value => states.push(value)});
 const started = session.start('synthetic-key'), socket = Socket.all[0];
 socket.ready();
 assert.equal(session.live, false);
 assert.equal(states.at(-1), 'connecting');
 socket.snapshot(); await started;
 assert.equal(session.live, true);
 assert.equal(states.at(-1), 'live');
 assert.equal(socket.url.includes('synthetic-key'), false);
});

test('lost commands reject, reconnect never replays, old callbacks cannot touch new socket', async t => {
 const session = setup(t), started = session.start('synthetic-key');
 const first = Socket.all[0]; first.ready(); first.snapshot(); await started;
 const command = session.request('submit', {submissionId: 'synthetic-file'});
 const rejected = assert.rejects(command, /Connection lost/);
 const commandId = first.sent.at(-1).id;
 first.onclose(); await rejected;
 assert.equal(session.pending.size, 0);
 t.mock.timers.tick(500);
 const second = Socket.all[1]; second.ready(); second.snapshot();
 first.receive({type: 'reply', id: commandId, value: {jobId: 'wrong'}});
 first.onclose();
 assert.equal(session.socket, second);
 assert.equal(session.live, true);
 assert.equal(Socket.all.flatMap(socket => socket.sent).filter(frame => frame.type === 'submit').length, 1);
});

test('a silent stream reconnects and explicit disconnect prevents further attempts', async t => {
 const session = setup(t), started = session.start('synthetic-key');
 const socket = Socket.all[0]; socket.ready(); socket.snapshot(); await started;
 t.mock.timers.tick(55000);
 assert.equal(session.live, false);
 assert.equal(socket.closed, true);
 session.stop(); t.mock.timers.tick(60000);
 assert.equal(Socket.all.length, 1);
 assert.equal(session.key, '');
});

test('temporary auth failure retains reconnect ability; rejected key stops it', async t => {
 let rejected = 0;
 const session = setup(t, {rejected: () => rejected++}), started = session.start('synthetic-key');
 const first = Socket.all[0]; first.ready(); first.snapshot(); await started;
 first.receive({type: 'error', status: 503});
 assert.equal(session.key, 'synthetic-key');
 t.mock.timers.tick(500);
 Socket.all[1].receive({type: 'error', status: 401});
 assert.equal(rejected, 1); assert.equal(session.key, '');
 t.mock.timers.tick(60000); assert.equal(Socket.all.length, 2);
});
