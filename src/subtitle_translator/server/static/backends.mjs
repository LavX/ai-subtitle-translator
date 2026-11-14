const escape = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const when = (seconds) => seconds ? new Date(seconds * 1000).toLocaleTimeString() : '';

export function renderBackends(status) {
 const backends = (status?.backends || []).map(b => `<li class="backend${b.available ? '' : ' unavailable'}">`
  + `<strong>${escape(b.name)}</strong> <small>${escape(b.source)}</small>`
  + (b.available ? '' : ` <span class="backend-error">${escape(b.error)}</span>`)
  + (b.usesSignIn && b.loginCommand ? `<div class="login-command">Sign in: <code>${escape(b.loginCommand)}</code></div>` : '')
  + '</li>').join('');
 const connections = (status?.connections || []).map(c => `<li class="connection${c.enabled ? '' : ' off'}">`
  + `<strong>${escape(c.id)}</strong> <span>${escape(c.model)}</span> <span>${c.enabled ? 'On' : 'Off'}</span>`
  + (c.signedIn === null || c.signedIn === undefined ? '' : ` <span>${c.signedIn ? 'Signed in' : 'Signed out'}</span>`)
  + (c.keys || []).map(k => ` <span class="key-state">Key ${escape(k.index)}: ${escape(k.state)}${k.until ? ` until ${escape(when(k.until))}` : ''}</span>`).join('')
  + '</li>').join('');
 return `<h3>Installed back ends</h3><ul class="backend-list">${backends || '<li>None</li>'}</ul>`
  + `<h3>Server back ends, in order</h3><ol class="connection-list">${connections || '<li>None configured</li>'}</ol>`;
}
