import test from 'node:test';
import assert from 'node:assert/strict';
import { renderBackends } from '../../src/subtitle_translator/server/static/backends.mjs';

test('renders back ends, errors, sign-in commands and connection state', () => {
 const html = renderBackends({
  backends: [
   {type: 'openrouter', name: 'OpenRouter', source: 'built-in', available: true, error: null, usesSignIn: false, loginCommand: null},
   {type: 'signin', name: 'Harness', source: 'pkg 0.1.0', available: true, error: null, usesSignIn: true, loginCommand: 'docker exec -it ai-subtitle-translator translator-login signin'},
   {type: 'broken', name: 'broken', source: 'other 1.0', available: false, error: 'ImportError: missing', usesSignIn: false, loginCommand: null},
  ],
  connections: [
   {id: 'signin', type: 'signin', enabled: true, model: 'm', signedIn: false, keys: []},
   {id: 'gemini', type: 'gemini', enabled: true, model: 'gemini-3.6-flash', signedIn: null, keys: [{index: 1, state: 'cooling', until: 1900000000}]},
  ],
 });
 assert.match(html, /translator-login signin/);
 assert.match(html, /Signed out/);
 assert.match(html, /ImportError: missing/);
 assert.match(html, /Key 1: cooling/);
});

test('escapes everything it prints', () => {
 const html = renderBackends({backends: [{type: 'x', name: '<img src=x onerror=alert(1)>', source: 'built-in', available: true, error: null, usesSignIn: false, loginCommand: null}], connections: []});
 assert.doesNotMatch(html, /<img/);
 assert.match(html, /&lt;img/);
});
