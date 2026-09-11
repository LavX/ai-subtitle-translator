import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { crc32, safeName, uniqueName, floorModel, routeModel, routingConfig, zip, latestFirst } from '../../src/subtitle_translator/static/archive.mjs';
test('selected routing overrides pasted shortcuts and preserves model variants', () => {
 assert.equal(routeModel('foo/bar:floor','nitro'), 'foo/bar:nitro');
 assert.equal(routeModel('foo/bar:nitro','default'), 'foo/bar');
 assert.equal(routeModel('foo/bar:free','nitro'), 'foo/bar:free');
 assert.equal(routeModel('foo/bar','floor'), 'foo/bar:floor');
});
test('SmartFast routing preserves exact free and thinking variants', () => {
 assert.equal(routeModel('  foo/bar:free  ', 'smartfast'), 'foo/bar:free:smartfast');
 assert.equal(routeModel('foo/bar:thinking', 'smartfast'), 'foo/bar:thinking:smartfast');
 assert.equal(routeModel('foo/bar:floor', 'smartfast'), 'foo/bar:smartfast');
 assert.equal(routeModel('foo/bar:smartfast', 'smartfast'), 'foo/bar:smartfast');
});
test('leaving SmartFast restores selected routing without losing the model variant', () => {
 assert.equal(routeModel('foo/bar:smartfast', 'floor'), 'foo/bar:floor');
 assert.equal(routeModel('foo/bar:smartfast', 'nitro'), 'foo/bar:nitro');
 assert.equal(routeModel('foo/bar:smartfast', 'default'), 'foo/bar');
 assert.equal(routeModel('foo/bar:free:smartfast', 'floor'), 'foo/bar:free');
 assert.equal(routeModel('foo/bar:thinking:smartfast', 'nitro'), 'foo/bar:thinking');
});
test('capability lookup removes the local route while retaining the exact variant', () => {
 const catalog = [{id: 'foo/bar', mandatory: false}, {id: 'foo/bar:thinking', mandatory: true}, {id: 'foo/bar:free', price: 0}];
 assert.equal(catalog.find(item => item.id === routeModel('foo/bar:thinking:smartfast', 'default')).mandatory, true);
 assert.equal(catalog.find(item => item.id === routeModel('foo/bar:free:smartfast', 'default')).price, 0);
});
test('SmartFast request config submits numeric tuning values including zero ceilings', () => {
 assert.deepEqual(routingConfig('foo/bar:free', 'smartfast', {
  medianPremiumPercent: '75.5', speedTolerancePercent: '0', sparsePremiumMultiplier: '2', maxPromptPrice: '0', maxCompletionPrice: '0',
 }), {model: 'foo/bar:free:smartfast', provider: {sort: 'smartfast', smartFast: {
  medianPremiumPercent: 75.5, speedTolerancePercent: 0, sparsePremiumMultiplier: 2, maxPromptPrice: 0, maxCompletionPrice: 0,
 }}});
});
test('other routes discard inactive SmartFast values and keep existing sort behavior', () => {
 for (const [route, model] of [['floor', 'foo/bar:floor'], ['nitro', 'foo/bar:nitro'], ['default', 'foo/bar']]) {
  assert.deepEqual(routingConfig('foo/bar:smartfast', route, {maxPromptPrice: ''}), {model, provider: {sort: route}});
 }
 assert.deepEqual(routingConfig('foo/bar:thinking', 'nitro'), {model: 'foo/bar:thinking', provider: {sort: 'nitro'}});
 assert.deepEqual(routingConfig('foo/bar', 'smartfast'), {model: 'foo/bar:smartfast', provider: {sort: 'smartfast'}});
});
test('invalid SmartFast values cannot become an unbounded or accidental free request', () => {
 for (const value of ['', ' ', 'NaN', 'Infinity', '-0.1', '1000.1', null, true]) {
  assert.throws(() => routingConfig('foo/bar', 'smartfast', {maxPromptPrice: value}), /input price/i);
 }
 for (const [field, value, label] of [['medianPremiumPercent', '1001', /median premium/i], ['speedTolerancePercent', '-1', /speed tolerance/i], ['sparsePremiumMultiplier', '0.99', /sparse.*multiplier/i], ['sparsePremiumMultiplier', '101', /sparse.*multiplier/i], ['maxCompletionPrice', '1001', /output price/i]]) {
  assert.throws(() => routingConfig('foo/bar', 'smartfast', {[field]: value}), label);
 }
});
test('CRC matches standard reference', () => assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926));
test('download names cannot contain paths or controls', () => {
 assert.equal(safeName('../../test\u0000.srt'), 'test.srt');
 assert.equal(safeName(''), 'subtitle.srt');
 const used = new Set();
 assert.equal(uniqueName('test.hu.srt', used), 'test.hu.srt');
 assert.equal(uniqueName('test.hu.srt', used), 'test.hu (2).srt');
});
test('floor routing replaces competing shortcut and preserves variants', () => {
 assert.equal(floorModel('foo/bar:nitro'), 'foo/bar:floor');
 assert.equal(floorModel('foo/bar:free'), 'foo/bar:free');
 assert.equal(floorModel('foo/bar'), 'foo/bar:floor');
});
test('ZIP is UTF8, stored, with CRC and matching directory', () => {
 const bytes=zip([{name:'árvíz.hu.srt',content:'Ne nézz hátra.'}]);
 const v=new DataView(bytes.buffer);
 assert.equal(v.getUint32(0,true),0x04034b50);
 assert.equal(v.getUint16(6,true),0x800);
 assert.equal(v.getUint16(8,true),0);
 assert.equal(v.getUint32(14,true),crc32(new TextEncoder().encode('Ne nézz hátra.')));
 assert.equal(v.getUint32(bytes.length-22,true),0x06054b50);
 assert.equal(v.getUint16(bytes.length-12,true),1);
 assert.throws(()=>zip([{name:'../bad',content:'x'}]), /filename/);
});
test('cue preview extracts dialogue and preserves literal markup safely', async () => {
 const { firstCue } = await import('../../src/subtitle_translator/static/archive.mjs');
 assert.equal(firstCue('1\r\n00:00:01,000 --> 00:00:02,000\r\nHello.\r\nSecond line.\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\nLater.'), 'Hello.\nSecond line.');
 assert.equal(firstCue('1\n00:00:01,000 --> 00:00:02,000\n<img src=x onerror=alert(1)>'), '<img src=x onerror=alert(1)>');
 assert.equal(firstCue('not a subtitle'), '');
});

test('cue navigation parses timestamps and complete multiline text', async () => {
 const { parseCues } = await import('../../src/subtitle_translator/static/archive.mjs');
 const cues = parseCues('1\r\n00:00:01,000 --> 00:00:02,000\r\nFirst\r\nSecond\r\n\r\n2\r\n00:00:03.000 --> 00:00:04.000\r\n<b>Later</b>');
 assert.equal(cues.length, 2);
 assert.equal(cues[0].text, 'First\nSecond');
 assert.equal(cues[0].key, '00:00:01.000-00:00:02.000#0');
 assert.equal(cues[1].text, '<b>Later</b>');
 assert.deepEqual(parseCues('not a subtitle'), []);
});

test('simultaneous cues keep separate translation matches', async () => {
 const { parseCues } = await import('../../src/subtitle_translator/static/archive.mjs');
 const source = parseCues('1\n00:00:01,000 --> 00:00:02,000\nSpeaker A\n\n2\n00:00:01,000 --> 00:00:02,000\nSpeaker B');
 const translated = parseCues('1\n00:00:01,000 --> 00:00:02,000\nTranslation A\n\n2\n00:00:01,000 --> 00:00:02,000\nTranslation B');
 const matches = new Map(translated.map(cue => [cue.key, cue.text]));
 assert.equal(matches.get(source[0].key), 'Translation A');
 assert.equal(matches.get(source[1].key), 'Translation B');
});

test('blank lines around timestamps retain exact backend-accepted captions and alignment', async () => {
 const { parseCues } = await import('../../src/subtitle_translator/static/archive.mjs');
 const content = readFileSync(new URL('./blank-lines.srt', import.meta.url), 'utf8');
 const cues = parseCues(content);
 assert.deepEqual(cues.map(cue => cue.text), ['First cue', '<i>Second cue</i>\nSecond line', 'Third cue <script>literal</script>', 'Last cue']);
 assert.deepEqual(cues.map(cue => cue.key), ['00:00:01.000-00:00:02.000#0', '00:00:03.000-00:00:04.000#0', '00:00:03.000-00:00:04.000#1', '00:00:05.000-00:00:06.000#0']);
 const translated = parseCues(content.replaceAll('\n\n', '\n').replaceAll('cue', 'caption'));
 const matches = new Map(translated.map(cue => [cue.key, cue.text]));
 assert.equal(matches.get(cues[1].key), '<i>Second caption</i>\nSecond line');
 assert.equal(matches.get(cues[2].key), 'Third caption <script>literal</script>');
});

test('queue order puts the latest event on top, not the newest creation', () => {
 const long = {key: 1, createdAt: '2026-09-08T10:00:00+00:00', startedAt: '2026-09-08T10:00:01+00:00', completedAt: '2026-09-08T10:00:30+00:00'};
 const short = {key: 2, createdAt: '2026-09-08T10:00:05+00:00', startedAt: '2026-09-08T10:00:06+00:00', completedAt: '2026-09-08T10:00:08+00:00'};
 const running = {key: 3, createdAt: '2026-09-08T10:00:10+00:00', startedAt: '2026-09-08T10:00:11+00:00'};
 // The earlier job finished last, so it leads; the running job started before that finish.
 assert.deepEqual(latestFirst([long, short, running]).map(r => r.key), [1, 3, 2]);
 // A file just added is the newest event of all.
 const added = {key: 4, addedAt: Date.parse('2026-09-08T10:00:40+00:00')};
 assert.deepEqual(latestFirst([long, short, running, added]).map(r => r.key), [4, 1, 3, 2]);
 // Equal stamps fall back to the row key, newest first.
 assert.deepEqual(latestFirst([{key: 5}, {key: 6}]).map(r => r.key), [6, 5]);
});
