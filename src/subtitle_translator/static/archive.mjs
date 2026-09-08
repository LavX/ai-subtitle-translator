const encoder = new TextEncoder();
export function crc32(bytes) {
 let crc = 0xffffffff;
 for (const byte of bytes) {
  crc ^= byte;
  for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
 }
 return (crc ^ 0xffffffff) >>> 0;
}
export function safeName(name) {
 return String(name).split(/[\\/]/).pop().replace(/[\u0000-\u001f\u007f<>:"|?*]/g, '').replace(/^\.+|[. ]+$/g, '').slice(0, 180) || 'subtitle.srt';
}
export function uniqueName(name, used) {
 const clean = safeName(name);
 const dot = clean.lastIndexOf('.');
 const stem = dot > 0 ? clean.slice(0, dot) : clean;
 const ext = dot > 0 ? clean.slice(dot) : '';
 let result = clean;
 let i = 2;
 while (used.has(result.toLowerCase())) result = `${stem} (${i++})${ext}`;
 used.add(result.toLowerCase());
 return result;
}
export function routeModel(model, routing) {
 const value = model.trim().replace(/:(nitro|floor)$/, '');
 return routing === 'default' || value.includes(':') ? value : `${value}:${routing}`;
}
export function floorModel(model) { return routeModel(model, 'floor'); }
export function zip(files) {
 if (files.length > 65535) throw new Error('Too many files');
 const parts = [], directory = [];
 let offset = 0, directorySize = 0;
 for (const file of files) {
  if (safeName(file.name) !== file.name) throw new Error('Unsafe filename');
  const name = encoder.encode(file.name), data = encoder.encode(file.content), crc = crc32(data);
  const local = new Uint8Array(30 + name.length), lv = new DataView(local.buffer);
  lv.setUint32(0, 0x04034b50, true); lv.setUint16(4, 20, true); lv.setUint16(6, 0x800, true);
  lv.setUint16(12, 33, true); lv.setUint32(14, crc, true); lv.setUint32(18, data.length, true); lv.setUint32(22, data.length, true); lv.setUint16(26, name.length, true); local.set(name, 30);
  const central = new Uint8Array(46 + name.length), cv = new DataView(central.buffer);
  cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, 20, true); cv.setUint16(6, 20, true); cv.setUint16(8, 0x800, true); cv.setUint16(14, 33, true); cv.setUint32(16, crc, true); cv.setUint32(20, data.length, true); cv.setUint32(24, data.length, true); cv.setUint16(28, name.length, true); cv.setUint32(42, offset, true); central.set(name, 46);
  parts.push(local, data); directory.push(central); offset += local.length + data.length; directorySize += central.length;
 }
 if (offset + directorySize > 0xffffffff) throw new Error('Archive is too large');
 const end = new Uint8Array(22), ev = new DataView(end.buffer);
 ev.setUint32(0, 0x06054b50, true); ev.setUint16(8, files.length, true); ev.setUint16(10, files.length, true); ev.setUint32(12, directorySize, true); ev.setUint32(16, offset, true);
 const result = new Uint8Array(offset + directorySize + end.length);
 let cursor = 0;
 for (const part of [...parts, ...directory, end]) { result.set(part, cursor); cursor += part.length; }
 return result;
}
export function firstCue(content) {
 const lines = String(content).replace(/\r\n?/g, '\n').split('\n');
 const start = lines.findIndex(line => /^\s*\d{2,}:\d{2}:\d{2}[,.]\d{3}\s*-->/.test(line));
 if (start < 0) return '';
 const cue = [];
 for (const line of lines.slice(start + 1)) {
  if (!line.trim()) break;
  cue.push(line);
  if (cue.join('\n').length > 400) break;
 }
 return cue.join('\n').slice(0, 400);
}

export function parseCues(content) {
 const source = String(content || '').replace(/\r\n?/g, '\n');
 const headers = [...source.matchAll(/^[ \t]*(?:\d+[ \t]*\n(?:[ \t]*\n)*)?[ \t]*(\d{2,}:\d{2}:\d{2}[,.]\d{3})[ \t]*-->[ \t]*(\d{2,}:\d{2}:\d{2}[,.]\d{3})[^\n]*/gm)];
 const cues = [], occurrences = new Map();
 for (let index = 0; index < headers.length; index++) {
  const match = headers[index];
  const text = source.slice(match.index + match[0].length, headers[index + 1]?.index ?? source.length).trim();
  if (text) {
   const timestamp = `${match[1]}-${match[2]}`.replace(/,/g, '.');
   const occurrence = occurrences.get(timestamp) || 0;
   occurrences.set(timestamp, occurrence + 1);
   cues.push({time: `${match[1]} → ${match[2]}`, key: `${timestamp}#${occurrence}`, text});
  }
 }
 return cues;
}

// Queue order: whatever happened most recently sits on top. A file just added, a job
// just submitted or started, a job that just finished or was cancelled. Creation order
// is only the tie-break, so a long job created earlier still rises when it finishes.
export function latestFirst(rows) {
 const stamp = value => Date.parse((value || '').replace(/([+-]\d\d:\d\d)Z$/, '$1')) || 0;
 const when = row => Math.max(row.addedAt || 0, stamp(row.createdAt), stamp(row.startedAt), stamp(row.completedAt));
 return [...rows].sort((a, b) => when(b) - when(a) || (b.key || 0) - (a.key || 0));
}
