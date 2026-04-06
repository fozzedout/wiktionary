#!/usr/bin/env node
// Build a compact radix-trie from a word list and emit a binary (WTRI)
// and/or JS module for direct loading in browser/node.
//
// Output formats:
//   - .bin  : compact binary (WTRI v1)
//   - .js   : ESM module exporting TRIE_DATA + TRIE_METADATA
//   - .json : build metadata
//
// Usage:
//   node scripts/build-trie.mjs [--input en-words.txt] [--out en-words-trie.js] [--format js|bin|both]

import fs from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

const VERSION = '1.0.0';

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { input: 'en-words.txt', out: 'en-words-trie.js', format: 'js' };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--input' && args[i+1]) { out.input = args[++i]; }
    else if (a === '--out' && args[i+1]) { out.out = args[++i]; }
    else if (a === '--format' && args[i+1]) { out.format = args[++i]; }
    else if (a === '-h' || a === '--help') {
      console.log('Usage: node scripts/build-trie.mjs [--input en-words.txt] [--out en-words-trie.js] [--format js|bin|both]');
      process.exit(0);
    }
  }
  return out;
}

function makeNode() { return { c: new Map(), e: false }; }

function buildTrie(words) {
  const root = makeNode();
  for (const w of words) {
    if (!w) continue;
    let node = root;
    for (let i = 0; i < w.length; i++) {
      const ch = w[i];
      let nxt = node.c.get(ch);
      if (!nxt) { nxt = makeNode(); node.c.set(ch, nxt); }
      node = nxt;
    }
    node.e = true;
  }
  return root;
}

function compressRadix(node) {
  const visit = (n) => {
    const cn = { e: n.e, edges: [] };
    for (const [ch, child0] of n.c.entries()) {
      let label = ch;
      let child = child0;
      while (!child.e && child.c.size === 1) {
        const [ch2, child2] = child.c.entries().next().value;
        label += ch2;
        child = child2;
      }
      const childComp = visit(child);
      cn.edges.push({ label, to: childComp });
    }
    cn.edges.sort((a,b) => a.label.codePointAt(0) - b.label.codePointAt(0));
    return cn;
  };
  return visit(node);
}

function assignIdsDFS(root) {
  const nodes = [];
  function dfs(n) {
    n.__id = nodes.length;
    nodes.push(n);
    for (const e of n.edges) dfs(e.to);
  }
  dfs(root);
  return nodes;
}

function encodeTrie(root) {
  const nodes = assignIdsDFS(root);
  const nodeCount = nodes.length;

  const nodeFirstEdge = new Array(nodeCount);
  const nodeEdgeCount = new Array(nodeCount);
  const labels = [];
  const edgeLabelStart = [];
  const edgeLabelLen = [];
  const edgeTo = [];

  let edgeCursor = 0;
  let labelCursor = 0;
  for (const n of nodes) {
    nodeFirstEdge[n.__id] = edgeCursor;
    nodeEdgeCount[n.__id] = n.edges.length;
    for (const e of n.edges) {
      edgeLabelStart.push(labelCursor);
      edgeLabelLen.push(e.label.length);
      edgeTo.push(e.to.__id);
      labels.push(e.label);
      labelCursor += e.label.length;
      edgeCursor++;
    }
  }

  const termBits = new Uint8Array(Math.ceil(nodeCount / 8));
  for (const n of nodes) {
    if (n.e) {
      const i = n.__id;
      termBits[i >> 3] |= (1 << (i & 7));
    }
  }

  const LBL = labels.join('');
  return {
    data: {
      LBL,
      nodeFirstEdge,
      nodeEdgeCount,
      edgeLabelStart,
      edgeLabelLen,
      edgeTo,
      termBits: Array.from(termBits),
    },
    stats: {
      nodeCount,
      edgeCount: edgeTo.length,
      labelBytes: LBL.length,
    }
  };
}

function utf8Bytes(str) {
  return new TextEncoder().encode(str);
}

function changeExt(p, ext) {
  const dir = path.dirname(p);
  const base = path.basename(p, path.extname(p));
  return path.join(dir, base + ext);
}

async function writeBinary(outPath, enc) {
  const { nodeFirstEdge, nodeEdgeCount, edgeLabelStart, edgeLabelLen, edgeTo, termBits, LBL } = enc.data;
  const nodeCount = enc.stats.nodeCount;
  const edgeCount = enc.stats.edgeCount;
  const lblBytes = utf8Bytes(LBL);
  const termBytes = Uint8Array.from(termBits);

  let maxLabelLen = 0;
  for (let i = 0; i < edgeLabelLen.length; i++) { const L = edgeLabelLen[i]; if (L > maxLabelLen) maxLabelLen = L; }
  if (maxLabelLen > 0xFFFF) throw new Error('edgeLabelLen exceeds uint16 range');

  // Layout:
  // magic[4] = 'WTRI', ver u8 = 1, pad u8=0, pad u16=0
  // nodeCount u32, edgeCount u32, labelBytes u32, termByteCount u32
  // nodeFirstEdge u32[nodeCount]
  // nodeEdgeCount u32[nodeCount]
  // edgeLabelStart u32[edgeCount]
  // edgeLabelLen u16[edgeCount]
  // edgeTo u32[edgeCount]
  // termBits u8[termByteCount]
  // LBL u8[labelBytes]
  const headerBytes = 4 + 1 + 1 + 2 + 4*4;
  const bodyBytes = 4*nodeCount + 4*nodeCount + 4*edgeCount + 2*edgeCount + 4*edgeCount + termBytes.length + lblBytes.length;
  const total = headerBytes + bodyBytes;
  const buf = new ArrayBuffer(total);
  const view = new DataView(buf);
  let off = 0;
  view.setUint8(off++, 'W'.charCodeAt(0));
  view.setUint8(off++, 'T'.charCodeAt(0));
  view.setUint8(off++, 'R'.charCodeAt(0));
  view.setUint8(off++, 'I'.charCodeAt(0));
  view.setUint8(off++, 1); // version
  view.setUint8(off++, 0);
  view.setUint16(off, 0, true); off += 2;
  view.setUint32(off, nodeCount, true); off += 4;
  view.setUint32(off, edgeCount, true); off += 4;
  view.setUint32(off, lblBytes.length, true); off += 4;
  view.setUint32(off, termBytes.length, true); off += 4;

  function copyArrayU32(arr) {
    const u32 = new Uint32Array(buf, off, arr.length);
    u32.set(arr);
    off += arr.length * 4;
  }
  function copyArrayU16(arr) {
    const u16 = new Uint16Array(buf, off, arr.length);
    u16.set(arr);
    off += arr.length * 2;
  }
  function copyArrayU8(arr) {
    const u8 = new Uint8Array(buf, off, arr.length);
    u8.set(arr);
    off += arr.length;
  }

  copyArrayU32(Uint32Array.from(nodeFirstEdge));
  copyArrayU32(Uint32Array.from(nodeEdgeCount));
  copyArrayU32(Uint32Array.from(edgeLabelStart));
  copyArrayU16(Uint16Array.from(edgeLabelLen));
  copyArrayU32(Uint32Array.from(edgeTo));
  copyArrayU8(termBytes);
  copyArrayU8(lblBytes);

  await fs.writeFile(outPath, new Uint8Array(buf));
}

async function main() {
  const { input, out, format } = parseArgs();
  const inputPath = path.resolve(process.cwd(), input);
  const outPath = path.resolve(process.cwd(), out);

  const t0 = performance.now();
  const raw = await fs.readFile(inputPath, 'utf8');
  const tRead = performance.now();
  const words = raw.replace(/\r/g, '').split('\n').map(w => w.trim().toLowerCase()).filter(Boolean);
  const tParse = performance.now();
  const trie = buildTrie(words);
  const tBuild = performance.now();
  const radix = compressRadix(trie);
  const tCompress = performance.now();
  const { data, stats } = encodeTrie(radix);
  const tEncode = performance.now();

  const banner = '// Auto-generated by scripts/build-trie.mjs — DO NOT EDIT\n';
  const meta = {
    builtAt: new Date().toISOString(),
    generatorVersion: VERSION,
    wordCount: words.length,
    nodeCount: stats.nodeCount,
    edgeCount: stats.edgeCount,
    labelBytes: stats.labelBytes,
    timingsMs: {
      read: +(tRead - t0).toFixed(2),
      parse: +(tParse - tRead).toFixed(2),
      buildTrie: +(tBuild - tParse).toFixed(2),
      compressRadix: +(tCompress - tBuild).toFixed(2),
      encode: +(tEncode - tCompress).toFixed(2),
      total: +(tEncode - t0).toFixed(2),
    }
  };

  const wantJS = format === 'js' || format === 'both' || !format;
  const wantBIN = format === 'bin' || format === 'both';

  if (wantJS) {
    const moduleSrc = `${banner}export const TRIE_DATA = ${JSON.stringify(data)};\nexport const TRIE_METADATA = ${JSON.stringify(meta)};\n`;
    await fs.writeFile(outPath, moduleSrc);
    console.log(`Wrote ${path.basename(outPath)} (nodes=${stats.nodeCount}, edges=${stats.edgeCount}, labels=${stats.labelBytes} chars)`);
  }

  if (wantBIN) {
    const binPath = changeExt(outPath, '.bin');
    await writeBinary(binPath, { data, stats });
    await fs.writeFile(changeExt(outPath, '.json'), JSON.stringify(meta, null, 2));
    console.log(`Wrote ${path.basename(binPath)} and ${path.basename(changeExt(outPath, '.json'))}`);
  }

  // Write manifest with extract metadata for stage-1 pipeline integration
  const manifest = {
    ...meta,
    trieBinaryFilename: path.basename(changeExt(outPath, '.bin')),
    inputFile: inputPath,
  };
  const manifestPath = changeExt(outPath, '.manifest.json');
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`Wrote ${path.basename(manifestPath)}`);

  console.log(`Timings (ms):`, meta.timingsMs);
}

main().catch(err => { console.error(err); process.exit(1); });
