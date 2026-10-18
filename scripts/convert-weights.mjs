#!/usr/bin/env node
/**
 * Convert Real-ESRGAN SRVGGNetCompact checkpoints (.pth) to fp16 safetensors.
 *
 *   npm run convert            # downloads the checkpoints into .cache/ if missing
 *
 * A .pth file is a zip holding a pickle (`data.pkl`) plus one raw blob per
 * tensor storage. There's no Python in the toolchain, so this carries a tiny
 * pickle VM that understands exactly the opcodes torch.save emits for a
 * state dict - enough to recover {name: tensor}, nothing more.
 *
 * Output goes to public/models/<name>.safetensors and is what the app loads.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { unzipSync } from "fflate";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CACHE = join(ROOT, ".cache");
const OUT = join(ROOT, "public", "models");
const RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0";

/** name -> sha256 of the upstream release asset */
const MODELS = {
  "realesr-general-x4v3": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
  "realesr-general-wdn-x4v3": "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
  "realesr-animevideov3": "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
};

// ---------------------------------------------------------------- pickle --

class Global {
  constructor(module, name) {
    this.module = module;
    this.name = name;
  }
  get qualname() {
    return `${this.module}.${this.name}`;
  }
}

/** Minimal unpickler for torch.save'd state dicts (protocol 2). */
function unpickle(buf, persistentLoad) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const td = new TextDecoder();
  let pc = 0;
  const stack = [];
  const marks = [];
  const memo = new Map();

  const popMark = () => {
    const at = marks.pop();
    return stack.splice(at);
  };
  const line = () => {
    const end = buf.indexOf(0x0a, pc);
    const s = td.decode(buf.subarray(pc, end));
    pc = end + 1;
    return s;
  };
  const call = (fn, args) => {
    switch (fn.qualname) {
      case "collections.OrderedDict":
        return new Map();
      case "torch._utils._rebuild_tensor_v2": {
        const [storage, offset, size, stride] = args;
        return { storage, offset, size, stride };
      }
      case "torch._utils._rebuild_parameter":
        return args[0];
      default:
        throw new Error(`unpickle: unsupported callable ${fn.qualname}`);
    }
  };

  for (;;) {
    const op = buf[pc++];
    switch (op) {
      case 0x80: pc++; break; // PROTO
      case 0x95: pc += 8; break; // FRAME
      case 0x2e: return stack.pop(); // STOP
      case 0x28: marks.push(stack.length); break; // MARK
      case 0x7d: stack.push(new Map()); break; // EMPTY_DICT
      case 0x5d: stack.push([]); break; // EMPTY_LIST
      case 0x29: stack.push([]); break; // EMPTY_TUPLE
      case 0x4e: stack.push(null); break; // NONE
      case 0x88: stack.push(true); break; // NEWTRUE
      case 0x89: stack.push(false); break; // NEWFALSE
      case 0x4b: stack.push(buf[pc]); pc += 1; break; // BININT1
      case 0x4d: stack.push(dv.getUint16(pc, true)); pc += 2; break; // BININT2
      case 0x4a: stack.push(dv.getInt32(pc, true)); pc += 4; break; // BININT
      case 0x8a: { // LONG1
        const n = buf[pc++];
        let v = 0n;
        for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(buf[pc + i]);
        if (n && buf[pc + n - 1] & 0x80) v -= 1n << BigInt(8 * n);
        pc += n;
        stack.push(Number(v));
        break;
      }
      case 0x47: stack.push(dv.getFloat64(pc, false)); pc += 8; break; // BINFLOAT
      case 0x58: { // BINUNICODE
        const n = dv.getUint32(pc, true);
        pc += 4;
        stack.push(td.decode(buf.subarray(pc, pc + n)));
        pc += n;
        break;
      }
      case 0x8c: { // SHORT_BINUNICODE
        const n = buf[pc++];
        stack.push(td.decode(buf.subarray(pc, pc + n)));
        pc += n;
        break;
      }
      case 0x63: { // GLOBAL
        const module = line();
        stack.push(new Global(module, line()));
        break;
      }
      case 0x93: { // STACK_GLOBAL
        const name = stack.pop();
        stack.push(new Global(stack.pop(), name));
        break;
      }
      case 0x71: memo.set(buf[pc++], stack.at(-1)); break; // BINPUT
      case 0x72: memo.set(dv.getUint32(pc, true), stack.at(-1)); pc += 4; break; // LONG_BINPUT
      case 0x94: memo.set(memo.size, stack.at(-1)); break; // MEMOIZE
      case 0x68: stack.push(memo.get(buf[pc++])); break; // BINGET
      case 0x6a: stack.push(memo.get(dv.getUint32(pc, true))); pc += 4; break; // LONG_BINGET
      case 0x74: stack.push(popMark()); break; // TUPLE
      case 0x85: stack.push([stack.pop()]); break; // TUPLE1
      case 0x86: { const b = stack.pop(); stack.push([stack.pop(), b]); break; } // TUPLE2
      case 0x87: { const c = stack.pop(), b = stack.pop(); stack.push([stack.pop(), b, c]); break; } // TUPLE3
      case 0x61: { const v = stack.pop(); stack.at(-1).push(v); break; } // APPEND
      case 0x65: { const items = popMark(); stack.at(-1).push(...items); break; } // APPENDS
      case 0x73: { const v = stack.pop(), k = stack.pop(); stack.at(-1).set(k, v); break; } // SETITEM
      case 0x75: { // SETITEMS
        const items = popMark();
        const d = stack.at(-1);
        for (let i = 0; i < items.length; i += 2) d.set(items[i], items[i + 1]);
        break;
      }
      case 0x52: { const args = stack.pop(); stack.push(call(stack.pop(), args)); break; } // REDUCE
      case 0x62: stack.pop(); break; // BUILD - only state-dict metadata, ignore
      case 0x51: stack.push(persistentLoad(stack.pop())); break; // BINPERSID
      default:
        throw new Error(`unpickle: unsupported opcode 0x${op.toString(16)} at ${pc - 1}`);
    }
  }
}

// --------------------------------------------------------------- loading --

function loadPth(path) {
  const files = unzipSync(readFileSync(path));
  const names = Object.keys(files);
  const pkl = names.find((n) => n.endsWith("/data.pkl"));
  if (!pkl) throw new Error(`${path}: no data.pkl - not a zip-format torch checkpoint`);
  const prefix = pkl.slice(0, -"data.pkl".length);

  const root = unpickle(files[pkl], (pid) => {
    const [kind, storageType, key, , numel] = pid;
    if (kind !== "storage") throw new Error(`unexpected persistent id ${kind}`);
    if (storageType.name !== "FloatStorage") throw new Error(`unsupported storage ${storageType.name}`);
    const raw = files[`${prefix}data/${key}`];
    return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + numel * 4));
  });

  const dict = root instanceof Map && root.has("params") ? root.get("params") : root;
  const tensors = new Map();
  for (const [name, t] of dict) {
    const numel = t.size.reduce((a, b) => a * b, 1);
    // contiguity check: torch.save keeps the stride, the reader assumes row-major
    let expect = 1;
    for (let i = t.size.length - 1; i >= 0; i--) {
      if (t.size[i] !== 1 && t.stride[i] !== expect) throw new Error(`${name} is not contiguous`);
      expect *= t.size[i];
    }
    tensors.set(name, { shape: t.size, data: t.storage.subarray(t.offset, t.offset + numel) });
  }
  return tensors;
}

// ------------------------------------------------------------ safetensors --

function f32ToF16(v) {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  f[0] = v;
  const x = u[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (exp >= 31) return sign | 0x7c00; // overflow -> inf (never happens for these weights)
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let h = mant >> shift;
    if ((mant >> (shift - 1)) & 1) h++; // round half up
    return sign | h;
  }
  let h = sign | (exp << 10) | (mant >> 13);
  if (mant & 0x1000) h++; // round to nearest
  return h;
}

function writeSafetensors(path, tensors, metadata) {
  const header = { __metadata__: metadata };
  let offset = 0;
  const chunks = [];
  for (const [name, t] of tensors) {
    const h = new Uint16Array(t.data.length);
    for (let i = 0; i < h.length; i++) h[i] = f32ToF16(t.data[i]);
    header[name] = { dtype: "F16", shape: t.shape, data_offsets: [offset, offset + h.byteLength] };
    offset += h.byteLength;
    chunks.push(new Uint8Array(h.buffer));
  }
  let json = JSON.stringify(header);
  json += " ".repeat((8 - (json.length % 8)) % 8); // keep the data 8-byte aligned
  const head = new Uint8Array(8);
  new DataView(head.buffer).setBigUint64(0, BigInt(json.length), true);
  writeFileSync(path, Buffer.concat([head, Buffer.from(json), ...chunks]));
}

// ------------------------------------------------------------------ main --

async function fetchCheckpoint(name, sha) {
  const path = join(CACHE, `${name}.pth`);
  if (!existsSync(path)) {
    mkdirSync(CACHE, { recursive: true });
    console.log(`downloading ${name}.pth`);
    const res = await fetch(`${RELEASE}/${name}.pth`);
    if (!res.ok) throw new Error(`download ${name}: HTTP ${res.status}`);
    writeFileSync(path, Buffer.from(await res.arrayBuffer()));
  }
  const got = createHash("sha256").update(readFileSync(path)).digest("hex");
  if (got !== sha) throw new Error(`${name}.pth: sha256 mismatch (${got})`);
  return path;
}

mkdirSync(OUT, { recursive: true });
for (const [name, sha] of Object.entries(MODELS)) {
  const tensors = loadPth(await fetchCheckpoint(name, sha));
  const convs = [...tensors.keys()].filter((k) => /^body\.\d+\.weight$/.test(k) && tensors.get(k).shape.length === 4);
  let maxErr = 0;
  for (const t of tensors.values()) for (const v of t.data) maxErr = Math.max(maxErr, Math.abs(v));
  writeSafetensors(join(OUT, `${name}.safetensors`), tensors, {
    source: `${RELEASE}/${name}.pth`,
    license: "BSD-3-Clause (Real-ESRGAN, Xintao Wang)",
    arch: "SRVGGNetCompact",
    num_conv: String(convs.length - 2),
    num_feat: String(tensors.get("body.0.weight").shape[0]),
    upscale: "4",
  });
  console.log(`${name}: ${tensors.size} tensors, ${convs.length} convs, max |w| ${maxErr.toFixed(3)}`);
}
