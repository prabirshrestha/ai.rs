// Regenerates the chord golden fixtures from the Pi TypeScript sources.
//
//   node crates/ai/src/chord/fixtures/generate.mjs <path-to>/packages/chord/src > crates/ai/src/chord/fixtures/delta.json
//
// Runs only the snapshot's own code (no dependencies). Inputs are JSON-parsed,
// so no containers are shared between `before` and `after`.
const src = process.argv[2];
if (!src) throw new Error("usage: generate.mjs <chord/src>");
const delta = await import(`${src}/delta/index.ts`);
const { diffRevisions, applyImmutable, encoder, track } = delta;

const clone = (value) => JSON.parse(JSON.stringify(value));

// ── 1. Op serialization ─────────────────────────────────────────────────────
const ops = [
	["r", { value: 1, nested: { list: [1, "two", null, true] } }],
	["r", null],
	["r", [1, 2, 3]],
	["s", ["count"], 2],
	["s", ["values", 0, "label"], "changed"],
	["s", ["a\u0000b", "é", "😀"], { "x": [{}] }],
	["d", ["remove"]],
	["d", ["values", 3]],
	["a", ["text"], " world"],
	["a", ["nested", "text"], "émoji 😀 \"quoted\"\n"],
	["t", ["text"], 6],
	["t", ["rows", 2, "output"], 0],
	["p", ["values"], 1, 0, [{ id: "c" }]],
	["p", [], 0, 4, []],
	["p", ["values"], 2, 1, [null, -1.5, 0.1, 12345678]],
	["m", ["values"], [2, 0, 1]],
	["m", [], [0]],
];

// ── 2. Wire encoding ────────────────────────────────────────────────────────
const wireBatches = [
	[["t", ["nested", "text"], 1], ["a", ["nested", "text"], "x"]],
	[["a", ["nested", "text"], "y"]],
	[["s", ["value"], 1], ["s", ["value"], 2], ["d", ["value"]]],
	[["p", ["rows"], 0, 0, [1]], ["m", ["rows"], [1, 0]], ["s", ["value"], 3]],
	[["r", { value: 3 }]],
	[["s", ["value"], 4], ["a", ["nested", "text"], "z"]],
];
const enc = encoder();
const wire = wireBatches.map((batch) => ({ ops: batch, wire: enc.encode(batch) }));

// ── 3. diffRevisions over a seeded corpus ───────────────────────────────────
let seed = 0x2545f491;
const random = () => {
	seed ^= seed << 13;
	seed ^= seed >>> 17;
	seed ^= seed << 5;
	return ((seed >>> 0) % 1_000_000) / 1_000_000;
};
const pick = (items) => items[Math.floor(random() * items.length)];
const words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "😀", "é"];
const leaf = () => pick([() => Math.floor(random() * 100), () => pick(words), () => random() < 0.5, () => null])();
const value = (depth) => {
	if (depth > 2 || random() < 0.35) return leaf();
	if (random() < 0.5) return Array.from({ length: Math.floor(random() * 5) }, (_, index) => ({ id: index * 10 + Math.floor(random() * 10), body: value(depth + 1) }));
	const object = {};
	for (let index = 0; index < 1 + Math.floor(random() * 4); index++) object[pick(words) + index] = value(depth + 1);
	return object;
};
const containers = (node, path, out) => {
	if (node !== null && typeof node === "object") {
		out.push([path, node]);
		for (const key of Object.keys(node)) containers(node[key], [...path, Array.isArray(node) ? Number(key) : key], out);
	}
	return out;
};
const mutate = (root) => {
	const all = containers(root, [], []);
	const [, target] = pick(all);
	if (Array.isArray(target)) {
		switch (Math.floor(random() * 6)) {
			case 0: target.push({ id: 900 + Math.floor(random() * 99), body: leaf() }); break;
			case 1: target.shift(); break;
			case 2: target.unshift({ id: 800 + Math.floor(random() * 99), body: leaf() }); break;
			case 3: target.reverse(); break;
			case 4: if (target.length > 0) target.splice(Math.floor(random() * target.length), 1); break;
			default: if (target.length > 0) target[Math.floor(random() * target.length)] = { id: 700, body: leaf() };
		}
		return;
	}
	const keys = Object.keys(target);
	const key = keys.length > 0 && random() < 0.8 ? pick(keys) : `new${Math.floor(random() * 10)}`;
	const current = target[key];
	switch (Math.floor(random() * 5)) {
		case 0: delete target[key]; break;
		case 1: target[key] = typeof current === "string" ? `${current}${pick(words)}` : pick(words); break;
		case 2: target[key] = typeof current === "string" ? `${current.slice(2)}${pick(words)}` : leaf(); break;
		case 3: target[key] = value(2); break;
		default: target[key] = leaf();
	}
};
const diffs = [];
for (let index = 0; index < 200; index++) {
	const before = { doc: value(0), text: pick(words).repeat(3), rows: Array.from({ length: 4 }, (_, id) => ({ id, label: pick(words) })) };
	const after = clone(before);
	for (let step = 0, steps = 1 + Math.floor(random() * 3); step < steps; step++) mutate(after);
	const operations = diffRevisions(clone(before), clone(after));
	if (JSON.stringify(applyImmutable(clone(before), operations)) === undefined) throw new Error("unreachable");
	diffs.push({ before, after, ops: operations });
}

// ── 4. Tracker emission for simple scripted mutations ───────────────────────
const at = (root, path) => path.reduce((node, key) => node[key], root);
const scripts = [
	{ initial: { count: 1, nested: { text: "a" }, values: [1] }, steps: [["set", ["count"], 2], ["append", ["nested", "text"], "b"], ["push", ["values"], 2]] },
	{ initial: { text: "abcdefgh" }, steps: [["roll", ["text"], 3, "xyz"]] },
	{ initial: { first: 1, second: 2 }, steps: [["set", ["first"], 3], ["delete", ["second"]], ["set", ["third"], 4]] },
	{ initial: { rows: [{ id: 1 }, { id: 2 }, { id: 3 }] }, steps: [["shift", ["rows"]]] },
	{ initial: { rows: [{ id: 1 }, { id: 2 }] }, steps: [["unshift", ["rows"], { id: 0 }]] },
	{ initial: { rows: [{ id: 1, label: "one" }, { id: 2, label: "two" }] }, steps: [["set", ["rows", 1, "label"], "changed"]] },
	{ initial: { live: { text: "partial", parts: [] } }, steps: [["append", ["live", "text"], " response"], ["push", ["live", "parts"], { type: "text" }]] },
	{ initial: { values: [1, 2] }, steps: [["length", ["values"], 4]] },
	{ initial: { values: [1, 2, 3, 4] }, steps: [["pop", ["values"]]] },
	{ initial: { value: { nested: [1, 2] } }, steps: [["set", ["value"], { nested: [1, 2] }]] },
	{ initial: { a: { b: { c: "x" } } }, steps: [["set", ["a", "b", "d"], [1]], ["delete", ["a", "b", "c"]]] },
];
const tracker = scripts.map(({ initial, steps }) => {
	const tracked = track(clone(initial));
	const change = tracked.beginChange();
	for (const [kind, path, ...args] of steps) {
		const parent = at(change.state, path.slice(0, -1));
		const key = path[path.length - 1];
		switch (kind) {
			case "set": parent[key] = clone(args[0]); break;
			case "append": parent[key] += args[0]; break;
			case "roll": parent[key] = `${parent[key].slice(args[0])}${args[1]}`; break;
			case "delete": delete parent[key]; break;
			case "push": parent[key].push(clone(args[0])); break;
			case "unshift": parent[key].unshift(clone(args[0])); break;
			case "shift": parent[key].shift(); break;
			case "pop": parent[key].pop(); break;
			case "length": parent[key].length = args[0]; break;
		}
	}
	const prepared = change.prepare();
	return { initial, steps, ops: prepared.ops, value: prepared.value };
});

process.stdout.write(`${JSON.stringify({ ops: ops.map((op) => ({ op, json: JSON.stringify(op) })), wire, diffs, tracker })}\n`);
