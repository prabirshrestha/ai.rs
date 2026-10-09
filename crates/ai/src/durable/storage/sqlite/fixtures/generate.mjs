// Regenerates the TS-written SQLite fixture and the reads TS performs on it.
//
//   node crates/ai/src/durable/storage/sqlite/fixtures/generate.mjs <path-to>/packages crates/ai/src/durable/storage/sqlite/fixtures
//
// Runs only the snapshot's own sources (Node >= 22.13: built-in node:sqlite and
// type stripping); `@earendil-works/chord` imports resolve to the snapshot's
// chord sources through a resolve hook, so nothing is installed.
import { register } from "node:module";
import { rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [packages, output] = process.argv.slice(2);
if (!packages || !output) throw new Error("usage: generate.mjs <pi-src/packages> <fixtures-dir>");
const chord = pathToFileURL(join(packages, "chord/src")).href;
const hook = `
const map = { "@earendil-works/chord": "${chord}/index.ts", "@earendil-works/chord/delta": "${chord}/delta/index.ts", "@earendil-works/chord/context": "${chord}/context/index.ts" };
export async function resolve(specifier, context, next) {
	if (specifier in map) return { url: map[specifier], shortCircuit: true };
	return next(specifier, context);
}`;
register(`data:text/javascript,${encodeURIComponent(hook)}`);

const durable = join(packages, "durable/src");
const { DatabaseSync } = await import("node:sqlite");
const { NodeSqliteDatabase } = await import(pathToFileURL(join(durable, "storage/sqlite/node.ts")).href);
const { SqliteStorage } = await import(pathToFileURL(join(durable, "storage/sqlite/storage.ts")).href);

const path = join(output, "ts-written.sqlite");
for (const suffix of ["", "-wal", "-shm"]) rmSync(`${path}${suffix}`, { force: true });
const connection = new DatabaseSync(path);
// Small pages keep the checked-in file small; the format is otherwise the adapter's own.
connection.exec("PRAGMA page_size = 1024");
connection.exec("PRAGMA journal_mode = WAL");
connection.exec("PRAGMA synchronous = NORMAL");
const storage = await SqliteStorage.open(new NodeSqliteDatabase(connection));
const context = { abortSignal: undefined };

const ROOT = 1;
const seqs = {};
const commit = async (name, writes) => {
	seqs[name] = await storage.commit(writes, context);
};
const pendingTask = (id, phase) => ({
	id,
	conversationId: ROOT,
	kind: "fixture.task",
	version: 2,
	input: { prompt: "héllo 😀" },
	background: false,
	abortRequested: false,
	state: { status: "pending", checkpoint: { phase } },
	memos: { first: 1 },
});
const conversationDocument = { kind: "fixture.history", scope: { kind: "conversation", conversationId: ROOT }, history: "rewindable", fork: "asOf" };

await commit("root", [
	{ type: "conversation", value: { id: ROOT } },
	{ type: "entry", value: { id: 2, conversationId: ROOT, kind: "pi.user", model: [{ role: "user", content: "hi", timestamp: 1 }], data: { text: "hi" } } },
	{ type: "entry", value: { id: 3, conversationId: ROOT, kind: "marker", head: 2, data: null } },
]);
await commit("task", [
	{ type: "task", value: pendingTask(5, "start") },
	{ type: "conversation", value: { id: 4, parent: { conversationId: ROOT, at: 2 } } },
	{ type: "conversation", value: { id: 6, owner: { conversationId: ROOT, taskId: 5 } } },
	{ type: "entry", value: { id: 7, conversationId: 4, kind: "fork.entry", data: { "é": ["😀", 1.5, true] }, byTaskId: 5 } },
	{ type: "submission", value: { id: 8, conversationId: ROOT, requestId: "req-é\u0000", type: "input", status: "queued" } },
	{ type: "submission", value: { id: 9, conversationId: ROOT, type: "write", status: "done", entry: 3 } },
]);
await commit("documents", [
	{ type: "document.create", record: { id: 10, ...conversationDocument }, content: { kind: "base", version: 1, value: { count: 0, text: "ab", rows: [] } } },
	{ type: "document.create", record: { id: 11, kind: "fixture.session", scope: { kind: "session" } }, content: { kind: "base", version: 3, value: { n: 0 } } },
	{ type: "document.create", record: { id: 12, kind: "fixture.family", key: "member/é", scope: { kind: "task", taskId: 5 } }, content: { kind: "base", version: 1, value: { seed: "x" } } },
	{ type: "document.create", record: { id: 13, kind: "fixture.retired", scope: { kind: "session" } }, content: { kind: "base", version: 1, value: {} } },
]);
await commit("deltas", [
	{ type: "document.change", id: 10, content: { kind: "delta", version: 1, ops: [["s", ["count"], 1], ["a", ["text"], "😀cd"], ["p", ["rows"], 0, 0, [{ id: 1 }, { id: 2 }]]] } },
	{ type: "document.change", id: 11, content: { kind: "delta", version: 3, ops: [["s", ["n"], 1]] } },
	{ type: "document.retire", id: 13 },
]);
await commit("checkpoint", [
	{ type: "document.change", id: 10, content: { kind: "base", version: 1, value: { count: 2, text: "ab😀cd", rows: [{ id: 2 }, { id: 1 }] } } },
	{ type: "task", value: { ...pendingTask(5, "next"), state: { status: "waiting", checkpoint: { phase: "next" }, on: [], policy: "failFast" } } },
]);
await commit("tail", [
	{ type: "document.change", id: 10, content: { kind: "delta", version: 1, ops: [["t", ["text"], 2], ["m", ["rows"], [1, 0]], ["d", ["count"]]] } },
	{ type: "document.change", id: 11, content: { kind: "base", version: 4, value: { n: 2, migrated: true } } },
]);
await commit("fork", [
	{ type: "document.copy", record: { id: 14, ...conversationDocument, scope: { kind: "conversation", conversationId: 4 } }, source: { id: 10, at: seqs.deltas } },
	{ type: "task", value: { ...pendingTask(5, "done"), state: { status: "terminal", outcome: { status: "completed", result: { ok: "✓" } } }, memos: undefined } },
	{ type: "submission", value: { id: 8, conversationId: ROOT, requestId: "req-é\u0000", type: "input", status: "done", entry: 2, answer: 3 } },
	{ type: "entry", value: { id: 15, conversationId: 4, kind: "fork.marker", head: 7 } },
]);
const minted = await storage.mintId();

const page = (value) => ({ items: value.items, ...(value.next === undefined ? {} : { next: value.next }) });
const reads = {
	seqs,
	minted,
	conversation4: await storage.conversation(4, context),
	conversation6: await storage.conversation(6, context),
	scanConversations: page(await storage.scanConversations({}, 2, undefined, context)),
	scanConversationsByTask: page(await storage.scanConversations({ ownerTaskId: 5 }, 10, undefined, context)),
	entry2: await storage.entry(2, context),
	entry2InFork: await storage.entry(4, 2, context),
	entry3InFork: (await storage.entry(4, 3, context)) ?? null,
	scanForkEntries: page(await storage.scanEntries({ conversationId: 4 }, 10, undefined, context)),
	rootHeadMarker: await storage.findLatestHeadMarker(ROOT, undefined, context),
	forkHeadMarkerBefore: (await storage.findLatestHeadMarker(4, 7, context)) ?? null,
	task5: await storage.task(5, context),
	terminalTasks: page(await storage.scanTasks({ status: "terminal", kind: "fixture.task" }, 10, undefined, context)),
	submission8: await storage.submission(8, context),
	submissionByRequest: await storage.submissionByRequest(ROOT, "req-é\u0000", context),
	doneSubmissions: page(await storage.scanSubmissions({ status: "done" }, 10, undefined, context)),
	findHistoryCurrent: await storage.findDocument({ kind: "fixture.history", scope: conversationDocument.scope }, "current", context),
	findFamilyMember: await storage.findDocument({ kind: "fixture.family", key: "member/é", scope: { kind: "task", taskId: 5 } }, "current", context),
	findRetiredBefore: await storage.findDocument({ kind: "fixture.retired", scope: { kind: "session" } }, seqs.documents, context),
	findRetiredAfter: (await storage.findDocument({ kind: "fixture.retired", scope: { kind: "session" } }, seqs.deltas, context)) ?? null,
	historyAtDocuments: await storage.document(10, seqs.documents, context),
	historyAtDeltas: await storage.document(10, seqs.deltas, context),
	historyCurrent: await storage.document(10, "current", context),
	sessionCurrent: await storage.document(11, "current", context),
	copiedCurrent: await storage.document(14, "current", context),
	sessionDocuments: page(await storage.scanDocuments({ scope: { kind: "session" }, at: seqs.documents }, 10, undefined, context)),
	rootDocuments: page(await storage.scanDocuments({ scope: conversationDocument.scope, at: "current", kind: "fixture.history" }, 10, undefined, context)),
	schema: connection.prepare("SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY name").all(),
};
await storage.close(context);
writeFileSync(join(output, "ts-reads.json"), `${JSON.stringify(reads, null, "\t")}\n`);
