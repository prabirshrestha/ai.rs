// Dumps an adapter's built-in tool catalog (`harness.builtinTools`) as JSON,
// with each schema reduced the way `asSchema` reduces it. Run from a built
// vercel/ai checkout by scripts/refresh-harness-bridges.sh:
//
//   node dump-harness-builtin-tools.mjs <vercel-ai> <package> <factory>
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';

const [vercelAi, packageName, factory] = process.argv.slice(2);
const packageDir = join(vercelAi, 'packages', packageName);
const adapter = await import(pathToFileURL(join(packageDir, 'dist/index.js')).href);
const { asSchema } = await import(
  pathToFileURL(join(vercelAi, 'packages/provider-utils/dist/index.js')).href
);

const harness = adapter[factory]();
const catalog = {};
for (const [key, tool] of Object.entries(harness.builtinTools)) {
  const entry = {};
  for (const field of ['title', 'description', 'nativeName', 'commonName', 'toolUseKind']) {
    if (tool[field] !== undefined) entry[field] = tool[field];
  }
  entry.inputSchema = await asSchema(tool.inputSchema).jsonSchema;
  if (tool.outputSchema !== undefined) {
    entry.outputSchema = await asSchema(tool.outputSchema).jsonSchema;
  }
  catalog[key] = entry;
}
process.stdout.write(`${JSON.stringify(catalog, null, 2)}\n`);
