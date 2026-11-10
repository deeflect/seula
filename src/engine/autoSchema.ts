import { FieldDef, FieldKind, SchemaDef } from "./schema";

function splitIdentifierWords(ident: string): string[] {
  const spaced = ident
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_\-\s]+/g, " ")
    .trim()
    .toLowerCase();
  return spaced.split(" ").filter((w) => w.length > 1);
}

function buildFieldAliases(fieldName: string): string[] {
  const low = fieldName.toLowerCase();
  const aliases = new Set<string>();
  const words = splitIdentifierWords(fieldName);

  for (const w of words) {
    if (w !== low && w.length >= 2) aliases.add(w);
    if (w.endsWith("s") && w.length > 3) aliases.add(w.slice(0, -1));
  }

  aliases.delete(low);
  return Array.from(aliases).slice(0, 8);
}

/**
 * Zero-configuration Schema Auto-Inference (`inferSchemaFromRows`).
 * Inspects raw JSON rows and constructs
 * a `SchemaDef` with typed fields plus natural-language aliases.
 */
export function inferSchemaFromRows(
  schemaName: string,
  rows: ReadonlyArray<Record<string, unknown>>,
): SchemaDef {
  if (!rows || rows.length === 0) {
    return { name: schemaName, description: "Auto-inferred schema", split: "transfer", fields: [] };
  }

  const sampleRows = rows.slice(0, 200);
  const keys = new Set<string>();
  for (const r of sampleRows) {
    for (const k of Object.keys(r)) keys.add(k);
  }

  const fields: FieldDef[] = [];
  const canonicalKeys = new Set<string>();

  for (const key of keys) {
    const lowKey = key.trim().toLowerCase().replace(/\s+/g, "_");
    if (canonicalKeys.has(lowKey)) throw new Error(`Columns conflict after normalization: ${key}`);
    canonicalKeys.add(lowKey);

    const nonNullVals: unknown[] = [];
    for (const r of sampleRows) {
      const v = r[key];
      if (v !== null && v !== undefined && v !== "") {
        nonNullVals.push(v);
      }
    }
    if (nonNullVals.length === 0) continue;

    const aliases = buildFieldAliases(key);

    // 1. Check BOOL
    const allBool = nonNullVals.every(
      (v) =>
        typeof v === "boolean" ||
        (typeof v === "string" &&
          ["true", "false", "yes", "no"].includes(v.trim().toLowerCase())),
    );
    if (allBool) {
      fields.push({
        name: lowKey,
        sourceKey: key,
        kind: "BOOL",
        aliases,
        enumValues: [],
      });
      continue;
    }

    // 2. Check NUMBER
    const numericVals: number[] = [];
    let allNumeric = true;
    for (const v of nonNullVals) {
      if (typeof v === "number" && Number.isFinite(v)) {
        numericVals.push(v);
      } else if (typeof v === "string") {
        const cleaned = v.replace(/[$,%\s]/g, "");
        const parsed = Number(cleaned);
        if (Number.isFinite(parsed) && cleaned.length > 0) {
          numericVals.push(parsed);
        } else {
          allNumeric = false;
          break;
        }
      } else {
        allNumeric = false;
        break;
      }
    }

    if (allNumeric && numericVals.length > 0) {
      fields.push({
        name: lowKey,
        sourceKey: key,
        kind: "NUMBER",
        aliases,
        enumValues: [],
      });
      continue;
    }

    // 3. ISO dates only; relative dates need a reference clock and are rejected by the parser.
    if (nonNullVals.every((v) => typeof v === "string" && isIsoDate(v))) {
      fields.push({ name: lowKey, sourceKey: key, kind: "DATE", aliases, enumValues: [] });
      continue;
    }

    // 4. Check ENUM (short categorical strings)
    const strVals = nonNullVals
      .map((v) => String(v).trim().toLowerCase().replace(/\s+/g, "_"))
      .filter((s) => s.length > 0 && s.length <= 32);

    const uniqueEnums = Array.from(new Set(strVals));
    const maxAllowedDistinct = Math.max(12, Math.floor(sampleRows.length * 0.8));
    if (uniqueEnums.length >= 1 && uniqueEnums.length <= maxAllowedDistinct && !["name", "company", "email", "client"].includes(lowKey)) {
      const kind: FieldKind = "ENUM";
      fields.push({
        name: lowKey,
        sourceKey: key,
        kind,
        aliases,
        enumValues: uniqueEnums,
      });
    } else if (nonNullVals.every((v) => typeof v === "string")) {
      fields.push({ name: lowKey, sourceKey: key, kind: "TEXT", aliases, enumValues: [] });
    }
  }

  return {
    name: schemaName,
    description: "Auto-inferred schema from raw rows",
    split: "transfer",
    fields,
  };
}

/**
 * Parses either a JSON array string or a simple CSV string into an array of row objects
 * so users can paste any raw table into Sieve and immediately query it in natural English.
 */
export function normalizeImportedRow(row: Record<string, unknown>, schema: SchemaDef): Record<string, string | number | boolean> {
  const result: Record<string, string | number | boolean> = {};
  for (const field of schema.fields) {
    const raw = row[field.sourceKey ?? field.name];
    if (raw === null || raw === undefined || raw === "") continue;
    const value = String(raw).trim();
    if (field.kind === "NUMBER") {
      const number = Number(value.replace(/[$,%\s]/g, ""));
      if (Number.isFinite(number)) result[field.name] = number;
    } else if (field.kind === "BOOL") {
      if (typeof raw === "boolean") result[field.name] = raw;
      else if (["true", "yes"].includes(value.toLowerCase())) result[field.name] = true;
      else if (["false", "no"].includes(value.toLowerCase())) result[field.name] = false;
    } else if (field.kind === "ENUM") {
      result[field.name] = value.toLowerCase().replace(/\s+/g, "_");
    } else if (field.kind === "DATE") {
      if (isIsoDate(value)) result[field.name] = value;
    } else result[field.name] = value;
  }
  return result;
}

export function isIsoDate(value: string): boolean {
  const time = Date.parse(value);
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(time) && new Date(time).toISOString().slice(0, 10) === value;
}

export function parseCsvOrJsonToRows(rawInput: string): Array<Record<string, unknown>> {
  const trimmed = rawInput.trim();
  if (!trimmed) return [];

  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed);
    const arr = Array.isArray(parsed) ? parsed : [parsed];
    if (!arr.every((item) => item && typeof item === "object" && !Array.isArray(item))) throw new Error("JSON must contain row objects");
    return arr as Array<Record<string, unknown>>;
  }

  const records: string[][] = [];
  let record: string[] = [];
  let cell = "";
  let quoted = false;
  let closedQuote = false;
  for (let i = 0; i < trimmed.length; i++) {
    const ch = trimmed[i];
    if (quoted) {
      if (ch === '"' && trimmed[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') { quoted = false; closedQuote = true; }
      else cell += ch;
    } else if (ch === '"' && !cell.trim() && !closedQuote) quoted = true;
    else if (ch === ",") { record.push(cell.trim()); cell = ""; closedQuote = false; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && trimmed[i + 1] === "\n") i++;
      record.push(cell.trim()); records.push(record); record = []; cell = ""; closedQuote = false;
    } else if (closedQuote && !/\s/.test(ch)) throw new Error("Invalid CSV after closing quote");
    else cell += ch;
  }
  if (quoted) throw new Error("Unclosed CSV quote");
  record.push(cell.trim()); records.push(record);
  if (records.length < 2) throw new Error("CSV requires a header and at least one row");
  const headers = records[0].map((h) => h.trim());
  if (headers.some((h) => !h) || new Set(headers).size !== headers.length) throw new Error("CSV headers must be nonempty and unique");
  const rows: Array<Record<string, unknown>> = [];
  for (let i = 1; i < records.length; i++) {
    const cells = records[i];
    if (cells.length !== headers.length) throw new Error(`CSV row ${i + 1} has ${cells.length} cells; expected ${headers.length}`);
    const row: Record<string, unknown> = {};
    for (let c = 0; c < headers.length; c++) {
      const h = headers[c];
      const rawCell = cells[c];
      if (rawCell.toLowerCase() === "true") {
        row[h] = true;
      } else if (rawCell.toLowerCase() === "false") {
        row[h] = false;
      } else if (/^-?\$?\d+(?:,\d{3})*(?:\.\d+)?$/.test(rawCell)) {
        row[h] = Number(rawCell.replace(/[$,]/g, ""));
      } else {
        row[h] = rawCell;
      }
    }
    rows.push(row);
  }
  return rows;
}
