import { LEX_AFTER_SET, LEX_AND_SET, LEX_BEFORE_SET, LEX_BOOL_FALSE_SET, LEX_BOOL_TRUE_SET, LEX_CONTAINS_SET, LEX_NEGATE_SET, MAX_TOKENS, tokenizeQuery } from "./featurize";
import { cleanNumericValue, NeuralParseResult, SieveModel } from "./model";
import { SchemaDef } from "./schema";

export type NumericOperator = "eq" | "neq" | "gt" | "gte" | "lt" | "lte";

export interface OperatorRewrite {
  start: number; // offsets in originalQuery
  end: number;
  phrase: string;
  replacement: string;
}

const SYMBOL: Record<NumericOperator, string> = {
  eq: "=", neq: "!=", gt: ">", gte: ">=", lt: "<", lte: "<=",
};

const KNOWN: Record<string, NumericOperator> = {
  "=": "eq", "==": "eq", ":": "eq", is: "eq", equals: "eq", equal: "eq", eq: "eq", matching: "eq", as: "eq",
  "!=": "neq", "!==": "neq", "<>": "neq", isnt: "neq", neq: "neq", except: "neq", excluding: "neq", other_than: "neq",
  ">": "gt", over: "gt", above: "gt", greater: "gt", more: "gt", exceeding: "gt", gt: "gt", exceeds: "gt",
  ">=": "gte", gte: "gte", min: "gte", at_least: "gte", atleast: "gte",
  "<": "lt", under: "lt", below: "lt", less: "lt", fewer: "lt", lt: "lt", within: "lt",
  "<=": "lte", lte: "lte", max: "lte", at_most: "lte", atmost: "lte",
  "greater than": "gt", "more than": "gt", "less than": "lt", "fewer than": "lt",
  "at least": "gte", "at most": "lte", "no fewer than": "gte", "no more than": "lte",
};

/** Apply caller-taught comparator words only in an unambiguous numeric field/value triplet. */
export function parseWithOperatorAliases(
  model: SieveModel,
  query: string,
  schema: SchemaDef,
  aliases: Readonly<Record<string, NumericOperator>>,
): { originalQuery: string; normalizedQuery: string; rewrites: OperatorRewrite[]; result: NeuralParseResult } {
  const dictionary = new Map<string, NumericOperator>();
  for (const [raw, target] of Object.entries(aliases)) {
    const phrase = raw.trim().toLowerCase().replace(/\s+/g, " ");
    const words = phrase.split(" ");
    if (!Object.hasOwn(SYMBOL, target) || !/^[a-z]+(?: [a-z]+){0,3}$/.test(phrase) ||
      words.some((word) => word === "or" || word === "either" || word === "not" ||
        LEX_AND_SET.has(word) || (LEX_NEGATE_SET.has(word) && !(word === "no" &&
          (phrase === "no more than" || phrase === "no fewer than"))))) {
      throw new Error(`Invalid numeric operator alias: ${raw}`);
    }
    const recognized = KNOWN[phrase];
    if (recognized && recognized !== target) throw new Error(`Conflicting operator meaning: ${raw}`);
    if (!recognized && words.some((word) => KNOWN[word] && KNOWN[word] !== target)) {
      throw new Error(`Conflicting operator meaning: ${raw}`);
    }
    if (words.some((word) => LEX_CONTAINS_SET.has(word) || LEX_BEFORE_SET.has(word) ||
      LEX_AFTER_SET.has(word) || LEX_BOOL_TRUE_SET.has(word) || LEX_BOOL_FALSE_SET.has(word))) {
      throw new Error(`Conflicting operator meaning: ${raw}`);
    }
    if (dictionary.has(phrase) && dictionary.get(phrase) !== target) throw new Error(`Conflicting operator alias: ${raw}`);
    dictionary.set(phrase, target);
  }

  const tokens = tokenizeQuery(query, MAX_TOKENS + 1);
  const rewrites: OperatorRewrite[] = [];
  if (tokens.length <= MAX_TOKENS && !/\S/.test(query.slice(tokens.at(-1)?.end ?? 0))) {
    const numericFields = new Map<string, string | null>();
    for (const field of schema.fields) {
      for (const name of [field.name, ...field.aliases]) {
        const key = name.toLowerCase();
        if (numericFields.has(key) && numericFields.get(key) !== field.name) numericFields.set(key, null);
        else if (!numericFields.has(key)) numericFields.set(key, field.kind === "NUMBER" ? field.name : null);
      }
    }
    const fieldAt = (index: number): string | undefined => {
      const token = tokens[index];
      return token && !/^['"]/.test(token.text) ? numericFields.get(token.norm.replace(/-/g, "_")) ?? undefined : undefined;
    };
    const valueAt = (index: number, field: string): boolean => {
      const token = tokens[index];
      return !!token && !/^['"]/.test(token.text) && cleanNumericValue(token.norm, field) !== null;
    };
    const adjacent = (left: number, right: number): boolean =>
      !!tokens[left] && !!tokens[right] && /^\s+$/.test(query.slice(tokens[left].end, tokens[right].start));

    for (let i = 0; i < tokens.length; i++) {
      for (let length = Math.min(4, tokens.length - i); length >= 1; length--) {
        const end = i + length - 1;
        if (tokens.slice(i, end + 1).some((token) => /^['"]/.test(token.text)) ||
          tokens.slice(i, end).some((_, offset) => !adjacent(i + offset, i + offset + 1))) continue;
        const phrase = tokens.slice(i, end + 1).map((token) => token.norm).join(" ");
        const target = dictionary.get(phrase);
        if (!target) continue;
        if (tokens.slice(i, end + 1).some((token) => schema.fields.some((field) =>
          [field.name, ...field.aliases, ...field.enumValues].some((word) => word.toLowerCase() === token.norm)))) continue;
        const before = fieldAt(i - 1);
        const after = fieldAt(end + 1);
        const prefixField = fieldAt(end + 2);
        const fieldFirst = !!before && adjacent(i - 1, i) && adjacent(end, end + 1) && valueAt(end + 1, before);
        const valueFirst = !!after && adjacent(i - 1, i) && adjacent(end, end + 1) && valueAt(i - 1, after);
        const operatorFirst = !!prefixField && adjacent(end, end + 1) && adjacent(end + 1, end + 2) && valueAt(end + 1, prefixField);
        if (!fieldFirst && !valueFirst && !operatorFirst) continue;
        const start = tokens[i].start;
        const stop = tokens[end].end;
        rewrites.push({ start, end: stop, phrase: query.slice(start, stop), replacement: SYMBOL[target] });
        i = end;
        break;
      }
    }
  }

  let normalizedQuery = query;
  for (const rewrite of [...rewrites].reverse()) {
    normalizedQuery = normalizedQuery.slice(0, rewrite.start) + rewrite.replacement + normalizedQuery.slice(rewrite.end);
  }
  return { originalQuery: query, normalizedQuery, rewrites, result: model.parse(normalizedQuery, schema) };
}
