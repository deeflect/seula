// Stores the yes/no verdicts the live page posts. Static files are served before this runs, so it only sees what is left.
// shortcut: no rate limit; the body cap and the KV daily write quota bound abuse. Add one if junk shows up in the data.
export default {
  async fetch(req, env) {
    if (req.method !== "POST" || new URL(req.url).pathname !== "/capture") return new Response("Not found", { status: 404 });
    if (+(req.headers.get("content-length") ?? 0) > 8192) return new Response("Too large", { status: 413 });
    const body = await req.text();
    if (body.length > 8192) return new Response("Too large", { status: 413 });
    let v;
    try { v = JSON.parse(body); } catch { return new Response("Bad request", { status: 400 }); }
    if (!["right", "wrong"].includes(v?.verdict) || typeof v.query !== "string" || !Array.isArray(v.header)) return new Response("Bad request", { status: 400 });
    await env.CAPTURE.put(`${new Date().toISOString()} ${crypto.randomUUID().slice(0, 8)}`, body);
    return new Response(null, { status: 204 });
  },
};
