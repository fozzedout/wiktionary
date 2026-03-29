/**
 * GET /api/validate?word=<word>
 *
 * Lightweight check: is this word in the dictionary?
 * Clients should prefer the trie for bulk validation;
 * this endpoint is a server-side fallback.
 */
export async function handleValidate(request, env) {
  const url = new URL(request.url);
  const word = (url.searchParams.get('word') || '').trim().toLowerCase();

  if (!word) {
    return json({ error: 'word parameter required' }, 400);
  }

  const row = await env.DB.prepare(
    'SELECT 1 FROM words WHERE word = ? LIMIT 1'
  ).bind(word).first();

  return json({ word, valid: !!row });
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
