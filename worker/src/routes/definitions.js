/**
 * GET /api/definitions?word=<word>
 *
 * Returns the word's short definitions from the dictionary DB.
 * Response: { word, found, definitions: [{ definition, pos }] }
 */
export async function handleDefinitions(request, env) {
  const url = new URL(request.url);
  const rawWord = (url.searchParams.get('word') || '').trim();

  if (!rawWord) {
    return json({ error: 'word parameter required' }, 400);
  }

  const lower = rawWord.toLowerCase();

  const wordRow = await env.DB.prepare(
    'SELECT word, raw FROM words WHERE word = ? LIMIT 1'
  ).bind(lower).first();

  const defs = (await env.DB.prepare(
    'SELECT definition, pos FROM definitions WHERE word = ? ORDER BY idx'
  ).bind(lower).all()).results || [];

  if (!wordRow && !defs.length) {
    return json({ word: lower, found: false, definitions: [] }, 404);
  }

  return json({
    word: wordRow?.raw || lower,
    found: true,
    definitions: defs,
  });
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
