/**
 * GET /api/definitions?word=<word>
 *
 * Returns the word's pocket definition from the dictionary DB.
 * Looks up the word in the words table, joins to definitions by id.
 * Form-of words (e.g. "houses") share their base word's definition id.
 *
 * Response: { word, found, definition }
 */
export async function handleDefinitions(request, env) {
  const url = new URL(request.url);
  const rawWord = (url.searchParams.get('word') || '').trim();

  if (!rawWord) {
    return json({ error: 'word parameter required' }, 400);
  }

  const lower = rawWord.toLowerCase();

  const row = await env.DB.prepare(
    'SELECT w.word, d.definition FROM words w JOIN definitions d ON d.id = w.id WHERE w.word = ? LIMIT 1'
  ).bind(lower).first();

  if (!row) {
    return json({ word: lower, found: false, definition: null }, 404);
  }

  return json({
    word: row.word,
    found: true,
    definition: row.definition,
  });
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
