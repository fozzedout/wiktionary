/**
 * API key authentication.
 * Keys stored via: npx wrangler secret put API_KEYS
 * Format: comma-separated list of valid keys.
 */
export function authenticate(request, env) {
  const apiKey = request.headers.get('X-API-Key')
    || new URL(request.url).searchParams.get('api_key');

  if (!apiKey) {
    return { ok: false, status: 401, error: 'API key required' };
  }

  const validKeys = (env.API_KEYS || '')
    .split(',')
    .map(k => k.trim())
    .filter(Boolean);

  if (!validKeys.length || !validKeys.includes(apiKey)) {
    return { ok: false, status: 403, error: 'Invalid API key' };
  }

  return { ok: true, key: apiKey };
}
