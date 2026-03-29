import { authenticate } from './auth.js';
import { handleDefinitions } from './routes/definitions.js';
import { handleValidate } from './routes/validate.js';
import FRONTEND_HTML from './frontend.html';

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type, X-API-Key',
    },
  });
}

function withCORS(response) {
  const headers = new Headers(response.headers);
  headers.set('Access-Control-Allow-Origin', '*');
  headers.set('Access-Control-Allow-Headers', 'Content-Type, X-API-Key');
  return new Response(response.body, {
    status: response.status,
    headers,
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, X-API-Key',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    // Front-end tester — no auth (uses its own key)
    if (url.pathname === '/' || url.pathname === '/index.html') {
      const testKey = env.FRONTEND_TEST_KEY || '';
      const html = FRONTEND_HTML.replace('{{API_KEY}}', testKey);
      return new Response(html, {
        headers: { 'Content-Type': 'text/html; charset=utf-8' },
      });
    }

    // Health check — no auth
    if (url.pathname === '/health') {
      return json({ status: 'ok' });
    }

    // All /api/* routes require auth
    if (url.pathname.startsWith('/api/')) {
      const auth = authenticate(request, env);
      if (!auth.ok) {
        return json({ error: auth.error }, auth.status);
      }
    }

    // Route dispatch
    if (request.method === 'GET') {
      if (url.pathname === '/api/definitions') {
        return withCORS(await handleDefinitions(request, env));
      }
      if (url.pathname === '/api/validate') {
        return withCORS(await handleValidate(request, env));
      }
    }

    return json({ error: 'Not found' }, 404);
  },
};
