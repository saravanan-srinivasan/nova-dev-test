// Frontend API client — wraps the FastAPI backend.
// Uses environment variable VITE_API_URL in production, or proxied /api in development

const API_BASE = import.meta.env.VITE_API_URL + "/api";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = `Request failed: ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  analyze: (payload) => request('/analyze', { method: 'POST', body: JSON.stringify(payload) }),

  followup: (sessionId, message) =>
    request('/followup', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, message }),
    }),

  health: (code, language) =>
    request('/health', {
      method: 'POST',
      body: JSON.stringify({ code, language }),
    }),

  complexity: (code, language) =>
    request('/complexity', {
      method: 'POST',
      body: JSON.stringify({ code, language }),
    }),

  listSessions: () => request('/sessions'),
  getSession: (id) => request(`/sessions/${id}`),
  deleteSession: (id) => request(`/sessions/${id}`, { method: 'DELETE' }),
  stats: () => request('/stats'),
  modes: () => request('/modes'),
};

/**
 * Stream the analyze pipeline using Server-Sent Events.
 * Events emitted: pipeline_plan, pipeline_step, llm_call_start, response, done, error
 *
 * Returns an unsubscribe function.
 */
export function streamAnalyze(payload, handlers) {
  const controller = new AbortController();

  fetch(`${API_BASE}/analyze/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const text = await res.text();
        handlers.onError?.(new Error(`Stream failed: ${res.status} ${text}`));
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Split on double newline (SSE event boundary)
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const block of events) {
          const lines = block.split('\n');
          let event = 'message';
          let data = '';
          for (const ln of lines) {
            if (ln.startsWith('event: ')) event = ln.slice(7).trim();
            else if (ln.startsWith('data: ')) data += ln.slice(6);
          }
          if (data) {
            try {
              const parsed = JSON.parse(data);
              handlers.onEvent?.(event, parsed);
            } catch (e) {
              handlers.onError?.(e);
            }
          }
        }
      }
    })
    .catch((e) => {
      if (e.name !== 'AbortError') handlers.onError?.(e);
    });

  return () => controller.abort();
}
