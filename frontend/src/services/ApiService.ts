import config from '../config';

/**
 * Simple API service for chat with conversation memory
 */
class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = config.apiBaseUrl;
  }

  /**
   * Send a message and get streaming response (SSE)
   */
  streamMessage(
    message: string,
    enableWebSearch: boolean,
    conversationHistory: Array<{ role: string; content: string }>,
    onChunk: (chunk: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void,
    onTraceId?: (traceId: string) => void
  ): AbortController {
    const abortController = new AbortController();

    fetch(`${this.baseURL}/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: message,
        messages: conversationHistory,
        enable_web_search: enableWebSearch
      }),
      signal: abortController.signal,
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('No reader available');
        }

        let buffer = '';

        const readChunk = (): Promise<void> => {
          return reader.read().then(({ done, value }) => {
            if (done) {
              onComplete();
              return;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');

            // Keep the last potentially incomplete line in the buffer
            buffer = lines.pop() || '';

            let currentEvent = '';

            for (const line of lines) {
              if (line.startsWith('event: ')) {
                currentEvent = line.slice(7).trim();
              } else if (line.startsWith('data: ')) {
                const data = line.slice(6); // Don't trim - preserve spaces

                if (currentEvent === 'trace_id' && onTraceId) {
                  onTraceId(data.trim());
                  currentEvent = '';
                  continue;
                }

                if (data.trim() === '[DONE]') {
                  onComplete();
                  return;
                }
                // Unescape newlines that were escaped in the backend for SSE transport
                const unescapedData = data.replace(/\\n/g, '\n');
                // Send all chunks, even empty ones (might be spaces)
                onChunk(unescapedData);
                currentEvent = '';
              }
            }

            return readChunk();
          });
        };

        return readChunk();
      })
      .catch(error => {
        if (error.name !== 'AbortError') {
          onError(error);
        }
      });

    return abortController;
  }

  /**
   * Submit feedback for a response
   */
  async submitFeedback(
    traceId: string,
    rating: 'like' | 'dislike',
    comment?: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${this.baseURL}/v1/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        trace_id: traceId,
        rating,
        comment
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }
}

// Create and export a singleton instance
const apiService = new ApiService();
export default apiService;
