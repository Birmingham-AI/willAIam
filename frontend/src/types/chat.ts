/**
 * Simple chat message interface
 */
export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  traceId?: string;  // For assistant messages, used for feedback
}

/**
 * Props for the MessageInput component
 */
export interface MessageInputProps {
  inputMessage: string;
  setInputMessage: (message: string) => void;
  handleSendMessage: (enableWebSearch?: boolean) => void;
  isLoading: boolean;
  cancelStreaming?: () => void;
  onNewChat?: () => void;
}
