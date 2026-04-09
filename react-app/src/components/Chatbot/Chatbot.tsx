import { useState, useRef, useEffect } from 'react';
import { useChat } from '../../hooks/useChat';
import styles from './Chatbot.module.css';

export function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const { messages, isLoading, sendMessage, clearChat } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (inputValue.trim() === '' || isLoading) return;

    const message = inputValue;
    setInputValue('');
    await sendMessage(message);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      <button
        className={styles.chatBubble}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Open chat"
      >
        💬
      </button>

      {isOpen && (
        <div className={styles.chatPanel}>
          <div className={styles.header}>
            <h2 className={styles.headerTitle}>Help Assistant</h2>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button
                className={styles.clearButton}
                onClick={clearChat}
                title="Clear chat history"
              >
                Clear
              </button>
              <button
                className={styles.closeButton}
                onClick={() => setIsOpen(false)}
                aria-label="Close chat"
              >
                ✕
              </button>
            </div>
          </div>

          <div className={styles.messagesContainer}>
            {messages.length === 0 && (
              <div
                style={{
                  textAlign: 'center',
                  color: '#999',
                  fontSize: '14px',
                  marginTop: '20px',
                }}
              >
                <p>Welcome! How can I help you?</p>
                <p style={{ fontSize: '12px' }}>
                  Ask me anything about the user manual.
                </p>
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={styles.message}>
                <div
                  className={`${styles.messageContent} ${
                    message.role === 'user'
                      ? styles.userMessage
                      : styles.assistantMessage
                  }`}
                  style={{
                    justifyContent:
                      message.role === 'user' ? 'flex-end' : 'flex-start',
                  }}
                >
                  <div
                    className={
                      message.role === 'user'
                        ? styles.userMessageContent
                        : styles.assistantMessageContent
                    }
                  >
                    {message.content}
                    {message.sources && message.sources.length > 0 && (
                      <div className={styles.sources}>
                        [Source: {message.sources.join(', ')}]
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className={styles.message}>
                <div className={styles.messageContent}>
                  <div
                    className={styles.assistantMessageContent}
                    style={{ padding: '12px' }}
                  >
                    <div className={styles.typingIndicator}>
                      <div className={styles.dot}></div>
                      <div className={styles.dot}></div>
                      <div className={styles.dot}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className={styles.inputContainer}>
            <input
              className={styles.input}
              type="text"
              placeholder="Type your question..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
            <button
              className={styles.sendButton}
              onClick={handleSendMessage}
              disabled={isLoading || inputValue.trim() === ''}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}
