import React, { useState } from 'react';
import axios from 'axios';

interface Citation {
  index: number;
  chunk_id: string;
  paper_title: string;
  arxiv_id: string;
  section: string;
  text: string;
  score: number;
}

interface QueryResponse {
  answer: string;
  citations: Citation[];
  grounding_confidence: number;
  self_check_passed: boolean;
  self_check_reasoning: string;
  query_type: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  grounding_confidence?: number;
  self_check_passed?: boolean;
  self_check_reasoning?: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedCitations, setExpandedCitations] = useState<number | null>(null);
  const [showCitationsPanel, setShowCitationsPanel] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post<QueryResponse>(`${API_URL}/query`, {
        question: input
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.answer,
        citations: response.data.citations,
        grounding_confidence: response.data.grounding_confidence,
        self_check_passed: response.data.self_check_passed,
        self_check_reasoning: response.data.self_check_reasoning
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your question. Please make sure the backend is running.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const getConfidenceBadge = (confidence?: number, passed?: boolean) => {
    if (confidence === undefined) return null;
    
    if (passed && confidence >= 0.7) {
      return (
        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
          ✓ Grounded ({(confidence * 100).toFixed(0)}%)
        </span>
      );
    } else if (passed) {
      return (
        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
          ⚠ Medium Confidence ({(confidence * 100).toFixed(0)}%)
        </span>
      );
    } else {
      return (
        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
          ✗ Low Confidence ({(confidence * 100).toFixed(0)}%)
        </span>
      );
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">PaperPilot</h1>
          <button
            onClick={() => setShowCitationsPanel(!showCitationsPanel)}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
          >
            {showCitationsPanel ? 'Hide Citations' : 'Show Citations'}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat Area */}
        <div className={`flex-1 flex flex-col ${showCitationsPanel ? 'mr-96' : ''}`}>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.length === 0 && (
                <div className="text-center py-12">
                  <h2 className="text-xl font-semibold text-gray-700 mb-2">
                    Welcome to PaperPilot
                  </h2>
                  <p className="text-gray-500">
                    Ask questions about research papers from arXiv. Answers include citations and grounding verification.
                  </p>
                </div>
              )}

              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-2xl rounded-2xl px-6 py-4 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                    }`}
                  >
                    {message.role === 'assistant' && (
                      <div className="mb-2 flex items-center gap-2">
                        {getConfidenceBadge(message.grounding_confidence, message.self_check_passed)}
                      </div>
                    )}
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    
                    {message.role === 'assistant' && message.citations && message.citations.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs text-gray-500 mb-2">Citations:</p>
                        <div className="flex flex-wrap gap-2">
                          {message.citations.map((citation, idx) => (
                            <button
                              key={idx}
                              onClick={() => setExpandedCitations(expandedCitations === idx ? null : idx)}
                              className="px-2 py-1 text-xs bg-blue-50 text-blue-700 rounded hover:bg-blue-100 transition-colors"
                            >
                              [{citation.index}] {citation.paper_title.substring(0, 30)}...
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white rounded-2xl px-6 py-4 shadow-sm border border-gray-200">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 bg-white p-6">
            <div className="max-w-3xl mx-auto">
              <div className="flex gap-3">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask a question about research papers..."
                  className="flex-1 resize-none rounded-lg border border-gray-300 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={2}
                  disabled={isLoading}
                />
                <button
                  onClick={sendMessage}
                  disabled={isLoading || !input.trim()}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Citations Panel */}
        {showCitationsPanel && (
          <div className="w-96 bg-white border-l border-gray-200 overflow-y-auto">
            <div className="p-4 border-b border-gray-200">
              <h3 className="font-semibold text-gray-900">Citations</h3>
            </div>
            <div className="p-4 space-y-4">
              {messages
                .filter(m => m.role === 'assistant' && m.citations && m.citations.length > 0)
                .map((message, msgIdx) => (
                  <div key={msgIdx} className="border-b border-gray-100 pb-4 last:border-0">
                    <p className="text-xs text-gray-500 mb-2">From response {msgIdx + 1}:</p>
                    {message.citations?.map((citation, citIdx) => (
                      <div
                        key={citIdx}
                        className="mb-3 p-3 bg-gray-50 rounded-lg text-sm"
                      >
                        <div className="font-medium text-gray-900 mb-1">
                          [{citation.index}] {citation.paper_title}
                        </div>
                        <div className="text-xs text-gray-500 mb-2">
                          {citation.arxiv_id} • {citation.section}
                        </div>
                        <p className="text-gray-700 text-xs line-clamp-4">
                          {citation.text}
                        </p>
                        <div className="mt-2 text-xs text-gray-400">
                          Score: {citation.score.toFixed(3)}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              {messages.filter(m => m.role === 'assistant' && m.citations && m.citations.length > 0).length === 0 && (
                <p className="text-gray-500 text-sm text-center py-8">
                  No citations yet. Ask a question to see sources.
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
