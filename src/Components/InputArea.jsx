import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, Image as ImageIcon, Sparkles } from 'lucide-react';
import './InputArea.css';

const InputArea = ({ onSendMessage, isLoading }) => {
  const [prompt, setPrompt] = useState('');
  const [mode, setMode] = useState('text'); // 'text' or 'image'
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [prompt]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (prompt.trim() && !isLoading) {
      onSendMessage(prompt, mode);
      setPrompt('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const currentPlaceholder = mode === 'image' 
    ? "Describe the image you want to generate..." 
    : "Ask KIA AI anything...";

  return (
    <div className="input-area-container">
      <form onSubmit={handleSubmit} className={`input-wrapper ${mode}`}>
        <div className="input-actions-left">
          <button 
            type="button" 
            className={`mode-toggle ${mode === 'image' ? 'active' : ''}`}
            onClick={() => setMode(mode === 'text' ? 'image' : 'text')}
            title={mode === 'image' ? "Switch to Text" : "Switch to Image Generation"}
          >
            {mode === 'image' ? <ImageIcon size={20} /> : <Sparkles size={20} />}
          </button>
        </div>
        
        <textarea
          ref={textareaRef}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          placeholder={currentPlaceholder}
          rows="1"
          disabled={isLoading}
        />
        
        <div className="input-actions-right">
          <button type="button" className="icon-btn" title="Use microphone">
            <Mic size={20} />
          </button>
          <button 
            type="submit" 
            className={`send-btn ${prompt.trim() ? 'active' : ''}`}
            disabled={!prompt.trim() || isLoading}
          >
            <Send size={20} />
          </button>
        </div>
      </form>
      <div className="footer-text">
        KIA AI may display inaccurate info, including about people, so double-check its responses.
      </div>
    </div>
  );
};

export default InputArea;
