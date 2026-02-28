import React, { useState, useEffect } from 'react';
import { Menu, Plus, MessageSquare, Menu as MenuIcon, Trash2, Search } from 'lucide-react';
import ChatHistory from './components/ChatHistory';
import InputArea from './components/InputArea';
import { generateContent, generateStreamContent } from './services/apiService';
import { loadChatHistory, createChat, updateChatMessages, deleteChat as dbDeleteChat } from './services/dbService';
import { supabase } from './services/supabaseClient';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearchMode, setIsSearchMode] = useState(false);

  // Advanced Model Settings
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(150);

  // Default to open on desktop, closed on mobile
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  // Load chat history from Supabase on mount
  useEffect(() => {
    const fetchHistory = async () => {
      const history = await loadChatHistory();
      if (history && history.length > 0) {
        setChatHistory(history);
      } else {
        // Fallback for brand new users
        setChatHistory([
          { id: 1, title: 'Welcome to KIA AI', messages: [{ sender: 'ai', type: 'text', text: 'Hello! I am KIA AI. How can I assist you today?' }] }
        ]);
      }
    };
    fetchHistory();
  }, []);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile && !sidebarOpen) {
        // Keep user preference on desktop, or optionally force open
      } else if (mobile) {
        setSidebarOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sidebarOpen]);

  // Update the current chat's stored messages internally and in cloud
  useEffect(() => {
    if (currentChatId && messages.length > 0) {
      setChatHistory(prev => prev.map(chat =>
        chat.id === currentChatId ? { ...chat, messages: messages } : chat
      ));

      // Save updates to Supabase (debouncing ignored for simplicity here)
      updateChatMessages(currentChatId, messages);
    }
  }, [messages, currentChatId]);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleSendMessage = async (text, mode) => {
    let activeChatId = currentChatId;

    // Add user message to current view
    const newMessages = [...messages, { sender: 'user', type: 'text', text }];
    setMessages(newMessages);

    // If this is the very first message in a new chat, add it to the sidebar history
    if (!currentChatId) {
      activeChatId = Date.now();
      setCurrentChatId(activeChatId);

      // Take the first 30 chars of the text as the title
      const title = text.length > 30 ? text.substring(0, 30) + "..." : text;

      setChatHistory(prev => [{ id: activeChatId, title, messages: newMessages }, ...prev]);

      // Save new chat to Supabase
      createChat(activeChatId, title, newMessages);
    }

    setIsLoading(true);

    // Call API (Colab Backend)
    if (mode === 'text') {
      let firstChunkReceived = false;

      try {
        await generateStreamContent(text, mode, (chunkText) => {
          if (!firstChunkReceived) {
            setIsLoading(false);
            firstChunkReceived = true;
            // Append an empty AI message immediately upon first response
            setMessages(prev => [...prev, { sender: 'ai', type: 'text', text: chunkText }]);
          } else {
            setMessages(prev => {
              const lastIndex = prev.length - 1;
              const updatedMessages = [...prev];
              updatedMessages[lastIndex] = {
                ...updatedMessages[lastIndex],
                text: updatedMessages[lastIndex].text + chunkText
              };
              return updatedMessages;
            });
          }
        }, { temperature: temperature, max_tokens: maxTokens }); // Inject parameters here
      } catch (error) {
        setMessages(prev => {
          const lastIndex = prev.length - 1;
          const updatedMessages = [...prev];
          updatedMessages[lastIndex] = {
            ...updatedMessages[lastIndex],
            text: updatedMessages[lastIndex].text + "\n\n[Stream Error]"
          };
          return updatedMessages;
        });
      }
    } else {
      // Image generation logic (stays standard request-response)
      try {
        const response = await generateContent(text, mode);
        setMessages([...newMessages, { sender: 'ai', type: 'image', imageUrl: response.content }]);
      } catch (error) {
        setMessages([...newMessages, { sender: 'ai', type: 'text', text: "An error occurred fetching the image." }]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setCurrentChatId(null);
    setIsSearchMode(false);
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  const loadChat = (chatId) => {
    const chatToLoad = chatHistory.find(c => c.id === chatId);
    if (chatToLoad) {
      setCurrentChatId(chatId);
      setMessages(chatToLoad.messages || []);
      setIsSearchMode(false);
      if (isMobile) {
        setSidebarOpen(false);
      }
    }
  };

  const deleteChat = async (e, chatId) => {
    e.stopPropagation(); // Avoid triggering the loadChat click

    // Remove from history locally
    setChatHistory(prev => prev.filter(c => c.id !== chatId));

    // Remove from Supabase
    await dbDeleteChat(chatId);

    // If we're currently viewing the deleted chat, clear the main window
    if (currentChatId === chatId) {
      setMessages([]);
      setCurrentChatId(null);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Overlay for Mobile */}
      {isMobile && sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)}></div>
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          {/* Hide the inner menu button if on mobile */}
          {!isMobile && (
            <button className="icon-btn" onClick={toggleSidebar}>
              <MenuIcon size={24} />
            </button>
          )}

          {/* Search Icon (Desktop) or Input (Mobile) */}
          {sidebarOpen && (
            <div style={{ display: 'flex', alignItems: 'center' }}>
              {isMobile ? (
                <input
                  type="text"
                  placeholder="Search chats..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text-primary)',
                    padding: '8px 12px',
                    borderRadius: 'var(--border-radius-xl)',
                    width: '100%',
                    fontSize: '0.9rem',
                    border: '1px solid var(--border-color)',
                    marginLeft: '8px'
                  }}
                />
              ) : (
                <button
                  className="icon-btn"
                  onClick={() => setIsSearchMode(true)}
                  title="Search historical chats"
                >
                  <Search size={20} />
                </button>
              )}
            </div>
          )}
        </div>

        <button className="new-chat-btn" onClick={handleNewChat}>
          <Plus size={20} />
          {sidebarOpen && <span>New chat</span>}
        </button>

        <div className="recent-chats" style={{ opacity: sidebarOpen ? 1 : 0, pointerEvents: sidebarOpen ? 'auto' : 'none', transition: 'opacity 0.2s' }}>
          <p className="section-title">Recent</p>
          {chatHistory
            .filter(chat => chat.title.toLowerCase().includes(searchQuery.toLowerCase()))
            .map((chat) => (
              <div
                key={chat.id}
                className={`history-item ${chat.id === currentChatId ? 'active' : ''}`}
                onClick={() => loadChat(chat.id)}
              >
                <MessageSquare size={16} />
                <span className="history-title">{chat.title}</span>
                <button
                  className="delete-btn"
                  onClick={(e) => deleteChat(e, chat.id)}
                  title="Delete chat"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
        </div>

        {/* Model Settings Panel */}
        <div className="settings-panel" style={{ opacity: sidebarOpen ? 1 : 0, transition: 'opacity 0.2s', marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid var(--border-color)' }}>
          <div className="settings-header" onClick={() => setShowSettings(!showSettings)} style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-secondary)' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>Advanced Model Settings</span>
            <span style={{ fontSize: '0.8rem' }}>{showSettings ? '▾' : '▸'}</span>
          </div>

          {showSettings && (
            <div className="settings-controls" style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div className="slider-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <span>Temperature (Creativity)</span>
                  <span>{temperature.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
                />
              </div>

              <div className="slider-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <span>Max Tokens</span>
                  <span>{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="500"
                  step="10"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
                />
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="top-nav">
          <div className="nav-left" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* Mobile Menu Toggle Button */}
            {isMobile && !sidebarOpen && (
              <button
                className="icon-btn mobile-menu-btn"
                onClick={toggleSidebar}
                style={{ color: 'var(--text-secondary)' }}
                title="Open Menu"
              >
                <MenuIcon size={24} />
              </button>
            )}
            <div className="nav-title" style={{ marginLeft: (!isMobile && !sidebarOpen) ? '16px' : '0' }}>
              <span className="kia-title">KIA AI</span>
              <span className="version-badge">2B Model</span>
            </div>
          </div>

          <div className="nav-right" style={{ display: 'flex', alignItems: 'center', marginLeft: 'auto', gap: '12px' }}>
            <div className="user-profile">
              <div className="avatar">R</div>
            </div>
          </div>
        </header>

        {isSearchMode && !isMobile ? (
          <div className="search-page" style={{ flex: 1, padding: '2rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ width: '100%', maxWidth: '800px', display: 'flex', flexDirection: 'column', marginTop: '40px' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '32px', color: 'var(--text-primary)', fontWeight: 500 }}>Search</h1>

              <div className="input-wrapper" style={{ marginBottom: '40px', display: 'flex', alignItems: 'center', background: 'var(--bg-secondary)', padding: '16px 20px', borderRadius: 'var(--border-radius-xl)', border: '1px solid var(--border-color)' }}>
                <Search size={24} style={{ color: 'var(--text-secondary)', marginRight: '16px' }} />
                <input
                  type="text"
                  placeholder="Search for chats"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  autoFocus
                  style={{ flex: 1, background: 'transparent', border: 'none', color: 'var(--text-primary)', fontSize: '1.1rem', outline: 'none' }}
                />
              </div>

              <p className="section-title" style={{ marginBottom: '16px', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>Recent</p>

              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {chatHistory
                  .filter(chat => chat.title.toLowerCase().includes(searchQuery.toLowerCase()))
                  .map((chat) => (
                    <div
                      key={chat.id}
                      onClick={() => loadChat(chat.id)}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '16px',
                        borderBottom: '1px solid var(--border-color)',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                    >
                      <span style={{ color: 'var(--text-primary)', fontSize: '0.95rem' }}>{chat.title}</span>
                      <span style={{ color: 'var(--text-placeholder)', fontSize: '0.85rem' }}>
                        {chat.created_at ? new Date(chat.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : 'Today'}
                      </span>
                    </div>
                  ))}
                {chatHistory.filter(chat => chat.title.toLowerCase().includes(searchQuery.toLowerCase())).length === 0 && (
                  <div style={{ color: 'var(--text-placeholder)', padding: '32px', textAlign: 'center' }}>No chats found</div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <>
            <ChatHistory messages={messages} isLoading={isLoading} />
            <div className="input-section">
              <InputArea onSendMessage={handleSendMessage} isLoading={isLoading} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
