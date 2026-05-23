import { useState, useRef, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import Dashboard from './pages/Dashboard';
import type { SessionState } from './types';

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(() => {
    const match = window.location.pathname.match(/\/chat\/(.+)/);
    return match ? match[1] : null;
  });

  // Per-session state cache. Stores messages + isLoading for every session,
  // including sessions with active background streams. Using a ref so background
  // stream writes don't trigger re-renders in the whole tree.
  const sessionCache = useRef<Map<string, SessionState>>(new Map());

  // Render trigger: increments ONLY when the currently visible session's data
  // changes. Background session updates mutate sessionCache silently.
  const [activeTick, setActiveTick] = useState(0);

  // Ref mirror of currentSessionId so async SSE handlers can read it without
  // capturing stale closures.
  const currentSessionIdRef = useRef<string | null>(currentSessionId);

  const handleSelectChat = useCallback((id: string | null) => {
    // Update the ref synchronously before setState so any stream event that fires
    // between now and the next render sees the new active session immediately.
    currentSessionIdRef.current = id;
    setCurrentSessionId(id);
  }, []);

  // Call this after every sessionCache mutation. Triggers a re-render only when
  // the mutated session is the one currently displayed.
  const bumpIfActive = useCallback((sessionId: string | null) => {
    if (sessionId === currentSessionIdRef.current) {
      setActiveTick(t => t + 1);
    }
  }, []);

  const handleSetSessionId = useCallback((id: string) => {
    currentSessionIdRef.current = id;
    setCurrentSessionId(id);
  }, []);

  if (window.location.pathname === '/dashboard') {
    return <Dashboard />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#212121]">
      <Sidebar
        isOpen={isSidebarOpen}
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onSelectChat={handleSelectChat}
        currentSessionId={currentSessionId}
      />
      <div className="flex-1 h-full">
        <ChatArea
          isSidebarOpen={isSidebarOpen}
          toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          sessionId={currentSessionId}
          setSessionId={handleSetSessionId}
          sessionCache={sessionCache}
          bumpIfActive={bumpIfActive}
          activeTick={activeTick}
        />
      </div>
    </div>
  );
}

export default App;
