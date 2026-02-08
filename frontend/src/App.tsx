import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  return (
    <div className="flex h-screen overflow-hidden bg-[#212121]">
      <Sidebar
        isOpen={isSidebarOpen}
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onSelectChat={setCurrentSessionId}
        currentSessionId={currentSessionId}
      />
      <div className="flex-1 h-full">
        <ChatArea
          isSidebarOpen={isSidebarOpen}
          toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          sessionId={currentSessionId}
          setSessionId={setCurrentSessionId}
        />
      </div>
    </div>
  );
}

export default App;
