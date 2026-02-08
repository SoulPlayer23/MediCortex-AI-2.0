import { useState, useRef, useEffect } from 'react';
import { Plus, MessageSquare, PanelLeftClose, Settings, LogOut, HelpCircle, UserCircle, Search } from 'lucide-react';

interface SidebarProps {
    isOpen: boolean;
    toggleSidebar: () => void;
    onSelectChat: (sessionId: string | null) => void;
    currentSessionId: string | null;
}

interface ChatSession {
    id: string;
    title: string;
    updated_at: string;
}

const Sidebar = ({ isOpen, toggleSidebar, onSelectChat, currentSessionId }: SidebarProps) => {
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const profileRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
                setShowProfileMenu(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        fetchSessions();
    }, [currentSessionId, isOpen]);

    const fetchSessions = async () => {
        try {
            const res = await fetch('http://localhost:8001/chats');
            if (res.ok) {
                const data = await res.json();
                setSessions(data);
            }
        } catch (e) {
            console.error("Failed to fetch chats", e);
        }
    };

    if (!isOpen) {
        // Render simple collapsed strip
        return (
            <div className="hidden md:flex flex-col flex-shrink-0 w-[60px] h-screen bg-[#171717] border-r border-white/5 transition-all p-3 relative z-50 items-center justify-between">
                <button
                    onClick={toggleSidebar}
                    className="p-2 rounded-lg hover:bg-[#212121] text-gray-400 hover:text-white transition-colors mt-1"
                    title="Open sidebar"
                >
                    <PanelLeftClose className="w-5 h-5" />
                </button>

                {/* Collapsed Profile Icon */}
                <button className="p-1.5 rounded-lg hover:bg-[#212121] transition-colors mb-1">
                    <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-white font-medium text-xs">
                        VS
                    </div>
                </button>
            </div>
        );
    }

    return (
        <div className="flex flex-col flex-shrink-0 w-[260px] h-screen bg-[#171717] transition-all p-3 relative z-50">

            {/* Header: Toggle + New Chat + Search */}
            <div className="flex flex-col gap-2 mb-4">
                <div className="flex items-center justify-between px-2">
                    <button
                        onClick={() => onSelectChat(null)}
                        className="p-2 -ml-2 rounded-lg hover:bg-[#212121] text-gray-400 hover:text-white transition-colors">
                        <Plus className="w-5 h-5" />
                    </button>

                    <button
                        onClick={toggleSidebar}
                        className="p-2 rounded-lg hover:bg-[#212121] text-gray-400 hover:text-white transition-colors"
                        title="Close sidebar"
                    >
                        <PanelLeftClose className="w-5 h-5" />
                    </button>
                </div>

                <div className="flex flex-col gap-1">
                    <button
                        onClick={() => onSelectChat(null)}
                        className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-[#212121] transition-colors text-sm text-[rgb(236,236,241)] text-left group">
                        <div className="flex items-center justify-center w-7 h-7 bg-white text-black rounded-full">
                            <Plus className="w-4 h-4" />
                        </div>
                        <span className="font-medium">New chat</span>
                    </button>

                    <button className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-[#212121] transition-colors text-sm text-[rgb(236,236,241)] text-left group">
                        <div className="flex items-center justify-center w-7 h-7 text-white">
                            <Search className="w-5 h-5" />
                        </div>
                        <span className="font-medium">Search chats</span>
                    </button>
                </div>
            </div>

            {/* History List */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden mb-2 pr-1 scrollbar-thin">
                <div className="flex flex-col gap-2">
                    <h3 className="px-2 text-xs font-medium text-gray-500 mt-2 mb-1">Recent</h3>
                    {sessions.map((session) => (
                        <button
                            key={session.id}
                            onClick={() => onSelectChat(session.id)}
                            className={`flex items-center gap-2 px-2 py-2 w-full rounded-lg hover:bg-[#212121] transition-colors text-sm text-[rgb(236,236,241)] group overflow-hidden relative ${currentSessionId === session.id ? 'bg-[#212121]' : ''}`}
                        >
                            <MessageSquare className="w-4 h-4 shrink-0 text-gray-400" />
                            <span className="truncate flex-1 text-left relative z-10">{session.title}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Profile Section */}
            <div className="relative pt-2" ref={profileRef}>
                {showProfileMenu && (
                    <div className="absolute bottom-full left-0 w-full mb-2 bg-[#2f2f2f] border border-[#424242] rounded-xl shadow-lg p-1.5 overflow-hidden animate-in fade-in zoom-in-95 duration-100">
                        <div className="px-3 py-3 border-b border-[#424242] mb-1">
                            <div className="text-sm font-medium text-white">Venkiteshiva</div>
                            <div className="text-xs text-gray-400">user@medicortex.ai</div>
                        </div>

                        <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-[#424242] text-sm text-gray-200 transition-colors">
                            <UserCircle className="w-4 h-4" />
                            Personalization
                        </button>
                        <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-[#424242] text-sm text-gray-200 transition-colors">
                            <Settings className="w-4 h-4" />
                            Settings
                        </button>
                        <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-[#424242] text-sm text-gray-200 transition-colors">
                            <HelpCircle className="w-4 h-4" />
                            Help
                        </button>
                        <div className="h-px bg-[#424242] my-1" />
                        <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-[#424242] text-sm text-gray-200 transition-colors">
                            <LogOut className="w-4 h-4" />
                            Sign out
                        </button>
                    </div>
                )}

                <button
                    onClick={() => setShowProfileMenu(!showProfileMenu)}
                    className="flex items-center gap-3 w-full px-2 py-2 rounded-lg hover:bg-[#212121] transition-colors text-sm text-white group"
                >
                    <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-white font-medium text-xs">
                        VS
                    </div>
                    <div className="flex-1 text-left font-medium">Venkiteshiva</div>
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
