import { useState, useRef, useEffect } from 'react';
import { Plus, MessageSquare, PanelLeftClose, Settings, LogOut, HelpCircle, UserCircle, Search, ChevronRight } from 'lucide-react';

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
            <div className="hidden md:flex flex-col flex-shrink-0 w-[60px] h-screen bg-zinc-950 border-r border-white/5 transition-all duration-300 p-3 relative z-50 items-center justify-between">
                <button
                    onClick={toggleSidebar}
                    className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors mt-1"
                    title="Open sidebar"
                >
                    <PanelLeftClose className="w-5 h-5" />
                </button>

                {/* Collapsed Profile Icon */}
                <button className="p-1.5 rounded-lg hover:bg-zinc-800 transition-colors mb-2">
                    <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center text-white font-medium text-xs shadow-lg shadow-emerald-900/20">
                        VS
                    </div>
                </button>
            </div>
        );
    }

    return (
        <div className="flex flex-col flex-shrink-0 w-[260px] h-screen bg-zinc-950 transition-all duration-300 relative z-50 border-r border-white/5">

            {/* Header: Toggle + New Chat */}
            <div className="flex flex-col gap-3 p-3 pb-2">
                <div className="flex items-center justify-between px-2">
                    <button
                        onClick={toggleSidebar}
                        className="p-2 -ml-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
                        title="Close sidebar"
                    >
                        <PanelLeftClose className="w-5 h-5" />
                    </button>

                    <button
                        onClick={() => onSelectChat(null)}
                        className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
                    >
                        <Plus className="w-5 h-5" />
                    </button>
                </div>

                <button
                    onClick={() => onSelectChat(null)}
                    className="flex items-center w-full gap-3 px-3 py-2.5 rounded-xl bg-white hover:bg-zinc-200 transition-all text-sm text-zinc-900 shadow-sm group">
                    <Plus className="w-4 h-4" />
                    <span className="font-semibold">New chat</span>
                </button>
            </div>

            {/* History List */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden mb-2 px-3 scrollbar-thin">
                <div className="flex flex-col gap-1">
                    <h3 className="px-2 text-xs font-semibold text-zinc-500 mt-4 mb-2 uppercase tracking-wider">Recent</h3>
                    {sessions.map((session) => (
                        <button
                            key={session.id}
                            onClick={() => onSelectChat(session.id)}
                            className={`flex items-center gap-2.5 px-3 py-2.5 w-full rounded-lg hover:bg-zinc-800/50 transition-all text-sm text-zinc-300 group overflow-hidden relative ${currentSessionId === session.id ? 'bg-zinc-800 text-white shadow-sm' : ''}`}
                        >
                            {/* <MessageSquare className={`w-4 h-4 shrink-0 transition-colors ${currentSessionId === session.id ? 'text-emerald-400' : 'text-zinc-500 group-hover:text-zinc-400'}`} /> */}
                            <span className="truncate flex-1 text-left relative z-10">{session.title}</span>
                            {currentSessionId === session.id && <ChevronRight className="w-3 h-3 text-zinc-500" />}
                        </button>
                    ))}
                </div>
            </div>

            {/* Profile Section */}
            <div className="p-3 border-t border-white/5" ref={profileRef}>
                {showProfileMenu && (
                    <div className="absolute bottom-16 left-3 right-3 bg-zinc-900 border border-white/10 rounded-xl shadow-2xl shadow-black/50 p-1.5 overflow-hidden animate-in fade-in zoom-in-95 duration-200 slide-in-from-bottom-2">
                        <div className="px-3 py-3 border-b border-white/5 mb-1">
                            <div className="text-sm font-semibold text-white">Venkiteshiva</div>
                            <div className="text-xs text-zinc-400">user@medicortex.ai</div>
                        </div>

                        <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-zinc-800 text-sm text-zinc-300 transition-colors">
                            <UserCircle className="w-4 h-4" />
                            Personalization
                        </button>
                        <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-zinc-800 text-sm text-zinc-300 transition-colors">
                            <Settings className="w-4 h-4" />
                            Settings
                        </button>
                        <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-zinc-800 text-sm text-zinc-300 transition-colors">
                            <HelpCircle className="w-4 h-4" />
                            Help
                        </button>
                        <div className="h-px bg-white/5 my-1" />
                        <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg hover:bg-red-500/10 hover:text-red-400 text-sm text-zinc-300 transition-colors">
                            <LogOut className="w-4 h-4" />
                            Sign out
                        </button>
                    </div>
                )}

                <button
                    onClick={() => setShowProfileMenu(!showProfileMenu)}
                    className="flex items-center gap-3 w-full px-2 py-2 rounded-xl hover:bg-zinc-800 transition-colors text-sm text-white group"
                >
                    <div className="w-9 h-9 rounded-full bg-emerald-600 flex items-center justify-center text-white font-medium text-xs shadow-md shadow-emerald-900/20 ring-2 ring-transparent group-hover:ring-emerald-500/30 transition-all">
                        VS
                    </div>
                    <div className="flex-1 text-left">
                        <div className="font-medium text-sm">Venkiteshiva</div>
                        <div className="text-xs text-zinc-500">Free Plan</div>
                    </div>
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
