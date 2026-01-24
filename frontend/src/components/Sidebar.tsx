import { useState, useRef, useEffect } from 'react';
import { Plus, MessageSquare, PanelLeftClose, Settings, LogOut, HelpCircle, UserCircle, Search } from 'lucide-react';

interface SidebarProps {
    isOpen: boolean;
    toggleSidebar: () => void;
}

const Sidebar = ({ isOpen, toggleSidebar }: SidebarProps) => {
    const [showProfileMenu, setShowProfileMenu] = useState(false);
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
                    <button className="p-2 -ml-2 rounded-lg hover:bg-[#212121] text-gray-400 hover:text-white transition-colors">
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
                    <button className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-[#212121] transition-colors text-sm text-[rgb(236,236,241)] text-left group">
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
                    <h3 className="px-2 text-xs font-medium text-gray-500 mt-2 mb-1">Today</h3>
                    {[1, 2].map((i) => (
                        <button
                            key={i}
                            className="flex items-center gap-2 px-2 py-2 w-full rounded-lg hover:bg-[#212121] transition-colors text-sm text-[rgb(236,236,241)] group overflow-hidden relative"
                        >
                            <MessageSquare className="w-4 h-4 shrink-0 text-gray-400" />
                            <span className="truncate flex-1 text-left relative z-10">Previous Conversation {i}</span>
                        </button>
                    ))}
                    <h3 className="px-2 text-xs font-medium text-gray-500 mt-4 mb-1">Yesterday</h3>
                    {[3, 4, 5].map((i) => (
                        <button
                            key={i}
                            className="flex items-center gap-2 px-2 py-2 w-full rounded-lg hover:bg-[#212121] transition-colors text-sm text-[rgb(236,236,241)] group overflow-hidden relative"
                        >
                            <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-[#171717] to-transparent group-hover:from-[#212121]"></div>
                            <MessageSquare className="w-4 h-4 shrink-0 text-gray-400" />
                            <span className="truncate flex-1 text-left relative z-10">Project Discussion {i}</span>
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
