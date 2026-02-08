import { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import InputArea from './InputArea';
import { Menu, PanelLeftOpen, SquarePen, ChevronDown } from 'lucide-react';
import clsx from 'clsx';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    attachments?: any[];
}

interface ChatAreaProps {
    isSidebarOpen: boolean;
    toggleSidebar: () => void;
    sessionId: string | null;
    setSessionId: (id: string) => void;
}

const ChatArea = ({ isSidebarOpen, toggleSidebar, sessionId, setSessionId }: ChatAreaProps) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        if (sessionId) {
            fetchMessages(sessionId);
        } else {
            setMessages([]);
        }
    }, [sessionId]);

    const fetchMessages = async (id: string) => {
        try {
            const res = await fetch(`http://localhost:8001/chats/${id}`);
            if (res.ok) {
                const data = await res.json();
                setMessages(data);
            }
        } catch (e) {
            console.error("Failed to fetch messages", e);
        }
    };

    const handleSend = async (content: string, attachments: any[] = []) => {
        // Add user message immediately
        const userMsg: Message = { role: 'user', content, attachments };
        setMessages((prev) => [...prev, userMsg]);
        setIsLoading(true);

        try {
            // Append attachment URLs to content if any (temporary solution for orchestrator context)
            let finalContent = content;
            if (attachments.length > 0) {
                finalContent += "\n\n[Attachments]:\n" + attachments.map(a => `${a.filename}: ${a.url}`).join("\n");
            }

            const response = await fetch('http://localhost:8001/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: finalContent,
                    session_id: sessionId
                }),
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();

            // If new session started, update ID
            if (data.session_id && data.session_id !== sessionId) {
                setSessionId(data.session_id);
            }

            const aiMsg: Message = {
                role: 'assistant',
                content: data.response
            };
            setMessages((prev) => [...prev, aiMsg]);

        } catch (error) {
            console.error("API Call Failed:", error);
            const errorMsg: Message = {
                role: 'assistant',
                content: "I'm sorry, I'm having trouble connecting to the Orchestrator. Please ensure the backend is running on port 8001."
            };
            setMessages((prev) => [...prev, errorMsg]);
        } finally {
            setIsLoading(false);
        }
    };

    const isEmptyState = messages.length === 0;

    return (
        <div className="flex-1 flex flex-col h-full relative bg-[#212121]">

            {/* Mobile Header / Desktop Toggle */}
            <div className="flex items-center justify-between p-2 sticky top-0 z-10">
                <div className="flex items-center">
                    {/* Sidebar Toggle (Only visible if sidebar is closed on desktop, or always on mobile) */}
                    <button
                        onClick={toggleSidebar}
                        className={clsx(
                            "p-3 rounded-lg hover:bg-[#2f2f2f] text-gray-400 hover:text-white transition-colors md:hidden",
                            isSidebarOpen ? "hidden" : "block"
                        )}
                    >
                        {isSidebarOpen ? <Menu className="w-6 h-6" /> : <PanelLeftOpen className="w-5 h-5" />}
                    </button>

                    <button className="flex items-center text-[#ececec] font-medium text-lg px-2 py-2 rounded-lg hover:bg-[#2f2f2f] ml-1">
                        MediCortex AI
                        <ChevronDown className="ml-1 w-4 h-4 text-gray-500" />
                    </button>
                </div>

                <button className="p-3 rounded-lg hover:bg-[#2f2f2f] text-gray-400 hover:text-white transition-colors">
                    <SquarePen className="w-5 h-5" />
                </button>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-y-auto w-full scrollbar-thin">

                {/* Empty State */}
                {isEmptyState ? (
                    <div className="flex flex-col items-center justify-center h-full px-4 pb-48">

                        <h2 className="text-2xl font-medium text-white mb-8">What can I help with?</h2>
                    </div>
                ) : (
                    <div className="flex flex-col pb-40 text-gray-100 w-full h-full overflow-y-auto scrollbar-thin">
                        <div className="w-full max-w-4xl mx-auto px-4 md:px-0 pt-4">
                            {messages.map((msg, idx) => (
                                <MessageBubble key={idx} role={msg.role} content={msg.content} />
                            ))}
                            <div ref={messagesEndRef} />
                        </div>
                    </div>
                )}

            </div>

            <InputArea onSend={handleSend} isLoading={isLoading} isEmptyState={isEmptyState} />
        </div>
    );
};

export default ChatArea;
