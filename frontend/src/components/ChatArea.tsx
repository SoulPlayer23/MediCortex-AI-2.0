import { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import InputArea from './InputArea';
import { Menu, PanelLeftOpen, SquarePen, ChevronDown } from 'lucide-react';
import clsx from 'clsx';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

interface ChatAreaProps {
    isSidebarOpen: boolean;
    toggleSidebar: () => void;
}

const ChatArea = ({ isSidebarOpen, toggleSidebar }: ChatAreaProps) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (content: string) => {
        // Add user message
        const userMsg: Message = { role: 'user', content };
        setMessages((prev) => [...prev, userMsg]);
        setIsLoading(true);

        // Simulate AI delay
        setTimeout(() => {
            const aiMsg: Message = {
                role: 'assistant',
                content: `I received: "**${content}**".\n\nI can help you analyze this request.`
            };
            setMessages((prev) => [...prev, aiMsg]);
            setIsLoading(false);
        }, 1500);
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
