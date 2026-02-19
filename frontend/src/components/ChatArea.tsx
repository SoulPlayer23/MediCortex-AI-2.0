import { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import InputArea from './InputArea';
import { Menu, PanelLeftOpen, BrainCircuit } from 'lucide-react';
import clsx from 'clsx';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    attachments?: any[];
    thinking?: string[]; // New: Thinking steps from the agent
    id?: number; // New: ID for streaming updates
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
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const skipNextFetch = useRef(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    useEffect(() => {
        if (skipNextFetch.current) {
            skipNextFetch.current = false;
            return;
        }

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

        // Create placeholder for AI message
        const aiMsgId = Date.now();
        const initialAiMsg: Message = {
            role: 'assistant',
            content: '',
            thinking: [],
            id: aiMsgId
        };
        setMessages((prev) => [...prev, initialAiMsg]);

        try {
            // Append attachment URLs to content if any
            let finalContent = content;
            if (attachments.length > 0) {
                finalContent += "\n\n[Attachments]:\n" + attachments.map(a => `${a.filename}: ${a.url}`).join("\n");
            }

            const response = await fetch('http://localhost:8001/chat/stream', {
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

            if (!response.body) throw new Error("No response body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.replace('data: ', '').trim();
                        if (dataStr === '[DONE]') {
                            setIsLoading(false);
                            break;
                        }

                        try {
                            const data = JSON.parse(dataStr);

                            // Handle Events
                            if (data.type === 'session_id') {
                                const newSessionId = data.content;
                                if (newSessionId !== sessionId) {
                                    skipNextFetch.current = true; // Prevent clearing messages!
                                    setSessionId(newSessionId);
                                    // Update URL without reloading to reflect new session
                                    window.history.pushState({}, '', `/chat/${newSessionId}`);
                                }
                            } else if (data.type === 'thought') {
                                setMessages(prev => prev.map(msg => {
                                    if (msg.id === aiMsgId) {
                                        const newThinking = msg.thinking ? [...msg.thinking, data.content] : [data.content];
                                        return { ...msg, thinking: newThinking };
                                    }
                                    return msg;
                                }));
                            } else if (data.type === 'token') {
                                setMessages(prev => prev.map(msg =>
                                    msg.id === aiMsgId ? { ...msg, content: msg.content + data.content } : msg
                                ));
                            } else if (data.type === 'response') {
                                // Fallback or final full replacement if needed (usually token stream covers it)
                                setMessages(prev => prev.map(msg =>
                                    msg.id === aiMsgId ? { ...msg, content: data.content } : msg
                                ));
                            } else if (data.type === 'error') {
                                console.error("Stream error:", data.content);
                            }
                        } catch (e) {
                            console.error("Failed to parse SSE line", line, e);
                        }
                    }
                }
            }

        } catch (error) {
            console.error("API Call Failed:", error);
            const errorMsg: Message = {
                role: 'assistant',
                content: "I'm sorry, I'm having trouble connecting to the Orchestrator. Please ensure the backend is running on port 8001."
            };
            setMessages((prev) => [...prev, errorMsg]);
            setIsLoading(false);
        }
    };

    const isEmptyState = messages.length === 0;

    return (
        <div className="flex-1 flex flex-col h-full relative bg-zinc-900">

            {/* Mobile Header / Desktop Toggle */}
            <div className="sticky top-0 z-20 flex items-center justify-between p-2">
                <div className="flex items-center">
                    {/* Sidebar Toggle */}
                    <button
                        onClick={toggleSidebar}
                        className={clsx(
                            "p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors md:hidden",
                            isSidebarOpen ? "hidden" : "block"
                        )}
                    >
                        {isSidebarOpen ? <Menu className="w-5 h-5" /> : <PanelLeftOpen className="w-5 h-5" />}
                    </button>

                    <div className="lex items-center gap-2 px-3 py-2 rounded-lg text-zinc-400 hover:text-white transition-colors cursor-pointer select-none">
                        <span className="font-semibold text-lg tracking-tight text-white">MediCortex AI</span>
                        <span className="text-xs bg-zinc-800 text-zinc-400 px-1.5 py-0.5 rounded ml-2">2.0</span>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div ref={scrollContainerRef} className="flex-1 overflow-y-auto w-full scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">

                {/* Empty State */}
                {isEmptyState ? (
                    <div className="flex flex-col items-center justify-center h-[55%] px-4 animate-in fade-in zoom-in-95 duration-500">
                        <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-6 shadow-[0_0_40px_-5px_rgba(255,255,255,0.3)]">
                            <BrainCircuit className="w-8 h-8 text-black" />
                        </div>
                        <h2 className="text-2xl font-semibold text-white mb-2">How can I help you today?</h2>
                        <p className="text-zinc-400 max-w-md text-center">
                            I'm an advanced medical reasoning agent. I can help analyze reports, diagnose symptoms, and check drug interactions.
                        </p>
                    </div>
                ) : (
                    <div className="flex flex-col pb-48 w-full">
                        {messages.map((msg, idx) => (
                            <MessageBubble
                                key={idx}
                                role={msg.role}
                                content={msg.content}
                                thinking={msg.thinking}
                            />
                        ))}
                        <div ref={messagesEndRef} className="h-4" />
                    </div>
                )}
            </div>

            <InputArea onSend={handleSend} isLoading={isLoading} isEmptyState={isEmptyState} />
        </div>
    );
};

export default ChatArea;
