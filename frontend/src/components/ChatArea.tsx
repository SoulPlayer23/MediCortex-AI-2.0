import { useState, useRef, useEffect, useCallback } from 'react';
import MessageBubble from './MessageBubble';
import InputArea from './InputArea';
import { Menu, PanelLeftOpen, BrainCircuit, ArrowDown } from 'lucide-react';
import clsx from 'clsx';
import type { Message, SessionState } from '../types';

// Sentinel cache key used for a brand-new chat before the backend assigns a session ID.
const PENDING_SESSION = '__pending__';

// DEPLOY-4: backend base URL is environment-driven so the same SPA build can
// run locally and on GitHub Pages. Falls back to localhost for `npm run dev`
// when no .env is loaded.
const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8001';

interface ChatAreaProps {
    isSidebarOpen: boolean;
    toggleSidebar: () => void;
    sessionId: string | null;
    setSessionId: (id: string) => void;
    sessionCache: React.MutableRefObject<Map<string, SessionState>>;
    bumpIfActive: (sessionId: string | null) => void;
    // activeTick is intentionally not read as a value — it exists solely so React
    // re-renders ChatArea whenever the active session's cache entry is mutated.
    activeTick: number;
}

const ChatArea = ({
    isSidebarOpen,
    toggleSidebar,
    sessionId,
    setSessionId,
    sessionCache,
    bumpIfActive,
    activeTick: _activeTick,
}: ChatAreaProps) => {
    // Derive messages and isLoading from the shared session cache instead of
    // local state, so active streams survive session switches.
    const cacheKey = sessionId ?? PENDING_SESSION;
    const { messages, isLoading } = sessionCache.current.get(cacheKey) ?? { messages: [], isLoading: false };

    const [showScrollButton, setShowScrollButton] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const skipNextFetch = useRef(false);
    const isNearBottomRef = useRef(true);

    const SCROLL_THRESHOLD = 100;

    // Helper: mutate a session's cache entry then notify the render system.
    const updateSession = useCallback((sid: string, updater: (prev: SessionState) => SessionState) => {
        const prev = sessionCache.current.get(sid) ?? { messages: [], isLoading: false };
        sessionCache.current.set(sid, updater(prev));
        bumpIfActive(sid);
    }, [sessionCache, bumpIfActive]);

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    // Track whether the user is near the bottom.
    useEffect(() => {
        const container = scrollContainerRef.current;
        if (!container) return;

        const handleScroll = () => {
            const { scrollTop, scrollHeight, clientHeight } = container;
            const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
            isNearBottomRef.current = distanceFromBottom <= SCROLL_THRESHOLD;
            setShowScrollButton(!isNearBottomRef.current && messages.length > 0);
        };

        container.addEventListener('scroll', handleScroll, { passive: true });
        return () => container.removeEventListener('scroll', handleScroll);
    }, [messages.length]);

    // Auto-scroll only when user is already near the bottom.
    useEffect(() => {
        if (isNearBottomRef.current) {
            scrollToBottom();
        }
    }, [messages, isLoading, scrollToBottom]);

    // Fetch message history when switching to a session — but skip if that session
    // already has an active stream running (isLoading === true), so we never
    // overwrite in-progress streaming state.
    useEffect(() => {
        if (skipNextFetch.current) {
            skipNextFetch.current = false;
            return;
        }

        if (!sessionId) {
            // New chat: clear any leftover PENDING state and show empty screen.
            sessionCache.current.delete(PENDING_SESSION);
            bumpIfActive(null);
            return;
        }

        const existing = sessionCache.current.get(sessionId);
        if (existing?.isLoading) {
            // This session has a live stream — switching back to it is instant,
            // no DB fetch needed. The stream is already writing into the cache.
            return;
        }

        fetchMessages(sessionId);
    }, [sessionId]);

    const fetchMessages = async (id: string) => {
        try {
            const res = await fetch(`${API_BASE}/chats/${id}`);
            if (res.ok) {
                const data = await res.json();
                // API serializes the Pydantic alias, so the field arrives as
                // `message_metadata`. Remap it to `metadata` so MessageBubble
                // can access it consistently (streaming path sets `metadata` directly).
                const messages = data.map((msg: any) => ({
                    ...msg,
                    metadata: msg.message_metadata,
                    attachments: msg.attachments ?? [],
                }));
                updateSession(id, () => ({ messages, isLoading: false }));
            }
        } catch (e) {
            console.error("Failed to fetch messages", e);
        }
    };

    const handleSend = async (content: string, attachments: any[] = []) => {
        isNearBottomRef.current = true;
        setShowScrollButton(false);

        // Capture the session this stream belongs to. For a new chat sessionId is
        // null, so we park state under PENDING_SESSION until the backend assigns an ID.
        let streamSessionId = sessionId ?? PENDING_SESSION;

        const userMsg: Message = { role: 'user', content, attachments };
        const aiMsgId = Date.now();
        const initialAiMsg: Message = { role: 'assistant', content: '', thinking: [], id: aiMsgId };

        updateSession(streamSessionId, prev => ({
            messages: [...prev.messages, userMsg, initialAiMsg],
            isLoading: true,
        }));

        try {
            const response = await fetch(`${API_BASE}/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: content,
                    session_id: sessionId,
                    attachments: attachments.length > 0 ? attachments : undefined,
                }),
            });

            if (!response.ok) throw new Error(`Error: ${response.statusText}`);
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
                    if (!line.startsWith('data: ')) continue;
                    const dataStr = line.replace('data: ', '').trim();

                    if (dataStr === '[DONE]') {
                        updateSession(streamSessionId, prev => ({ ...prev, isLoading: false }));
                        break;
                    }

                    try {
                        const data = JSON.parse(dataStr);

                        if (data.type === 'session_id') {
                            const newSessionId = data.content;
                            if (newSessionId !== sessionId) {
                                // Migrate accumulated state from PENDING_SESSION to the real ID.
                                const current = sessionCache.current.get(streamSessionId);
                                if (current) {
                                    sessionCache.current.set(newSessionId, current);
                                    sessionCache.current.delete(streamSessionId);
                                }
                                streamSessionId = newSessionId;
                                skipNextFetch.current = true;
                                setSessionId(newSessionId);
                                window.history.pushState({}, '', `/chat/${newSessionId}`);
                            }
                        } else if (data.type === 'thought') {
                            updateSession(streamSessionId, prev => ({
                                ...prev,
                                messages: prev.messages.map(msg =>
                                    msg.id === aiMsgId
                                        ? { ...msg, thinking: [...(msg.thinking ?? []), data.content] }
                                        : msg
                                ),
                            }));
                        } else if (data.type === 'metadata') {
                            updateSession(streamSessionId, prev => ({
                                ...prev,
                                messages: prev.messages.map(msg =>
                                    msg.id === aiMsgId ? { ...msg, metadata: data.content } : msg
                                ),
                            }));
                        } else if (data.type === 'token') {
                            updateSession(streamSessionId, prev => ({
                                ...prev,
                                messages: prev.messages.map(msg =>
                                    msg.id === aiMsgId
                                        ? { ...msg, content: msg.content + data.content }
                                        : msg
                                ),
                            }));
                        } else if (data.type === 'response') {
                            updateSession(streamSessionId, prev => ({
                                ...prev,
                                messages: prev.messages.map(msg =>
                                    msg.id === aiMsgId ? { ...msg, content: data.content } : msg
                                ),
                            }));
                        } else if (data.type === 'error') {
                            console.error("Stream error:", data.content);
                        }
                    } catch (e) {
                        console.error("Failed to parse SSE line", line, e);
                    }
                }
            }

        } catch (error) {
            console.error("API Call Failed:", error);
            updateSession(streamSessionId, prev => ({
                messages: prev.messages.map(msg =>
                    msg.id === aiMsgId
                        ? { ...msg, content: "I'm sorry, I'm having trouble connecting to the Orchestrator. Please ensure the backend is running on port 8001." }
                        : msg
                ),
                isLoading: false,
            }));
        }
    };

    const isEmptyState = messages.length === 0;
    const streamingMsgId = isLoading
        ? [...messages].reverse().find(m => m.role === 'assistant')?.id
        : undefined;

    return (
        <div className="flex-1 flex flex-col h-full relative bg-zinc-900">

            {/* Mobile Header / Desktop Toggle */}
            <div className="sticky top-0 z-20 flex items-center justify-between p-2">
                <div className="flex items-center">
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

                {isEmptyState ? (
                    <div className="flex flex-col items-center justify-center min-h-full gap-8 animate-in fade-in zoom-in-95 duration-500">
                        <div className="flex flex-col items-center">
                            <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-6 shadow-[0_0_40px_-5px_rgba(255,255,255,0.3)]">
                                <BrainCircuit className="w-8 h-8 text-black" />
                            </div>
                            <h2 className="text-2xl font-semibold text-white mb-2">How can I help you today?</h2>
                            <p className="text-zinc-400 max-w-md text-center">
                                I'm an advanced medical reasoning agent. I can help analyze reports, diagnose symptoms, and check drug interactions.
                            </p>
                        </div>
                        <InputArea onSend={handleSend} isLoading={isLoading} isEmptyState={isEmptyState} />
                    </div>
                ) : (
                    <div className="flex flex-col pb-4 w-full">
                        {messages.map((msg, idx) => (
                            <MessageBubble
                                key={idx}
                                role={msg.role}
                                content={msg.content}
                                attachments={msg.attachments}
                                thinking={msg.thinking}
                                metadata={msg.metadata}
                                isStreaming={msg.id !== undefined && msg.id === streamingMsgId}
                            />
                        ))}
                        <div ref={messagesEndRef} className="h-4" />
                    </div>
                )}
            </div>

            {showScrollButton && (
                <button
                    onClick={scrollToBottom}
                    className="absolute bottom-28 right-6 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-zinc-700 hover:bg-zinc-600 text-zinc-200 text-sm shadow-lg transition-all animate-in fade-in slide-in-from-bottom-2 duration-200"
                >
                    <ArrowDown className="w-3.5 h-3.5" />
                    Scroll to bottom
                </button>
            )}

            {!isEmptyState && <InputArea onSend={handleSend} isLoading={isLoading} isEmptyState={isEmptyState} />}
        </div>
    );
};

export default ChatArea;
