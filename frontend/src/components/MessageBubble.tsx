import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import clsx from 'clsx';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { ChevronDown, ChevronRight, BrainCircuit } from 'lucide-react';

interface MessageBubbleProps {
    role: 'user' | 'assistant';
    content: string;
    thinking?: string[];
    metadata?: any;
    isStreaming?: boolean;
}

const MessageBubble = ({ role, content, thinking, metadata, isStreaming }: MessageBubbleProps) => {
    const isUser = role === 'user';
    const [isThinkingOpen, setIsThinkingOpen] = useState(false);
    const [isMetadataOpen, setIsMetadataOpen] = useState(false);

    return (
        <div className={clsx("w-full py-6 md:py-8", isUser ? "bg-transparent" : "bg-transparent")}>
            <div className="w-full max-w-3xl mx-auto px-4 flex gap-4 md:gap-6">

                {/* Avatar */}
                <div className="flex-shrink-0 flex flex-col relative items-end">
                    {isUser ? (
                        <div className="w-8 h-8 rounded-full bg-zinc-700 flex items-center justify-center text-zinc-200 text-xs font-semibold">
                            VS
                        </div>
                    ) : (
                        <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center shadow-lg shadow-white/10">
                            <BrainCircuit className="w-5 h-5 text-black" />
                        </div>
                    )}
                </div>

                {/* Content */}
                <div className="relative flex-1 overflow-hidden">
                    {/* Name */}
                    <div className="font-semibold text-sm text-zinc-300 mb-1.5 opacity-90">
                        {isUser ? "You" : "MediCortex"}
                    </div>

                    {/* Thinking Process (Accordion) */}
                    {!isUser && thinking && (
                        <div className="mb-4">
                            <button
                                onClick={() => setIsThinkingOpen(!isThinkingOpen)}
                                className="flex items-center gap-2 text-xs font-medium text-zinc-400 hover:text-zinc-200 transition-colors bg-zinc-800/50 px-3 py-1.5 rounded-lg w-fit mb-2 border border-zinc-800"
                            >
                                {isThinkingOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                                <span>Thinking Process</span>
                                {/* Active Indicator if thinking is empty or just started */}
                                {thinking.length === 0 && (
                                    <span className="flex relative h-2 w-2 ml-1">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-zinc-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-zinc-500"></span>
                                    </span>
                                )}
                            </button>

                            {isThinkingOpen && (
                                <div className="pl-4 border-l-2 border-zinc-700 space-y-3 my-3 animate-in slide-in-from-top-2 duration-200">
                                    {thinking.length > 0 ? (
                                        thinking.map((step, idx) => (
                                            <div key={idx} className="bg-zinc-800/40 rounded-md p-2.5 text-xs text-zinc-400 font-mono border-l-2 border-zinc-700 hover:bg-zinc-800/60 transition-colors">
                                                <span className="text-zinc-500 mr-2">[{idx + 1}]</span>
                                                <ReactMarkdown components={{ p: ({ children }) => <span className="inline">{children}</span> }}>{step}</ReactMarkdown>
                                            </div>
                                        ))
                                    ) : (
                                        !content && (
                                            <div className="text-xs text-zinc-500 italic flex items-center gap-2">
                                                <BrainCircuit className="w-3 h-3 animate-pulse" />
                                                Initializing agents...
                                            </div>
                                        )
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Verification & Metadata (Accordion) */}
                    {!isUser && metadata && (
                        <div className="mb-4">
                            <button
                                onClick={() => setIsMetadataOpen(!isMetadataOpen)}
                                className="flex items-center gap-2 text-xs font-medium text-emerald-400/80 hover:text-emerald-300 transition-colors bg-emerald-900/20 px-3 py-1.5 rounded-lg w-fit mb-2 border border-emerald-900/30"
                            >
                                {isMetadataOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                                <span>Verification & Metadata</span>
                            </button>

                            {isMetadataOpen && (
                                <div className="pl-4 border-l-2 border-emerald-800/50 space-y-3 my-3 animate-in slide-in-from-top-2 duration-200">
                                    <div className="bg-zinc-900/60 rounded-md p-3 text-xs text-zinc-300 font-mono border border-zinc-800/50">
                                        <div className="grid grid-cols-2 gap-2">
                                            <div className="text-zinc-500">LLM Engine:</div>
                                            <div className="text-emerald-400">{metadata.llm_used || 'Unknown'}</div>

                                            <div className="text-zinc-500">Judge Score:</div>
                                            <div className="text-zinc-300">{metadata.judge_score ? `${metadata.judge_score}/5` : 'N/A'}</div>

                                            {metadata.judge_confidence && (
                                                <>
                                                    <div className="text-zinc-500">Confidence:</div>
                                                    <div className="text-zinc-300">{metadata.judge_confidence}</div>
                                                </>
                                            )}
                                        </div>
                                        {metadata.judge_reason && (
                                            <div className="mt-2 pt-2 border-t border-zinc-800/50 text-zinc-400 italic">
                                                "{metadata.judge_reason}"
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Streaming pulse — shown after thinking steps arrive but before first token */}
                    {isStreaming && !content && thinking && thinking.length > 0 && (
                        <div className="flex items-center gap-2 mb-4 text-xs text-zinc-500 animate-in fade-in duration-300">
                            <span className="flex gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:0ms]" />
                                <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:150ms]" />
                                <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:300ms]" />
                            </span>
                            Generating response...
                        </div>
                    )}

                    {/* Message Text */}
                    <div className="prose prose-invert prose-p:leading-relaxed prose-pre:p-0 prose-pre:bg-zinc-900/50 prose-pre:rounded-xl max-w-none text-zinc-100 text-[15px] md:text-base font-normal tracking-wide">
                        <ReactMarkdown
                            components={{
                                code({ node, inline, className, children, ...props }: any) {
                                    const match = /language-(\w+)/.exec(className || '')
                                    return !inline && match ? (
                                        <div className="rounded-xl overflow-hidden my-4 border border-zinc-700/50 shadow-md">
                                            <div className="bg-zinc-900 px-4 py-2 text-xs font-sans flex justify-between items-center text-zinc-400 border-b border-zinc-800">
                                                <span className="font-mono">{match[1]}</span>
                                                <span className="cursor-pointer hover:text-white transition-colors">Copy code</span>
                                            </div>
                                            <SyntaxHighlighter
                                                {...props}
                                                style={vscDarkPlus}
                                                language={match[1]}
                                                PreTag="div"
                                                customStyle={{ margin: 0, padding: '16px' }}
                                            >
                                                {String(children).replace(/\n$/, '')}
                                            </SyntaxHighlighter>
                                        </div>
                                    ) : (
                                        <code {...props} className={clsx(className, "bg-zinc-800/80 px-1.5 py-0.5 rounded text-sm text-zinc-200 font-mono")}>
                                            {children}
                                        </code>
                                    )
                                },
                                p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
                                ul: ({ children }) => <ul className="list-disc pl-5 mb-4 space-y-1">{children}</ul>,
                                ol: ({ children }) => <ol className="list-decimal pl-5 mb-4 space-y-1">{children}</ol>,
                                li: ({ children }) => <li className="mb-1">{children}</li>,
                                h1: ({ children }) => <h1 className="text-2xl font-bold mb-4 mt-6 text-white">{children}</h1>,
                                h2: ({ children }) => <h2 className="text-xl font-bold mb-3 mt-5 text-white">{children}</h2>,
                                h3: ({ children }) => <h3 className="text-lg font-bold mb-2 mt-4 text-white">{children}</h3>,
                                blockquote: ({ children }) => <blockquote className="border-l-4 border-zinc-600 pl-4 italic text-zinc-400 my-4">{children}</blockquote>
                            }}
                        >
                            {content}
                        </ReactMarkdown>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MessageBubble;
