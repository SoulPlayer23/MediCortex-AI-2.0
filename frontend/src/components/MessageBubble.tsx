import ReactMarkdown from 'react-markdown';
import clsx from 'clsx';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface MessageBubbleProps {
    role: 'user' | 'assistant';
    content: string;
}

const MessageBubble = ({ role, content }: MessageBubbleProps) => {
    const isUser = role === 'user';

    return (
        <div className={clsx("w-full text-gray-100", isUser ? "dark:bg-transparent" : "dark:bg-transparent")}>
            <div className={clsx(
                "p-4 flex m-auto w-full",
                isUser ? "justify-end" : "justify-start"
            )}>
                <div className={clsx(
                    "relative px-5 py-3.5 text-[15px] md:text-base",
                    isUser ? "bg-[#2f2f2f] rounded-[26px] max-w-[85%] sm:max-w-[75%]" : "max-w-full"
                )}>

                    {/* Message Content */}
                    <div className={clsx(
                        "prose prose-invert prose-p:leading-relaxed prose-pre:p-0 prose-pre:bg-[#2f2f2f] prose-pre:rounded-lg max-w-none",
                        isUser ? "text-[#ececec]" : "text-[#d1d5db]"
                    )}>
                        <ReactMarkdown
                            components={{
                                code({ node, inline, className, children, ...props }: any) {
                                    const match = /language-(\w+)/.exec(className || '')
                                    return !inline && match ? (
                                        <div className="rounded-lg overflow-hidden my-3 border border-gray-700">
                                            <div className="bg-[#2f2f2f] px-3 py-1.5 text-xs font-sans flex justify-between items-center text-gray-200">
                                                <span>{match[1]}</span>
                                                <span className="text-xs cursor-pointer hover:text-white flex items-center gap-1">
                                                    Copy code
                                                </span>
                                            </div>
                                            <SyntaxHighlighter
                                                {...props}
                                                style={vscDarkPlus}
                                                language={match[1]}
                                                PreTag="div"
                                                customStyle={{ margin: 0, borderRadius: 0 }}
                                            >
                                                {String(children).replace(/\n$/, '')}
                                            </SyntaxHighlighter>
                                        </div>
                                    ) : (
                                        <code {...props} className={className}>
                                            {children}
                                        </code>
                                    )
                                }
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
