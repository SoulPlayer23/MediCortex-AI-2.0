import { useState, useRef, useEffect } from 'react';
import { Sparkles, Paperclip, Mic, ArrowUp } from 'lucide-react';
import clsx from 'clsx';

interface InputAreaProps {
    onSend: (message: string) => void;
    isLoading: boolean;
    isEmptyState: boolean;
}

const InputArea = ({ onSend, isLoading, isEmptyState }: InputAreaProps) => {
    const [input, setInput] = useState('');
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [input]);

    const handleSend = () => {
        if (input.trim() && !isLoading) {
            onSend(input);
            setInput('');
            if (textareaRef.current) {
                textareaRef.current.style.height = '24px';
            }
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className={clsx(
            "w-full px-4 z-20",
            isEmptyState ? "absolute top-1/2 left-1/2 -translate-x-1/2 mt-8 max-w-2xl" : "absolute bottom-0 left-0 w-full bg-[#212121] pt-2 pb-6"
        )}>
            <div className={clsx("mx-auto", isEmptyState ? "w-full" : "max-w-3xl")}>
                <div className="relative flex flex-col w-full bg-[#2f2f2f] rounded-[26px] shadow-sm focus-within:ring-1 focus-within:ring-gray-500 overflow-hidden">

                    {/* File Attach Button */}
                    <button className="absolute left-3 bottom-3 p-2 text-gray-400 hover:text-white hover:bg-[#424242] rounded-full transition-colors">
                        <Paperclip className="w-5 h-5" />
                    </button>

                    <textarea
                        ref={textareaRef}
                        rows={1}
                        placeholder="Message MediCortex AI..."
                        className="m-0 w-full resize-none border-0 bg-transparent py-[14px] pr-12 pl-12 focus:ring-0 focus-visible:ring-0 max-h-[200px] overflow-y-auto scrollbar-hide outline-none text-white leading-6 text-base"
                        style={{ height: '52px', minHeight: '52px' }}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />

                    <div className="absolute right-3 bottom-2 flex items-center gap-2">
                        {!input.trim() && (
                            <button className="p-2 text-gray-400 hover:text-white hover:bg-[#424242] rounded-full transition-colors">
                                <Mic className="w-5 h-5" />
                            </button>
                        )}
                        <button
                            disabled={!input.trim() || isLoading}
                            onClick={handleSend}
                            className={clsx(
                                "p-2 rounded-full transition-all duration-200",
                                input.trim() && !isLoading ? "bg-white text-black hover:bg-gray-200" : "bg-[#676767] text-[#2f2f2f] cursor-not-allowed opacity-50"
                            )}
                        >
                            {isLoading ? (
                                <Sparkles className="w-4 h-4 animate-spin" />
                            ) : (
                                <ArrowUp className="w-5 h-5" />
                            )}
                        </button>
                    </div>
                </div>
                <div className="text-center text-xs text-[#b4b4b4] mt-2 pb-2">
                    MediCortex AI can make mistakes. Check important info.
                </div>
            </div>
        </div>
    );
};

export default InputArea;
