import { useState, useRef, useEffect } from 'react';
import { Sparkles, Paperclip, Mic, ArrowUp, X, File } from 'lucide-react';
import clsx from 'clsx';

interface InputAreaProps {
    onSend: (message: string, attachments?: any[]) => void;
    isLoading: boolean;
    isEmptyState: boolean;
}

const InputArea = ({ onSend, isLoading, isEmptyState }: InputAreaProps) => {
    const [input, setInput] = useState('');
    const [attachments, setAttachments] = useState<any[]>([]);
    const [isUploading, setIsUploading] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [input]);

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            const file = e.target.files[0];
            setIsUploading(true);

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('http://localhost:8001/upload', {
                    method: 'POST',
                    body: formData,
                });

                if (res.ok) {
                    const data = await res.json();
                    setAttachments(prev => [...prev, {
                        ...data,
                        type: file.type
                    }]);
                }
            } catch (error) {
                console.error("Upload failed", error);
            } finally {
                setIsUploading(false);
                if (fileInputRef.current) fileInputRef.current.value = '';
            }
        }
    };

    const removeAttachment = (index: number) => {
        setAttachments(prev => prev.filter((_, i) => i !== index));
    };

    const handleSend = () => {
        if ((input.trim() || attachments.length > 0) && !isLoading && !isUploading) {
            onSend(input, attachments);
            setInput('');
            setAttachments([]);
            if (textareaRef.current) {
                textareaRef.current.style.height = '56px';
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
            "w-full px-4 z-30 transition-all duration-300 ease-in-out",
            isEmptyState ? "absolute top-[55%] left-1/2 -translate-x-1/2 -translate-y-1/2 max-w-2xl" : "absolute bottom-6 left-0 w-full"
        )}>
            <div className={clsx("mx-auto", isEmptyState ? "w-full" : "max-w-3xl")}>
                <div className="relative flex flex-col w-full bg-zinc-800/70 backdrop-blur-xl rounded-[26px] shadow-2xl border border-white/10 ring-1 ring-black/5 overflow-hidden transition-all focus-within:ring-white/10 focus-within:bg-zinc-800/90">

                    {/* Attachments Preview */}
                    {attachments.length > 0 && (
                        <div className="flex px-4 pt-4 gap-2 overflow-x-auto scrollbar-hide">
                            {attachments.map((file, idx) => (
                                <div key={idx} className="flex items-center gap-2 bg-zinc-700/50 px-3 py-2 rounded-lg text-xs text-zinc-200 border border-white/5 shadow-sm">
                                    <File className="w-3.5 h-3.5 text-zinc-400" />
                                    <span className="truncate max-w-[150px]">{file.filename}</span>
                                    <button onClick={() => removeAttachment(idx)} className="text-zinc-400 hover:text-white ml-1">
                                        <X className="w-3 h-3" />
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="flex items-end gap-2 p-3">
                        {/* File Attach Button */}
                        <input
                            type="file"
                            ref={fileInputRef}
                            className="hidden"
                            onChange={handleFileSelect}
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploading}
                            className="p-2.5 text-zinc-400 hover:text-white hover:bg-zinc-700 rounded-full transition-colors disabled:opacity-50">
                            <Paperclip className="w-5 h-5" />
                        </button>

                        <textarea
                            ref={textareaRef}
                            rows={1}
                            placeholder="Message MediCortex AI..."
                            className="flex-1 w-full bg-transparent py-3 px-2 focus:ring-0 focus:outline-none max-h-[200px] overflow-y-auto scrollbar-hide text-white leading-relaxed text-[16px] placeholder:text-zinc-500 font-normal resize-none"
                            style={{ minHeight: '44px' }}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                        />

                        {/* Send / Mic Button */}
                        {(!input.trim() && attachments.length === 0) ? (
                            <button className="p-2.5 text-zinc-400 hover:text-white hover:bg-zinc-700 rounded-full transition-colors">
                                <Mic className="w-5 h-5" />
                            </button>
                        ) : (
                            <button
                                disabled={isLoading || isUploading}
                                onClick={handleSend}
                                className="p-2 rounded-full bg-white text-black hover:bg-zinc-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-white/10"
                            >
                                {isLoading || isUploading ? (
                                    <Sparkles className="w-5 h-5 animate-spin p-0.5" />
                                ) : (
                                    <ArrowUp className="w-5 h-5" />
                                )}
                            </button>
                        )}
                    </div>
                </div>
                <div className="text-center text-xs text-zinc-500 mt-3 font-medium opacity-60">
                    MediCortex AI can make mistakes. Check important info.
                </div>
            </div>
        </div>
    );
};

export default InputArea;
