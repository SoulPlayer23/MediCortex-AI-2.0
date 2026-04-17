export interface Message {
    role: 'user' | 'assistant';
    content: string;
    attachments?: any[];
    thinking?: string[];
    metadata?: any;
    id?: number;
}

export interface SessionState {
    messages: Message[];
    isLoading: boolean;
}
