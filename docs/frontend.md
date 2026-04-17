# Frontend — MediCortex AI 2.0

React 19 + Vite + Tailwind CSS SPA in `frontend/`. Talks to the orchestrator at `http://localhost:8001`.

## Key Components

| File | Purpose |
|---|---|
| `frontend/src/App.tsx` | Root layout; owns `isSidebarOpen` and `currentSessionId` state |
| `frontend/src/components/Sidebar.tsx` | Fetches `/chats` on mount and on `currentSessionId`/`isOpen` change. Clicking a session calls `onSelectChat(session.id)` |
| `frontend/src/components/ChatArea.tsx` | Fetches `/chats/{id}` on `sessionId` change; streams `/chat/stream` SSE for new messages. Smart scroll: auto-scrolls only when within 100 px of bottom (`isNearBottomRef`); shows `ArrowDown` button otherwise |
| `frontend/src/components/MessageBubble.tsx` | Renders user/assistant messages with Markdown + syntax highlighting. Collapsible **Thinking Process** accordion for ReAct steps. Bouncing-dots indicator while `isStreaming && content === '' && thinking.length > 0` |
| `frontend/src/components/InputArea.tsx` | Text input with attachment and microphone icons |
| `frontend/src/types.ts` | Shared TypeScript types |
| `frontend/src/index.css` | Global styles |

## Dev Notes

- Dev server: `http://localhost:5173`
- Backend URL hardcoded to `localhost:8001` — update in `ChatArea.tsx` if orchestrator moves
- **Vite HMR limitation**: File changes via Claude Code Edit/Write tools on the Windows `D:` drive do **not** trigger chokidar. After any frontend file change, restart dev server manually (`Ctrl-C` then `npm run dev`)
