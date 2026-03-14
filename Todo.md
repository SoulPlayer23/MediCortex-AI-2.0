# MediCortex AI 2.0 — Technical Todo

---

## Open Issues

### 🔴 High

_None_

---

### 🟡 Medium

#### AGG-1 — Aggregator emits duplicate sections
**Component:** `orchestrator.py` → `node_aggregator` (GPT-4o-mini system prompt)
**Observed:** "Consider Specialist Referral" appeared twice in the diabetes response with slightly different wording (endocrinologist vs. primary care physician). The judge (`node_reviewer`) flagged it inline: *"The referral suggestions were repeated multiple times; please ensure to streamline this in practice."*
**Root cause:** The aggregator is not merging near-identical recommendations from multiple agents before emitting the final Markdown.
**Fix:** Tighten the aggregator system prompt to explicitly deduplicate sections and consolidate near-identical recommendations into a single entry.

---

#### AGG-2 — Aggregator does not deduplicate redundant source snippets
**Component:** `orchestrator.py` → `node_aggregator` (GPT-4o-mini system prompt)
**Observed:** In the metformin + ibuprofen drug response, Drugs.com Sources 3–7 all contained the same sentence ("ibuprofen is one of 394 medications known to interact with metformin") with no new information per entry.
**Root cause:** The aggregator passes through all source snippets from agent outputs without collapsing those that carry identical information.
**Fix:** Update the aggregator system prompt to deduplicate source snippets — keep only the first occurrence of a unique fact and drop subsequent entries that add no new information.

---

### 🔵 Low / UX

#### UI-4 — Page refresh on a chat URL loads blank empty state
**Component:** `frontend/src/App.tsx`
**Observed:** Refreshing the browser while viewing `/chat/{session_id}` renders a blank new-chat screen even though the URL still points to that session. The chat history is lost until the user navigates via the sidebar.
**Root cause:** `App.tsx` initialises `currentSessionId` as `null` and never reads `window.location.pathname`. The URL written by `window.history.pushState` is ignored on hard refresh, so `ChatArea` never calls `fetchMessages`.
**Fix:** Seed `currentSessionId` from the URL on first render:
```ts
const [currentSessionId, setCurrentSessionId] = useState<string | null>(() => {
  const match = window.location.pathname.match(/\/chat\/(.+)/);
  return match ? match[1] : null;
});
```

---

#### UI-5 — Last messages scroll under the input bar and disclaimer
**Component:** `frontend/src/components/InputArea.tsx`, `ChatArea.tsx`
**Observed:** When scrolling to the bottom, the last message(s) are hidden behind the absolutely-positioned input box and the "MediCortex AI can make mistakes" disclaimer text.
**Root cause:** `InputArea` uses `absolute bottom-6` so it floats on top of the `overflow-y-auto` scroll container. The messages list uses a fixed `pb-48` (192px) to create clearance, but this doesn't account for a taller textarea (up to 200px), attachment previews, or the disclaimer (~32px). `messagesEndRef` is only `h-4` (16px), so the scroll anchor lands too close to the last message.
**Fix:** Replace fixed `pb-48` with a larger clearance (e.g. `pb-56`) and increase `messagesEndRef` height to match the maximum possible InputArea height, or restructure InputArea to be `relative` (in-flow) and remove the absolute positioning.

---

#### UI-3 — Duplicate message bubbles on backend connection failure
**Component:** `frontend/src/components/ChatArea.tsx` → `handleSend`
**Observed:** When the backend is unreachable, two assistant bubbles appear: the initial streaming placeholder (with "Thinking Process" pulsing dot) and a second error message bubble below it.
**Root cause:** On fetch failure, the `catch` block appends a new error `Message` object instead of replacing the existing placeholder (`aiMsgId`).
**Fix:** In the `catch` block, replace the placeholder by mapping over messages and updating the entry matching `aiMsgId` with the error content, rather than pushing a new message.

---


---

## Resolved

#### UI-1 — No streaming progress indicator during long responses
Added bouncing dots + "Generating response..." indicator in `MessageBubble.tsx`, shown when `isStreaming && !content && thinking.length > 0`. Verified working in browser during ~4 min MedGemma inference.

#### UI-2 — Chat does not auto-scroll to latest message
Implemented smart scroll in `ChatArea.tsx` using `isNearBottomRef`. Auto-scrolls only when within 100px of bottom; shows "↓ Scroll to bottom" button otherwise. Verified working in browser.
