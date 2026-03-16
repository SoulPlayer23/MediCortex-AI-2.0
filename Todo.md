# MediCortex AI 2.0 — Technical Todo

---

## Open Issues

### 🔴 High

_None_

---

### 🟡 Medium

_None_

---

### 🔵 Low / UX

#### UI-5 — Last messages scroll under the input bar and disclaimer
**Component:** `frontend/src/components/InputArea.tsx`, `ChatArea.tsx`
**Observed:** When scrolling through a long response, message content is visible behind the absolutely-positioned input box and the "MediCortex AI can make mistakes" disclaimer text.
**Root cause:** `InputArea` uses `absolute bottom-6` so it floats on top of the `overflow-y-auto` scroll container. The scroll container extends all the way to the viewport bottom, so content at any scroll position can appear behind the input bar. Padding hacks (`pb-48`, `pb-80`) only help at the very end of the content, not during mid-scroll.
**Fix:** Either make `InputArea` in-flow (remove `absolute`) in chat mode so the scroll container naturally stops above it, or add a same-height in-flow spacer div in `ChatArea.tsx` to reserve that space. Note: Vite HMR does not detect file changes made from WSL/git-bash on this machine — requires a manual frontend restart (`cd frontend && npm run dev`) after applying the fix.

#### AGG-2 — Sources not surfaced as a distinct UI element
**Observed:** Sources cited by agents appear inline as raw Markdown hyperlinks scattered through the response text (e.g. `[Mayo Clinic](https://...)`). Same source URL can appear multiple times across different claims.
**Goal:** Collect all unique source URLs from the aggregated response and render them as a dedicated "Sources" section or UI component (e.g. numbered footnotes, a collapsible panel, or pill-style chips) rather than inline links.
**Planned work:** Aggregator extracts unique `(title, url)` pairs → passes them as structured metadata → `MessageBubble.tsx` renders a Sources block below the response.
**Note:** Cross-agent deduplication of same-URL citations should also be handled here.


---


---

## Resolved

#### UI-1 — No streaming progress indicator during long responses
Added bouncing dots + "Generating response..." indicator in `MessageBubble.tsx`, shown when `isStreaming && !content && thinking.length > 0`. Verified working in browser during ~4 min MedGemma inference.

#### UI-2 — Chat does not auto-scroll to latest message
Implemented smart scroll in `ChatArea.tsx` using `isNearBottomRef`. Auto-scrolls only when within 100px of bottom; shows "↓ Scroll to bottom" button otherwise. Verified working in browser.

#### UI-4 — Page refresh on a chat URL loads blank empty state
Seeded `currentSessionId` from `window.location.pathname` in `App.tsx` using a lazy `useState` initializer.

#### UI-3 — Duplicate message bubbles on backend connection failure
`catch` block now maps over messages to replace the placeholder (`aiMsgId`) instead of pushing a new error bubble.

#### AGG-1 — Aggregator emits duplicate sections
Added explicit deduplication rules to the `node_aggregator` system prompt in `orchestrator.py`: merge near-identical recommendations, keep only first occurrence of repeated source facts.
