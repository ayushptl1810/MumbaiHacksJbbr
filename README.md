## Project Aegis – Real‑Time Fact‑Checking & Educational Platform

Project Aegis is an end‑to‑end fact‑checking and media literacy platform.  
It monitors rumours in real time, debunks them using AI + editorial logic, and then teaches users how to spot similar misinformation through curated educational modules.

The stack is:

- **Frontend**: React (Vite), React Router, Framer Motion, Tailwind‑style utilities
- **Backend**: FastAPI, MongoDB, Razorpay subscriptions, WebSockets
- **Infra / Tooling**: Axios, ESLint, Vite dev server

---

## High‑Level Features

- **Authentication & Profiles**
  - Email + password signup/login.
  - Signup captures: **name, email, phone number, age**, and **domain preferences** (Politics, Technology, Health, Crime, Military, Sports, Entertainment, Social Media only).
  - User profile shows personal info, interests, subscription tier, and next renewal date.

- **Subscriptions & Billing**
  - Razorpay subscription integration (Pro / Enterprise vs Free).
  - Subscriptions valid for **one month**; renewal handled via Razorpay subscription lifecycle.
  - Subscription data stored in MongoDB, and each user has a `subscription_tier` (Free / Pro / Enterprise).
  - Profile icon + profile page styling respond to tier with distinct colors:
    - Free: grey
    - Pro: ocean blue / cyan
    - Enterprise: deep purple

- **Rumour Feed & Live Alerts**
  - Real‑time rumours/claims streamed via WebSocket from MongoDB change streams.
  - Only verdicts **False** and **Uncertain** are displayed in the live alerts section.
  - Each rumour shows: claim, mapped verdict, **confidence percentage**, and time‑ago.
  - Clicking a rumour opens a rich **Rumour Modal** with:
    - Claim, Verdict, Verified On
    - Confidence percentage
    - Long **Body** section with the reasoning
    - **Summary** section with “Read more / Read less” handling for long text
    - Original post link (using `post_content.heading` linked to `final_source`)
    - Even grid layout for Body/Summary and Verdict/Verified On

- **Chatbot Verification (Verify page)**
  - Chatbot view allows users to type multi‑line input, attach files, or record audio.
  - Frontend preserves multi‑line formatting; backend endpoint `/chatbot/verify` handles mixed text+files.
  - Uses an `InputProcessor` on the backend to route to the right verification workflow.

- **Educational Modules**
  - Educational content is generated/stored in MongoDB (`weekly_posts.educational_module`) and surfaced via:
    - `GET /educational/modules` (summaries)
    - `GET /educational/modules/{module_id}` (detailed module)
  - Each module describes a **misinformation technique** (e.g., manipulation patterns), with:
    - Overview & technique explanation
    - Red flags
    - Verification tips
    - Related patterns & user action items
    - Real‑world example (heading, claim, verdict, body, tags, source URL)

- **Personalized Learning (“For You” modules)**
  - Each educational module has `tags` (derived from post `metadata.tags`).
  - The Modules page shows domain tags on the cards and in the module overview.
  - Users can:
    - Filter by **text search** (module title/description).
    - Filter by **domain tags** (attractive pill‑style chips).
    - Click **“For You”** to automatically filter modules based on their own `domain_preferences` from the profile.

---

## How People Actually Use It (End‑User Flows)

- **1. Quick rumour check from social media**
  - User sees a viral post on X / Instagram and copies the text or link.
  - Opens **Verify → Chatbot**, pastes the claim (multi‑line supported), optionally attaches a screenshot or video.
  - The chatbot routes the request to the right verifier and responds with:
    - Verdict (False / Uncertain / Mostly True, etc.)
    - Reasoning
    - Sources and confidence
  - If the rumour is part of an existing debunk, it also appears in **Live Alerts**, with a modal for deep‑dive context.

- **2. Browsing live misinformation to stay ahead**
  - From any page, the user clicks the **alerts icon** in the navbar.
  - Sees a live stream of only **False** and **Uncertain** posts, with confidence percentages.
  - Clicks into a card → opens the **Rumour Modal** with:
    - Clean, balanced layout (Claim / Verdict / Verified On / Confidence / Body / Summary).
    - “Read more” where needed so long summaries remain readable.
    - Direct link to the original post/source.

- **3. Setting up a learning profile**
  - On **Signup**, the user selects domains they care about (e.g. Politics + Technology + Health).
  - These preferences:
    - Shape the “For You” modules list.
    - Are visible as tags in the **Profile** page for transparency.

- **4. Learning with educational modules**
  - User goes to **Modules** to understand patterns behind misinformation, not just individual cases.
  - On the main Modules page they can:
    - Use search to look for specific topics.
    - Toggle domain tags to focus on subjects (e.g. “Elections”, “Health”, “Finance” if present in tags).
    - Click **For You** to instantly filter to modules that match their saved domain preferences.
  - Each module card shows:
    - Title, description, estimated time, difficulty badges, and attractive domain tags.
  - Inside a module detail view, the user sees:
    - High‑level overview and tags at the top.
    - Red flags, verification tips, real‑world example, and concrete “What you can do” actions.

- **5. Managing access and billing**
  - In the **Profile** page, the user can see:
    - Whether they’re on Free / Pro / Enterprise (with a matching color theme).
    - Renewal date and days remaining.
  - Clicking **Manage Subscription** takes them to the subscription page:
    - Free users can see what they get by upgrading.
    - Pro/Enterprise users can manage billing via Razorpay (create/cancel subscription).

These flows are designed so a first‑time user can go from **“I saw this claim, is it real?”** to **“I understand why this pattern of misinformation works and how to defend against it”** in a few clicks.

---

## Architecture Overview

### Frontend

Located under `frontend/` (Vite + React).

Key pieces:

- `src/contexts/AuthContext.jsx`
  - Manages `user`, `isAuthenticated`, `loading`.
  - Provides `login`, `logout`, and `refreshUser` (calls `/auth/me`).
  - Pulls `subscription_tier` and domain preferences into the global context.

- `src/layouts/Navbar.jsx`
  - Top nav with routes: Home, Verify, Modules, Subscription (only when logged in).
  - Rumours sidebar and Rumour modal trigger.
  - Auth area:
    - Shows login button for guests.
    - For authenticated users: colored profile icon (tier‑aware) and logout button.

- `src/pages/Auth/Signup.jsx`
  - Signup form collects:
    - Full name
    - Email
    - Password + confirm
    - Phone number
    - Age
    - **Domain preferences**: checkboxes for Politics, Technology, Health, Crime, Military, Sports, Entertainment, Social Media only.
  - Calls `/auth/signup` with `domain_preferences` and personal data.

- `src/pages/Profile/Profile.jsx`
  - Protected profile page (via `ProtectedRoute`).
  - Fetches subscription status via `subscriptionService.getSubscriptionStatus(user.id)`.
  - Displays:
    - Personal info: name, email, phone, age
    - Interests (domain preferences) as animated tags
    - Subscription card with tier color, status, next renewal date, days remaining
    - Razorpay subscription metadata (last payment, subscription ID).
  - “Manage Subscription” and “Back to Home” actions.

- `src/pages/Subscription/Subscription.jsx`
  - Lists Free / Pro / Enterprise plans with a modern pricing table UI.
  - Loads Razorpay Checkout script and config.
  - For Pro:
    - Fetches or creates Razorpay plan.
    - Calls `/subscriptions/create` with `user.id` and plan ID.
    - Opens Razorpay checkout for recurring subscription.
    - On success, refreshes user (`refreshUser`) and shows confirmation.

- `src/pages/Verify/Verify.jsx` & `src/pages/Verify/ChatbotView.jsx`
  - Full chat layout with message history, file upload, audio recording, and multi‑line textarea.
  - Submits messages to `/chatbot/verify` as `FormData` (`text_input` + `files`).
  - Renders AI and user bubbles, preserves newlines via `whitespace-pre-wrap`.

- `src/hooks/useRumoursFeed.js`
  - Connects to WebSocket feed for real‑time rumours.
  - Transforms MongoDB documents into frontend shape:
    - Maps verdict strings to **False** or **Uncertain**.
    - Derives numeric confidence from `confidence_percentage` or a confidence level string.
  - Filters to only include relevant verdicts for live alerts.

- `src/components/RumourCard.jsx` & `src/components/RumourModal.jsx`
  - Card shows claim snippet, verdict, and confidence percentage between verdict and time‑ago.
  - Modal has redesigned layout:
    - Balanced grid for Verdict/Verified On and Body/Summary.
    - Read‑more / Read‑less for long text.
    - Original post link with heading + external‑link icon.

- `src/pages/Modules/Modules.jsx`
  - Fetches the educational modules list.
  - **Main modules view**:
    - Search field.
    - “For You” button (if logged in & has domain preferences).
    - Tag filter chips for domains/tags.
    - Module cards using `ModuleCard` with difficulty badges and tags.
  - **Module detail view**:
    - Overview header with title, description, and domain tags.
    - Stats cards (time, trending score, red flags, verification tips).
    - Technique explanation, red flags, verification tips, real‑world example, user actions, related patterns.

---

### Backend

Located under `backend/` (FastAPI).

Key areas:

- `main.py`
  - Auth endpoints:
    - `POST /auth/signup` – creates user in MongoDB (`users` collection), hashes password, stores domain & tag preferences.
    - `POST /auth/login` – verifies credentials, returns mock token and user payload.
    - `GET /auth/me` – derives user from mock token, attaches `subscription_tier` (from user doc or active subscription).
  - Subscription endpoints (Razorpay):
    - `GET /subscriptions/config` – exposes Razorpay key ID to frontend.
    - `POST /subscriptions/plans` / `GET /subscriptions/plans` – plan management.
    - `POST /subscriptions/create` – creates Razorpay subscription + upserts a document into `subscriptions` collection.
    - `GET /subscriptions/status` – returns latest subscription per user, syncing status with Razorpay.
    - `POST /subscriptions/cancel` – cancels subscription and updates MongoDB.
    - `POST /webhooks/razorpay` – processes `subscription.activated`, `subscription.charged`, `subscription.cancelled`, `payment.failed` events and updates MongoDB + user `subscription_tier`.
  - Educational modules:
    - `GET /educational/modules` – pulls summarized modules from `weekly_posts` via `MongoDBService.get_educational_modules_list()`.
    - `GET /educational/modules/{module_id}` – detailed module via `get_educational_module_by_id()`.
  - Chatbot verification:
    - `POST /chatbot/verify` – entrypoint that passes `text_input` and `files` to an `InputProcessor`, then routes to proper verification logic.
  - Classic verify endpoints:
    - `/verify/text`, `/verify/image`, `/verify/video` – direct verification routes for text, image, and video evidence.

- `services/mongodb_service.py`
  - Connects to MongoDB (`aegis` DB) with collections:
    - `debunk_posts`, `weekly_posts`, `subscriptions`, `users`, `chat_sessions`, `chat_messages`.
  - Rumours:
    - `get_recent_posts`, `search_similar_rumours` etc.
  - Educational modules:
    - `get_educational_modules_list()` – returns unique modules by misinformation type with tags, trending score, etc.
    - `get_educational_module_by_id()` – returns full module content plus example and tags.
  - Users:
    - `create_user`, `get_user_by_email`, `get_user_by_id`.
    - `update_user_subscription_tier(user_id, subscription_tier)` – keeps `subscription_tier` in sync with Razorpay events.
  - Subscriptions:
    - `upsert_subscription`, `get_user_subscription`, `update_subscription_status`, `get_subscription_by_razorpay_id`.

---

## Data Models (Conceptual)

### User (MongoDB `users` collection)

- `name`: string
- `email`: string (unique)
- `password`: sha256 hash (demo; use bcrypt in prod)
- `phone_number`: string
- `age`: number
- `domain_preferences`: string[] (domains chosen at signup)
- `tag_preferences`: string[] (e.g., Misinformation, Fact Check, Viral – if extended)
- `subscription_tier`: `"Free" | "Pro" | "Enterprise"`
- `created_at`, `updated_at`: timestamps

### Subscription (MongoDB `subscriptions` collection)

- `user_id`: string (ref to user)
- `razorpay_subscription_id`: string
- `razorpay_plan_id`: string
- `plan_name`: string (`"Pro"`, `"Enterprise"`, etc.)
- `status`: string (`"created"`, `"active"`, `"cancelled"`, `"expired"`, etc.)
- `amount`: number
- `currency`: string (`"INR"`)
- `current_start`, `current_end`, `next_billing_at`: unix timestamps
- `last_payment_*` fields
- `created_at`, `updated_at`

### Educational Module (from `weekly_posts.educational_module`)

- `misinformation_type`: human‑readable name (also used for module ID)
- `technique_explanation`: string
- `red_flags`: string[]
- `verification_tips`: string[]
- `related_patterns`: string[]
- `user_action_items`: string[]
- `sources_of_technique`: string[]
- `trending_score`: number
- Tags & context:
  - `tags`: from `post.metadata.tags`
  - `example.heading`, `example.body`, `example.claim`, `example.verdict`, `example.tags`, `example.source_url`

---

## Running the Project

> These commands may need adjustment depending on your environment; use them as a reference.

### Backend (FastAPI)

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure MongoDB is running and `MONGO_CONNECTION_STRING` is set in `backend/.env`.
4. Ensure Razorpay keys are configured in `backend/.env` (`RAZORPAY_ID`, `RAZORPAY_KEY`, webhook secret).
5. Run the FastAPI app (using uvicorn, example):

```bash
uvicorn main:app --reload
```

The backend will listen on `http://127.0.0.1:8000` by default.

### Frontend (Vite + React)

1. From the `frontend/` directory, install dependencies:

```bash
npm install
```

2. Ensure `VITE_API_BASE_URL` is set in `frontend/.env` (e.g. `http://127.0.0.1:8000`).
3. Start the dev server:

```bash
npm run dev
```

The app will usually run on `http://localhost:5173` (or similar) with HMR.

---

## How Personalization Works (For You Filters)

1. User signs up and chooses domain preferences (e.g. Politics, Technology, Health).
2. These preferences are stored in MongoDB and returned via `/auth/me`.
3. The Modules page pulls `user.domain_preferences` from `AuthContext`.
4. Clicking **For You** sets the modules tag filter to those preferences.
5. Only modules whose `tags` intersect with the selected preferences are shown.

This links rumours, educational content, and user profile into a cohesive personalized learning path.

---

## Roadmap Ideas

- Richer recommendation engine for modules (based on past completions + rumours seen).
- More granular subscription tiers and usage‑based limits.
- Multi‑language support for rumours, chatbot, and educational content.
- Admin UI for curators to manage modules, tags, and verification rules.

Project Aegis is designed to be a **high‑signal, high‑trust companion** for users navigating today’s information landscape—offering real‑time fact‑checks, deep context, and practical education, all in one place.

