# Project Aegis - Comprehensive Project Description

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Usability](#usability)
3. [How It Works](#how-it-works)
4. [Plan of Action](#plan-of-action)
5. [User Interface Flow](#user-interface-flow)

---

## 🎯 Project Overview

**Project Aegis** is an advanced, end-to-end misinformation detection and verification system designed to combat the spread of false information across digital platforms. The project combines AI-powered content analysis, automated fact-checking, and educational content generation to provide a comprehensive solution for identifying and debunking misinformation.

### Core Purpose

- **Early Detection**: Identify trending misinformation before it goes viral
- **Automated Verification**: Fact-check claims using AI and trusted sources
- **Educational Response**: Generate debunk posts and educational content
- **User Empowerment**: Provide tools for individuals to verify content themselves

### Key Components

1. **Backend API** (FastAPI) - Verification services for text, images, videos, and audio
2. **Agent Pipeline** (Python) - Automated trend scanning and fact-checking orchestration
3. **Frontend Web App** (React) - Interactive user interface for verification and learning

---

## 💡 Usability

### Target Users

#### 1. **General Public**

- **Use Case**: Verify suspicious content encountered on social media
- **Features**:
  - Upload images/videos for verification
  - Submit text claims for fact-checking
  - Voice input for hands-free verification
  - Access educational modules to learn about misinformation

#### 2. **Content Moderators**

- **Use Case**: Monitor and verify trending content at scale
- **Features**:
  - Automated trend scanning from Reddit
  - Batch verification of multiple claims
  - Real-time alerts via WebSocket
  - Structured JSON output for integration

#### 3. **Educators & Researchers**

- **Use Case**: Access educational content and track misinformation patterns
- **Features**:
  - Interactive learning modules
  - Progress tracking
  - Historical data access
  - Research-friendly data formats

### Key Usability Features

#### **Multi-Modal Input Support**

- **Text**: Direct text input or paste claims
- **Images**: Upload images or provide URLs for reverse image search
- **Videos**: Upload video files or YouTube/Instagram links
- **Audio**: Voice recording with speech-to-text conversion
- **Mixed**: Combine multiple input types in a single verification

#### **Intelligent Processing**

- **Auto-Detection**: System automatically identifies content type
- **Context Understanding**: AI extracts claim context and dates
- **Batch Processing**: Efficient handling of multiple items
- **Real-time Feedback**: WebSocket updates for long-running processes

#### **Accessibility**

- **Dark Mode**: Full dark mode support for reduced eye strain
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Keyboard Navigation**: Full keyboard support for accessibility
- **Clear Feedback**: Visual indicators for verification status

#### **Educational Integration**

- **Contextual Learning**: Educational content based on verification results
- **Progress Tracking**: Track learning progress and earn points
- **Difficulty Levels**: Beginner, intermediate, and advanced modules
- **Interactive Content**: Expandable sections and practical examples

---

## ⚙️ How It Works

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  ChatbotView │  │  ModulesView │  │ ProgressView │          │
│  │  (Verify)    │  │  (Learn)     │  │  (Track)     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                          │                                        │
│                    WebSocket / REST API                           │
└──────────────────────────┼────────────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────────────┐
│                    BACKEND (FastAPI)                               │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Input Processor (LLM Router)                 │    │
│  │  • Detects content type (text/image/video/audio)         │    │
│  │  • Extracts context and metadata                         │    │
│  │  • Routes to appropriate verifier                         │    │
│  └──────────────────────────────────────────────────────────┘    │
│                          │                                        │
│         ┌────────────────┼────────────────┐                      │
│         │                │                │                      │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐             │
│  │   Text      │  │   Image     │  │   Video    │             │
│  │  Verifier   │  │   Verifier  │  │  Verifier  │             │
│  │             │  │             │  │            │             │
│  │ • Google    │  │ • Reverse   │  │ • Frame    │             │
│  │   Search    │  │   Image     │  │   Extract  │             │
│  │ • Fact      │  │   Search    │  │ • Deepfake │             │
│  │   Check     │  │ • Metadata  │  │   Detect   │             │
│  │   API       │  │   Analysis   │  │ • Metadata │             │
│  └─────────────┘  └─────────────┘  └────────────┘             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Educational Content Generator                      │   │
│  │  • Module content generation                              │   │
│  │  • Contextual learning based on results                   │   │
│  │  • Redis caching for performance                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         MongoDB Service                                   │   │
│  │  • Store verification results                             │   │
│  │  • Change streams for real-time updates                  │   │
│  │  • Recent posts retrieval                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────────────┐
│                    AGENT PIPELINE (Python)                        │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │         Orchestrator Agent                                │    │
│  │  • Coordinates all agents                                 │    │
│  │  • Manages workflow                                      │    │
│  │  • Aggregates results                                    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                          │                                        │
│         ┌────────────────┼────────────────┐                      │
│         │                │                │                      │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐             │
│  │   Trend     │  │    Claim     │  │ Explanation│             │
│  │   Scanner   │  │   Verifier   │  │   Agent    │             │
│  │             │  │              │  │            │             │
│  │ • Reddit    │  │ • Google     │  │ • Debunk   │             │
│  │   Monitor   │  │   Search     │  │   Posts     │             │
│  │ • Web       │  │ • Fact       │  │ • Educ.     │             │
│  │   Scraping  │  │   Checkers   │  │   Content   │             │
│  │ • AI        │  │ • Batch      │  │ • Batch     │             │
│  │   Analysis  │  │   Verify     │  │   Generate  │             │
│  └─────────────┘  └─────────────┘  └────────────┘             │
│                                                                   │
│  Powered by: Google Gemini 2.5 Flash, Google Agents SDK          │
└───────────────────────────────────────────────────────────────────┘
```

### Technical Stack

#### **Frontend**

- **React 18** with Vite for fast development
- **Framer Motion** for smooth animations
- **Tailwind CSS** for responsive styling
- **Lucide React** for icons
- **WebSocket** for real-time updates

#### **Backend**

- **FastAPI** for REST API and WebSocket server
- **Google Gemini 2.5 Flash** for AI analysis
- **Google Custom Search API** for fact-checking
- **Pillow/OpenCV** for image/video processing
- **MongoDB** for data persistence
- **Redis** (Upstash) for caching

#### **Agent Pipeline**

- **Google Agents SDK** for multi-agent orchestration
- **PRAW** for Reddit API integration
- **Beautiful Soup/Trafilatura** for web scraping
- **Batch Processing** for efficiency (95% API call reduction)

### Core Technologies

#### **AI & Machine Learning**

- **Google Gemini 2.5 Flash**: Content analysis, summarization, claim extraction
- **Google Generative AI SDK**: Primary AI interface
- **LiteLLM**: Multi-provider LLM integration
- **Batch Processing**: Optimized workflows reducing API calls by 90%

#### **Data Sources**

- **Reddit API** (PRAW): Trending post monitoring
- **Google Custom Search**: Fact-checking source verification
- **NewsAPI**: News source aggregation
- **RSS/Atom Feeds**: Real-time content monitoring

#### **Verification Methods**

1. **Text Verification**

   - Google Custom Search across fact-checking sites
   - AI-powered claim extraction and analysis
   - Source credibility assessment
   - Verdict: true/false/mixed/uncertain

2. **Image Verification**

   - Reverse image search
   - Metadata analysis (EXIF data)
   - AI-powered content analysis
   - Context matching with claims

3. **Video Verification**

   - Frame extraction at intervals
   - Deepfake detection
   - Metadata analysis
   - Temporal consistency checks

4. **Audio Verification**
   - Speech-to-text conversion
   - Deepfake audio detection
   - Content analysis
   - Voice authenticity checks

---

## 📋 Plan of Action

### Workflow Overview

The system operates in two primary modes:

#### **Mode 1: User-Initiated Verification (Frontend → Backend)**

```
User Input
    │
    ├─► Text Claim
    ├─► Image Upload/URL
    ├─► Video Upload/URL
    └─► Audio Recording
    │
    ▼
Input Processor (LLM)
    │
    ├─► Detects content type
    ├─► Extracts context
    └─► Routes to verifier
    │
    ▼
Appropriate Verifier
    │
    ├─► Text Verifier → Google Search + Fact Check API
    ├─► Image Verifier → Reverse Image Search + Metadata
    ├─► Video Verifier → Frame Analysis + Deepfake Detection
    └─► Audio Verifier → Speech-to-Text + Deepfake Detection
    │
    ▼
Result Aggregation
    │
    ├─► Combines multiple results
    ├─► Generates overall verdict
    └─► Formats response
    │
    ▼
User Response
    │
    ├─► Verification verdict
    ├─► Confidence score
    ├─► Source citations
    └─► Educational content (optional)
```

#### **Mode 2: Automated Pipeline (Agent System)**

```
Orchestrator Agent Starts
    │
    ▼
Step 1: Trend Scanner Agent
    │
    ├─► Scans Reddit subreddits
    ├─► Identifies trending posts
    ├─► Extracts post content
    └─► Web scrapes external links
    │
    ▼
Step 2: AI Analysis (Gemini 2.5 Flash)
    │
    ├─► Batch processes all posts
    ├─► Generates summaries
    ├─► Extracts verifiable claims
    └─► Assesses risk levels
    │
    ▼
Step 3: Claim Verifier Agent
    │
    ├─► Receives claims from trend scanner
    ├─► Batch verifies claims (up to 15 at once)
    ├─► Google Custom Search for fact-checking
    ├─► AI analysis of search results
    └─► Generates verification verdicts
    │
    ▼
Step 4: Explanation Agent
    │
    ├─► Generates debunk posts for false claims
    ├─► Creates educational content
    ├─► Formats for social media
    └─► Batch generates (up to 10 at once)
    │
    ▼
Step 5: Results Compilation
    │
    ├─► Combines all results
    ├─► Creates structured JSON
    ├─► Saves to MongoDB
    └─► Triggers WebSocket updates
    │
    ▼
Final Output
    │
    ├─► Structured JSON file
    ├─► MongoDB records
    └─► Real-time frontend updates
```

### Detailed Process Steps

#### **User Verification Flow**

1. **Input Collection**

   - User enters text, uploads file, or records audio
   - System validates input format and size
   - Files are temporarily stored

2. **Content Processing**

   - Input Processor uses Gemini to:
     - Identify content type
     - Extract claim context
     - Determine verification approach
   - Routes to appropriate verifier service

3. **Verification Execution**

   - **Text**: Searches Google Custom Search (fact-checking sites)
   - **Image**: Performs reverse image search + metadata analysis
   - **Video**: Extracts frames, analyzes for deepfakes
   - **Audio**: Converts to text, checks for deepfake audio

4. **Result Generation**

   - Aggregates verification results
   - Determines overall verdict (true/false/mixed/uncertain)
   - Extracts key messages and sources
   - Formats user-friendly response

5. **Response Delivery**
   - Returns structured JSON to frontend
   - Displays verdict with confidence
   - Shows source citations
   - Optionally generates educational content

#### **Automated Pipeline Flow**

1. **Trend Scanning Phase**

   - Orchestrator triggers Trend Scanner Agent
   - Scans configured Reddit subreddits
   - Filters by velocity and engagement
   - Collects post metadata and content

2. **Content Enrichment**

   - Scrapes external links from posts
   - Extracts full article content
   - Combines Reddit content with external sources

3. **AI Analysis Phase**

   - Sends all posts to Gemini 2.5 Flash in batch
   - AI generates:
     - Comprehensive summaries
     - Verifiable claims extraction
     - Risk level assessment
     - Context understanding

4. **Claim Verification Phase**

   - Extracts verifiable claims from AI analysis
   - Sends claims to Claim Verifier Agent
   - Batch processes up to 15 claims simultaneously
   - Each claim:
     - Searches Google Custom Search
     - Queries fact-checking sites (Snopes, PolitiFact, etc.)
     - AI analyzes search results
     - Generates verification verdict

5. **Explanation Generation Phase**

   - For false/mixed claims, triggers Explanation Agent
   - Generates debunk posts with:
     - Clear factual corrections
     - Source citations
     - Educational explanations
   - Formats for social media sharing

6. **Results Compilation**
   - Orchestrator combines all results
   - Creates structured JSON output
   - Saves to MongoDB
   - Updates frontend via WebSocket

### Performance Optimizations

#### **Batch Processing**

- **95% API Call Reduction**: Processes 20 posts in 1 API call instead of 20
- **Efficient Resource Usage**: Minimizes API costs and latency
- **Scalable Architecture**: Handles large volumes efficiently

#### **Caching Strategy**

- **Redis Caching**: Educational content cached for 24 hours
- **URL Processing Cache**: Prevents duplicate processing
- **Result Storage**: MongoDB for historical data

#### **Error Handling**

- **Graceful Degradation**: Continues processing if one component fails
- **Retry Logic**: Automatic retries for transient failures
- **Fallback Mechanisms**: Basic processing when APIs unavailable

---

## 🖥️ User Interface Flow

### Main Application Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Shell                          │
│  ┌──────────────┐                    ┌──────────────────┐   │
│  │   Sidebar    │                    │   Main Content   │   │
│  │              │                    │                  │   │
│  │ • Navigation │                    │  ┌────────────┐ │   │
│  │ • Current    │                    │  │   Header   │ │   │
│  │   Rumours    │                    │  └────────────┘ │   │
│  │              │                    │  ┌────────────┐ │   │
│  │              │                    │  │   View     │ │   │
│  │              │                    │  │   Content   │ │   │
│  │              │                    │  └────────────┘ │   │
│  └──────────────┘                    └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Navigation Structure

```
Sidebar Navigation
    │
    ├─► Fact Check (Chatbot)
    │   └─► Verification Interface
    │
    ├─► Educational Modules
    │   ├─► Modules Grid View
    │   ├─► Module Content View
    │   └─► Progress Tracking
    │
    └─► Current Rumours
        └─► Real-time Rumour Feed
```

### Detailed UI Flows

#### **Flow 1: Fact-Checking (Chatbot View)**

```
1. User lands on Chatbot View
   │
   ├─► Sees welcome message from AI assistant
   ├─► Input area with text field
   ├─► File upload button
   ├─► Voice recording button
   └─► Send button
   │
2. User provides input
   │
   ├─► Option A: Types text claim
   │   └─► Text appears in input field
   │
   ├─► Option B: Uploads file(s)
   │   ├─► File preview appears above input
   │   ├─► Shows file name and icon
   │   └─► Can remove files before sending
   │
   ├─► Option C: Records audio
   │   ├─► Clicks microphone button
   │   ├─► Recording banner appears (red)
   │   ├─► Timer shows recording duration
   │   ├─► Stops recording
   │   ├─► Audio sent to speech-to-text
   │   └─► Transcript appears in input field
   │
   └─► Option D: Combines multiple inputs
       └─► Can mix text + files + audio
   │
3. User clicks "Send"
   │
   ├─► Input cleared
   ├─► User message appears in chat
   ├─► Loading indicator shows
   └─► Request sent to backend
   │
4. Backend Processing
   │
   ├─► Input Processor analyzes content
   ├─► Routes to appropriate verifier
   ├─► Verification executes
   └─► Results aggregated
   │
5. Response Display
   │
   ├─► AI message appears in chat
   ├─► Shows verification verdict
   ├─► Displays confidence level
   ├─► Lists source citations (if any)
   └─► Optionally shows educational content
   │
6. User Actions
   │
   ├─► Can ask follow-up questions
   ├─► Can verify another claim
   ├─► Can navigate to educational modules
   └─► Can view current rumours
```

#### **Flow 2: Educational Modules**

```
1. User navigates to "Educational Modules"
   │
   ├─► Sees modules grid view
   ├─► Each module shows:
   │   ├─► Title
   │   ├─► Description
   │   ├─► Difficulty levels
   │   ├─► Estimated time
   │   └─► Completion status
   │
2. User selects a module
   │
   ├─► Module content loads
   ├─► Shows module header with:
   │   ├─► Title and description
   │   ├─► Difficulty selector
   │   └─► Back button
   │
3. Module Content View
   │
   ├─► Overview section
   ├─► Learning objectives
   ├─► Content sections (expandable)
   │   ├─► Click to expand/collapse
   │   ├─► Interactive examples
   │   └─► Visual aids
   │
   ├─► Practical tips section
   ├─► Common mistakes section
   └─► Complete module button
   │
4. User interacts with content
   │
   ├─► Expands sections to read
   ├─► Changes difficulty level
   │   └─► Content updates dynamically
   ├─► Completes module
   │   └─► Earns points
   │   └─► Progress updated
   │
5. User navigates back
   │
   └─► Returns to modules grid
      └─► Completed modules marked
```

#### **Flow 3: Current Rumours (Real-time Feed)**

```
1. User views sidebar
   │
   ├─► "Current Rumours" section visible
   ├─► Shows recent rumours from MongoDB
   └─► Updates via WebSocket
   │
2. User clicks on a rumour
   │
   ├─► Rumour modal opens (full screen)
   ├─► Shows rumour details:
   │   ├─► Claim text
   │   ├─► Verification status
   │   ├─► Verdict (true/false/mixed)
   │   ├─► Source citations
   │   ├─► Explanation
   │   └─► Original post link
   │
3. User can interact
   │
   ├─► Close modal
   ├─► View original post
   └─► Request verification
   │
4. Real-time Updates
   │
   └─► New rumours appear automatically
      └─► Via WebSocket connection
```

#### **Flow 4: Progress Tracking**

```
1. User navigates to "Your Progress"
   │
   ├─► Progress overview displayed
   ├─► Shows:
   │   ├─► Current level
   │   ├─► Points earned
   │   ├─► Streak days
   │   ├─► Total time spent
   │   └─► Badges earned
   │
2. Progress Summary
   │
   ├─► Completed modules list
   ├─► Module completion percentages
   └─► Learning statistics
   │
3. User can
   │
   ├─► View detailed progress
   ├─► See achievement badges
   └─► Track learning journey
```

### UI Components Breakdown

#### **Sidebar Components**

1. **SidebarHeader**

   - Project logo/title
   - Collapse/expand toggle
   - Dark mode indicator

2. **SidebarNavigation**

   - Fact Check button
   - Educational Modules button
   - Progress button
   - Active state indicators

3. **CurrentRumours**
   - List of recent rumours
   - Click to view details
   - Real-time updates

#### **Main Content Views**

1. **ChatbotView**

   - Message history
   - Input area
   - File upload
   - Voice recording
   - Loading states

2. **ModulesView**

   - Grid of available modules
   - Module cards with details
   - Filter/search (future)

3. **ContentView**

   - Module content display
   - Expandable sections
   - Interactive elements
   - Completion tracking

4. **ProgressView**
   - Progress overview
   - Statistics
   - Achievements
   - Learning history

#### **Modal Components**

1. **RumourModal**

   - Full-screen rumour details
   - Verification information
   - Source citations
   - Close button

2. **InfoModal**
   - Project information
   - How to use guide
   - About section

### Responsive Design

#### **Desktop (≥1024px)**

- Sidebar always visible (collapsible)
- Full feature set
- Multi-column layouts
- Hover interactions

#### **Tablet (768px - 1023px)**

- Collapsible sidebar
- Touch-optimized controls
- Adjusted spacing
- Swipe gestures

#### **Mobile (<768px)**

- Sidebar hidden by default
- Hamburger menu
- Full-screen modals
- Touch-friendly buttons
- Optimized input areas

### Dark Mode Support

- **Automatic Detection**: Respects system preference
- **Manual Toggle**: Toggle in header
- **Persistent**: Saves preference to localStorage
- **Smooth Transitions**: Animated color changes
- **Full Coverage**: All components support dark mode

### Accessibility Features

- **Keyboard Navigation**: Full keyboard support
- **Screen Reader**: ARIA labels and semantic HTML
- **Color Contrast**: WCAG AA compliant
- **Focus Indicators**: Clear focus states
- **Error Messages**: Descriptive error feedback

---

## 🔄 Integration Points

### Frontend ↔ Backend Communication

1. **REST API Endpoints**

   - `/chatbot/verify` - Main verification endpoint
   - `/verify/text` - Text-only verification
   - `/verify/image` - Image verification
   - `/verify/video` - Video verification
   - `/educational/modules` - Get modules list
   - `/educational/modules/{id}` - Get module content
   - `/mongodb/recent-posts` - Get recent rumours
   - `/speech-to-text` - Audio transcription

2. **WebSocket Connection**
   - `/ws` - Real-time updates
   - MongoDB change streams
   - Live rumour feed updates

### Backend ↔ Agent Pipeline

1. **Data Flow**

   - Agent pipeline runs independently
   - Results stored in MongoDB
   - Backend reads from MongoDB
   - Frontend receives via WebSocket

2. **Orchestration**
   - Can be triggered manually
   - Can be scheduled (cron jobs)
   - Can be API-triggered

---

## 📊 Data Flow Summary

```
User Input
    ↓
Frontend (React)
    ↓ (HTTP/WebSocket)
Backend API (FastAPI)
    ↓
Input Processor (Gemini)
    ↓
Verifier Services
    ↓
Result Aggregation
    ↓
MongoDB Storage
    ↓
WebSocket Broadcast
    ↓
Frontend Update

OR

Agent Pipeline (Automated)
    ↓
Trend Scanner
    ↓
AI Analysis (Gemini)
    ↓
Claim Verifier
    ↓
Explanation Agent
    ↓
MongoDB Storage
    ↓
WebSocket Broadcast
    ↓
Frontend Update
```

---

## 🎯 Key Features Summary

### For End Users

✅ Multi-modal content verification (text, image, video, audio)  
✅ Real-time verification results  
✅ Educational modules for learning  
✅ Progress tracking and gamification  
✅ Dark mode and responsive design  
✅ Voice input support

### For Content Moderators

✅ Automated trend scanning  
✅ Batch verification processing  
✅ Real-time alerts via WebSocket  
✅ Structured JSON output  
✅ Historical data access

### For Developers

✅ RESTful API for integration  
✅ WebSocket for real-time updates  
✅ Modular architecture  
✅ Comprehensive logging  
✅ Error handling and fallbacks

---

## 🚀 Future Enhancements

### Planned Features

- Twitter/X integration for trend scanning
- Real-time dashboard for monitoring
- Custom model training for domain-specific detection
- Multi-language support
- Video content analysis (YouTube/TikTok)
- Network analysis for influence tracking
- Mobile application
- Public API for researchers

---

**Project Aegis - Defending Truth in the Digital Age**

Built with ❤️ for combating misinformation | Powered by Google Gemini 2.5 Flash
