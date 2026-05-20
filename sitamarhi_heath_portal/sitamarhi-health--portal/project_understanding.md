# Project Understanding: Sitamarhi Health Portal

## Overview
A **React + TypeScript + Vite** web application for monitoring maternal and child health metrics at **CHC Nanpur**, Sitamarhi District, Bihar. This is a frontend-only demo/prototype that visualizes block-level health data with bilingual (English/Hindi) support.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **React 19** | UI framework |
| **TypeScript 6** | Type safety |
| **Vite 8** | Build tool & dev server |
| **Tailwind CSS 4** | Utility-first styling |
| **React Router DOM v7** | Client-side routing |
| **Recharts** | Charting library (available but custom SVG is used in pages) |
| **Material Symbols** | Icon set |
| **Vercel** | Deployment (configured via `vercel.json`) |

---

## Project Structure

```
sitamarhi-health--portal/
├── index.html                 # Entry HTML (title: "CHC Nanpur | Bihar Health Monitoring")
├── vite.config.ts             # Vite config: React plugin + Tailwind CSS plugin
├── eslint.config.js           # ESLint flat config (React + TypeScript)
├── tsconfig*.json             # TypeScript configs (app + node)
├── vercel.json                # SPA rewrites for Vercel deployment
├── package.json               # Dependencies & scripts
├── public/
│   └── favicon.svg
└── src/
    ├── main.tsx               # App entry: BrowserRouter + AuthProvider + App
    ├── App.tsx                # Route definitions
    ├── index.css              # Global styles (Tailwind directives + custom @theme)
    ├── assets/
    │   ├── hero.png
    │   ├── react.svg
    │   └── vite.svg
    ├── components/
    │   ├── KpiCard.tsx        # Reusable KPI metric card component
    │   ├── Layout.tsx         # Main layout wrapper (sidebar + header + content area)
    │   └── TopBar.tsx         # Sidebar nav + top header bar (responsive, mobile overlay)
    ├── context/
    │   └── AuthContext.tsx     # Simple auth context (hardcoded demo credentials)
    ├── data/
    │   └── mockData.ts        # All mock data: KPIs, benchmarks, alerts, ANC, anaemia, deliveries, zero-board, AI summary
    └── pages/
        ├── LoginScreen.tsx    # Login with role selection (Med Supervisor / ANM / ASHA)
        ├── Dashboard.tsx      # Block dashboard with KPIs, benchmarks, alerts, HSC rankings, AI insights
        ├── AncMonitoring.tsx  # ANC monitoring: trends, sector performance, block comparison
        ├── AnaemiaTracker.tsx # Anaemia case management: HB tests, severe cases, treatment tracking
        ├── DeliveryMonitoring.tsx # Institutional vs home delivery tracking
        ├── ZeroBoard.tsx      # Issue tracking board for non-performing ANMs/HSCs
        └── AiSummary.tsx      # Monthly AI-generated block performance summary (EN/HI)
```

---

## Routes

| Route | Page | Description | Auth Required |
|---|---|---|---|
| `/login` | LoginScreen | Role-based authentication | No |
| `/dashboard` | Dashboard | Block-level KPI dashboard | Yes |
| `/anc` | AncMonitoring | Antenatal care monitoring | Yes |
| `/anaemia` | AnaemiaTracker | Severe anaemia case tracking | Yes |
| `/delivery` | DeliveryMonitoring | Institutional vs home delivery | Yes |
| `/zero-board` | ZeroBoard | Non-compliance & issue tracking | Yes |
| `/ai-summary` | AiSummary | AI-generated monthly review (EN/HI) | Yes |
| `/*` | Redirect | Fallback to `/login` | - |

---

## Features

### 1. Authentication
- Role-based login: **Medical Supervisor**, **ANM**, **ASHA Worker**
- Demo credentials: `root` / `root`
- Protected routes via `ProtectedRoute` wrapper component
- Simple auth context (no backend, no JWT, no persistence)

### 2. Dashboard
- **KPI Cards**: ANC Registration (99%), 1st Trimester ANC (97%), 4+ ANC Completion (98%), Institutional Delivery (51%), Severe Anaemia (7%)
- **District Benchmarks**: CHC Nanpur vs Sitamarhi district average comparison bars
- **Critical Alerts**: HSC-wise alert table with severity (critical/warning)
- **Lowest Performing HSC Units**: Ranked list with metric bars
- **AI Block Review**: Summary insights card
- Floating action button (FAB)

### 3. ANC Monitoring
- Monthly trend chart (custom SVG line + area chart)
- Sector-wise performance table (achievement %, 1st trimester %)
- Block comparison bar chart
- Time range selectors

### 4. Anaemia Tracker
- KPI summary: Tests conducted, severe cases (Hb<7), orange alert (Hb 7-9), FCM stock
- Cases table: Patient details, HB levels, risk level (critical/orange/yellow), treatment status
- Color-coded severity badges

### 5. Delivery Monitoring
- KPI summary: Delivery ELA (target), total, institutional, home deliveries
- Monthly trend chart (institutional delivery %)
- ANM-wise delivery rate comparison
- District benchmark indicator

### 6. Zero Board
- Issue tracking for ANMs and HSC sectors
- Severity indicators: Red (critical), Yellow (warning), Green (okay)
- Days overdue tracking
- Action required descriptions

### 7. AI Summary
- Monthly narrative summary (EN + HI)
- Strengths & weaknesses list
- Recommended actions
- Metric comparison (current vs previous month)

---

## Data Strategy
- **All data is mocked** in `src/data/mockData.ts` - no backend API integration
- Data models defined as TypeScript interfaces:
  - `KpiMetric`, `BenchMarkItem`, `AlertItem`, `RankItem`
  - `SectorPerformance`, `ANMData`, `AnaemiaCase`, `DeliveryRecord`
  - `ZeroBoardItem`, `MonthlyTrend`
- Designed to be replaced with real API calls in production

---

## Styling
- **Tailwind CSS 4** with custom `@theme` tokens in `index.css`
- Google Fonts: **Public Sans** (headings), **Inter** (body)
- **Material Symbols** icon font
- Soft UI card design with rounded corners, shadows
- Color palette:
  - Primary: Custom blue
  - Status colors: Critical (red), Warning (orange/yellow), Success (green)
- Responsive: Mobile sidebar overlay, adaptive grid layouts

---

## DevOps & Deployment
- **Vite** for development and build
- **Vercel** deployment with SPA rewrites (`vercel.json`)
- Scripts: `dev`, `build`, `lint`, `preview`

---

## Key Observations
1. **Demo/Prototype Nature**: No real backend, hardcoded auth, mock data only
2. **Bilingual Ready**: Full English/Hindi toggle on Login and AI Summary pages
3. **Offline Capable**: Runs entirely client-side
4. **Mobile Responsive**: Sidebar becomes a slide-over on mobile; responsive grids
5. **No State Management**: Uses React's built-in state only (no Redux/Zustand)
6. **No Testing**: No test files or testing framework configured
7. **Custom Charts**: Many chart visualizations use inline SVG rather than Recharts, keeping the bundle lighter