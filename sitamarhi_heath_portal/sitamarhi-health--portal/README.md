# Sitamarhi Health Portal 🏥

A **React + TypeScript + Vite** web application for monitoring maternal and child health metrics at **CHC Nanpur**, Sitamarhi District, Bihar. This is a frontend prototype/demo that visualizes block-level health data with bilingual (English/Hindi) support.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint
npm run lint
```

## 🔑 Demo Credentials

| Username | Password | Role |
|----------|----------|------|
| `root` | `root` | Medical Supervisor |
| `root` | `root` | ANM |
| `root` | `root` | ASHA Worker |

## 📋 Features

### Dashboard
- Block-level KPI cards (ANC Registration, 1st Trimester ANC, 4+ ANC Completion, Institutional Delivery, Severe Anaemia)
- District benchmark comparisons (CHC Nanpur vs Sitamarhi district average)
- Critical HSC-wise alert table with severity indicators
- Lowest performing HSC unit rankings
- AI-generated block strategy insights

### ANC Monitoring
- Monthly trend chart with target and benchmark lines
- Sector-wise performance table with achievement and 1st trimester registration rates
- Block comparison across Sitamarhi district

### Anaemia Tracker
- HB test summary KPIs
- Patient-level case management table with severity (critical/orange/yellow)
- Treatment status tracking (pending, FCM given, referred, follow-up)

### Delivery Monitoring
- Delivery target vs achievement KPIs
- Monthly institutional delivery trend chart
- ANM-wise delivery rate comparison

### Zero Board
- Non-compliance and issue tracking for ANMs and HSC sectors
- Color-coded severity (Red: Critical, Yellow: Warning, Green: OK)
- Days overdue and action required tracking

### AI Summary
- Monthly bilingual (English/Hindi) narrative summary
- Strengths, weaknesses, and recommended actions
- Metric comparison (current vs previous month)

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **React 19** | UI framework |
| **TypeScript 6** | Type safety |
| **Vite 8** | Build tool & dev server |
| **Tailwind CSS 4** | Utility-first styling |
| **React Router DOM v7** | Client-side routing |
| **Recharts** | Charting library |
| **Material Symbols** | Icon set |
| **Vercel** | Deployment platform |

## 📁 Project Structure

```
sitamarhi-health--portal/
├── index.html              # Entry HTML
├── vite.config.ts          # Vite configuration
├── vercel.json             # Vercel deployment config
├── package.json            # Dependencies & scripts
├── public/
│   └── favicon.svg
└── src/
    ├── main.tsx            # App entry point
    ├── App.tsx             # Route definitions
    ├── index.css           # Global styles & Tailwind theme
    ├── components/         # Reusable UI components
    │   ├── KpiCard.tsx
    │   ├── Layout.tsx
    │   └── TopBar.tsx
    ├── context/            # React context providers
    │   └── AuthContext.tsx
    ├── data/               # Mock data & TypeScript interfaces
    │   └── mockData.ts
    └── pages/              # Route pages
        ├── LoginScreen.tsx
        ├── Dashboard.tsx
        ├── AncMonitoring.tsx
        ├── AnaemiaTracker.tsx
        ├── DeliveryMonitoring.tsx
        ├── ZeroBoard.tsx
        └── AiSummary.tsx
```

## 🌐 Routes

| Route | Page | Auth Required |
|---|---|---|
| `/login` | Login Screen | No |
| `/dashboard` | Block Dashboard | Yes |
| `/anc` | ANC Monitoring | Yes |
| `/anaemia` | Anaemia Tracker | Yes |
| `/delivery` | Delivery Monitoring | Yes |
| `/zero-board` | Zero Board | Yes |
| `/ai-summary` | AI Summary | Yes |

## 🎨 Styling

- **Tailwind CSS 4** with custom design tokens
- Google Fonts: **Public Sans** (headings), **Inter** (body)
- **Material Symbols** icon font
- Soft UI design system with rounded cards and shadows
- Fully responsive layout (mobile sidebar overlay, adaptive grids)
- Color-coded severity system (critical/warning/success)

## 🌍 Bilingual Support

- English/Hindi toggle on Login page
- English/Hindi toggle on AI Summary page
- Designed for extension to other pages

## 🚢 Deployment

This project is configured for deployment on **Vercel**:

```bash
npm run build
vercel --prod
```

The `vercel.json` configuration ensures SPA routing works correctly on Vercel.

## 📝 Notes

- This is a **frontend-only demo/prototype** with mock data
- No backend API or database is connected
- Authentication is simulated with hardcoded credentials
- All health data in `src/data/mockData.ts` is sample data for demonstration
- The application works entirely client-side and offline after initial load
- No testing framework is configured

## 📄 License

MIT