# AlphaDesign - Web Interface

**Interactive Frontend for F1 Wing Optimization Visualization**

## Overview

This directory contains the web-based user interface for AlphaDesign, built with React, TypeScript, and modern web technologies. The interface provides real-time visualization of the optimization process, 3D wing model viewing, and comprehensive analysis dashboards.

## Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and building
- **UI Components**: Radix UI primitives with custom styling
- **Styling**: Tailwind CSS with custom design system
- **3D Visualization**: Three.js with React Three Fiber
- **Charts**: Recharts for performance analytics
- **State Management**: React Query for server state
- **Routing**: React Router for navigation

## Quick Start

### Prerequisites

- **Node.js 18+** with npm or bun
- **Modern browser** with WebGL support

### Installation

```bash
# Navigate to website directory
cd website

# Install dependencies (using bun - faster alternative to npm)
bun install

# Or with npm
npm install
```

### Development

```bash
# Start development server
bun run dev
# Or: npm run dev

# Open browser to http://localhost:5173
```

### Building for Production

```bash
# Build optimized production bundle
bun run build
# Or: npm run build

# Preview production build locally
bun run preview
# Or: npm run preview
```

## Features

### 🎯 Optimization Dashboard
- **Real-time Progress**: Live updates during optimization runs
- **Performance Metrics**: Downforce, drag, and efficiency charts
- **Generation History**: Evolution of designs over time
- **Best Designs Gallery**: Top-performing wing configurations

### 🎨 3D Visualization
- **Interactive Wing Models**: Rotate, zoom, and inspect designs
- **STL File Viewer**: Direct 3D model loading and rendering
- **Comparison Mode**: Side-by-side design comparison
- **Performance Heatmaps**: Visualize airflow and pressure

### 📊 Analytics Interface
- **Fitness Evolution**: Track optimization convergence
- **Parameter Analysis**: Understand design parameter impacts
- **Statistical Reports**: Comprehensive performance summaries
- **Export Capabilities**: Download results and visualizations

### 🎛️ Control Panel
- **Optimization Settings**: Configure algorithm parameters
- **Real-time Controls**: Start, pause, and stop optimization
- **Configuration Management**: Save and load optimization presets
- **System Monitoring**: Resource usage and performance metrics

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # Base UI primitives (buttons, inputs, etc.)
│   ├── charts/         # Chart components for analytics
│   ├── 3d/             # Three.js 3D visualization components
│   └── dashboard/      # Dashboard-specific components
├── pages/              # Main application pages
│   ├── Dashboard.tsx   # Optimization dashboard
│   ├── Visualizer.tsx  # 3D model viewer
│   ├── Analytics.tsx   # Performance analysis
│   └── Settings.tsx    # Configuration panel
├── hooks/              # Custom React hooks
├── lib/                # Utility functions and configurations
├── data/               # Static data and mock data
└── assets/             # Images, icons, and static assets
```

## Component Architecture

### Dashboard Components
- **OptimizationProgress**: Real-time progress indicators
- **PerformanceCharts**: Interactive performance visualizations
- **DesignGallery**: Grid of generated wing designs
- **ControlPanel**: Optimization control interface

### 3D Visualization
- **WingViewer**: Main 3D model display component
- **STLLoader**: Load and render STL files
- **ComparisonView**: Side-by-side model comparison
- **CameraControls**: Interactive camera manipulation

### Analytics Dashboard
- **FitnessChart**: Optimization convergence visualization
- **ParameterHeatmap**: Design parameter correlation matrix
- **PerformanceMetrics**: Key performance indicators
- **ExportTools**: Data export and sharing utilities

## Configuration

### Environment Variables

Create `.env.local` for local development:

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8001

# Feature Flags
VITE_ENABLE_3D_VISUALIZATION=true
VITE_ENABLE_REAL_TIME_UPDATES=true

# Development
VITE_DEBUG_MODE=true
```

### Tailwind Configuration

Custom design system in `tailwind.config.ts`:

```typescript
// Custom colors for F1 theme
colors: {
  primary: {
    50: '#f0f9ff',
    500: '#3b82f6',
    900: '#1e3a8a'
  },
  accent: {
    500: '#ef4444',  // F1 red
    600: '#dc2626'
  }
}
```

## API Integration

The frontend communicates with the RL optimization backend:

### REST Endpoints
- `GET /api/optimization/status` - Current optimization state
- `POST /api/optimization/start` - Start new optimization run
- `GET /api/designs/latest` - Get latest generated designs
- `GET /api/analytics/performance` - Performance metrics

### WebSocket Connection
- **Real-time Updates**: Live optimization progress
- **Design Notifications**: New design generation alerts
- **System Status**: Backend health and resource usage

## Development Guidelines

### Code Style
- **TypeScript**: Strict type checking enabled
- **ESLint**: Code quality and consistency
- **Prettier**: Automated code formatting
- **Component Organization**: One component per file

### Testing
```bash
# Run linting
bun run lint
# Or: npm run lint

# Type checking
tsc --noEmit
```

### Performance Optimization
- **Code Splitting**: Lazy load heavy components
- **Image Optimization**: Compressed assets and WebP format
- **Bundle Analysis**: Regular bundle size monitoring
- **3D Optimization**: Efficient Three.js rendering

## Deployment

### Vercel (Recommended)

The project is configured for Vercel deployment:

```bash
# Deploy to Vercel
vercel --prod
```

### Static Hosting

For other static hosts:

```bash
# Build static files
bun run build

# Deploy dist/ folder to your hosting provider
```

## Browser Support

- **Chrome 90+** (recommended for best performance)
- **Firefox 88+**
- **Safari 14+**
- **Edge 90+**

**WebGL Required**: 3D visualization requires WebGL-capable browser

## Contributing

### Development Workflow
1. Create feature branch from `main`
2. Implement changes with TypeScript types
3. Test in multiple browsers
4. Ensure responsive design works
5. Submit pull request with screenshots

### UI/UX Guidelines
- **Responsive Design**: Mobile-first approach
- **Accessibility**: WCAG 2.1 AA compliance
- **Performance**: < 3s initial load time
- **F1 Theming**: Red and blue color scheme

## Troubleshooting

### Common Issues

**3D Models Not Loading**
- Check WebGL support: `chrome://gpu/`
- Verify STL file format and size
- Clear browser cache and reload

**Real-time Updates Not Working**
- Check WebSocket connection in DevTools
- Verify backend API is running
- Check firewall/proxy settings

**Build Errors**
- Clear `node_modules` and reinstall
- Check Node.js version compatibility
- Verify all dependencies are installed

## Support

For frontend-specific issues:
1. Check browser console for errors
2. Verify API connectivity
3. Test in different browsers
4. Open GitHub issue with browser and OS details
