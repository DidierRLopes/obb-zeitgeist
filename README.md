# zeitgeist [![Daily Report](https://github.com/pathikrit/zeitgeist/actions/workflows/daily_report.yml/badge.svg)](https://github.com/pathikrit/zeitgeist/actions/workflows/daily_report.yml)

Simple script to go from prediction markets → LLM → Macro report

Now available as **OpenBB Workspace widgets** for integrated financial analysis!

## Today's Report
- <https://pathikrit.github.io/zeitgeist/>

## Features

### 📊 Original Marimo Notebook
Interactive notebook interface for generating market insights reports

### 🏢 OpenBB Workspace Integration
Professional widgets for OpenBB Workspace with:
- **Markdown Widget**: Clean, text-based market insights
- **HTML Widget**: Beautifully styled reports with gradients
- **Run Button Control**: Manual execution to prevent API overuse
- **Investor Profiles**: Tailored analysis for equities, crypto, or commodities
- **Header Authentication**: Secure API key management

## Quick Start

### Option 1: Marimo Notebook (Original Interface)
```shell
git clone git@github.com:pathikrit/zeitgeist.git
cd zeitgeist
uv sync
uv run marimo edit zeitgeist.py
```

### Option 2: OpenBB Workspace Widgets
```shell
# Setup
git clone git@github.com:pathikrit/zeitgeist.git
cd zeitgeist
uv sync

# Configure API keys (see API Configuration below)
echo "OPENAI_API_KEY=your-key-here" >> .env

# Run widget server
uv run python widget_server.py
```

Server runs on `http://localhost:8000`

## API Configuration

### Required
- **OpenAI API Key**: For AI-powered market analysis
  - Get at: https://platform.openai.com/api-keys

### Optional
- **GNews API Key**: For enhanced news data (defaults to free tier)
  - Get at: https://gnews.io (100 requests/day free)

### Setup Methods

**Local Development (.env file):**
```env
OPENAI_API_KEY=your-openai-api-key-here
GNEWS_API_KEY=your-gnews-api-key-here  # Optional
```

**OpenBB Workspace (Headers):**
- `X-OPENAI-API-KEY`: Your OpenAI API key
- `X-GNEWS-API-KEY`: Your GNews API key (optional)

## OpenBB Workspace Widgets

### Available Widgets
1. **Zeitgeist Market Insights** (Markdown)
   - Clean text-based reports
   - Endpoint: `/zeitgeist_report`
   
2. **Zeitgeist HTML Report** (Styled)
   - Beautiful HTML with gradients
   - Endpoint: `/zeitgeist_html`

### Widget Features
- **Manual Execution**: Run button prevents auto-refresh loops
- **Investor Profiles**: 
  - Equities & ETFs (default)
  - Cryptocurrency 
  - Commodities
- **Data Sources**: Kalshi, Polymarket, GNews
- **AI Analysis**: GPT-4 powered insights

### Integration with OpenBB
1. Start widget server: `uv run python widget_server.py`
2. In OpenBB Workspace, add backend: `http://localhost:8000`
3. Use headers for API keys (recommended) or environment variables
4. Find widgets in "Market Analysis" category

## Data Sources
- **Kalshi**: US-focused prediction markets
- **Polymarket**: Decentralized prediction markets  
- **GNews**: Global news headlines
- **AI Analysis**: OpenAI GPT-4 for synthesis

## Files
- `zeitgeist.py`: Original Marimo notebook
- `widget_server.py`: OpenBB Workspace widget server
- `OPENBB_SETUP.md`: Detailed OpenBB integration guide
- `requirements.txt`: Python dependencies for standalone setup

See `OPENBB_SETUP.md` for comprehensive setup and troubleshooting guide.
