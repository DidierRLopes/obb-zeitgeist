# OpenBB Zeitgeist

From prediction markets to Macro report using AI. Based on https://github.com/pathikrit/zeitgeist.

<img width="800" alt="CleanShot 2025-11-09 at 09 46 36@2x" src="https://github.com/user-attachments/assets/1b3996a2-627d-4550-b717-eceb913049c7" />

## Connecting to OpenBB Workspace

<img width="800" alt="CleanShot 2025-11-09 at 09 50 37@2x" src="https://github.com/user-attachments/assets/ace1cab4-ba7c-4970-8bf0-13e69743c22c" />

Follow these steps to connect this backend as a data source in OpenBB Pro:

1. Log in to your OpenBB Pro account at [pro.openbb.co](https://pro.openbb.co)
2. Navigate to the **Apps** page
3. Click the **Connect backend** button
4. Fill in the following details:
   - **Name**: Zeitgeist
   - **URL**: `https://obb-zeitgeist.fly.dev/`
5. Click Add Authentication and add your API keys (you only need keys for the AI model you plan to use):

   **For Google Gemini (recommended, default):**
   - **Key**: X-GEMINI-API-KEY
   - **Value**: Get from https://aistudio.google.com/apikey
   - **Location**: Header

   **For OpenAI (optional):**
   - **Key**: X-OPENAI-API-KEY
   - **Value**: Get from https://platform.openai.com/api-keys
   - **Location**: Header

   **For News (optional but recommended):**
   - **Key**: X-GNEWS-API-KEY
   - **Value**: Get at https://gnews.io (100 requests/day free)
   - **Location**: Header

6. Click the **Test** button to verify the connection
7. If the test is successful, click the **Add** button

Once added, you'll find Zeitgest app available in the Apps section of OpenBB Workspace.

<img width="800" alt="CleanShot 2025-11-09 at 09 54 23@2x" src="https://github.com/user-attachments/assets/0ac88304-a9c7-490a-8440-26f2646b5d7e" />

### Available Widgets

1. **Zeitgeist Market Insights** (Markdown)
   - Clean text-based reports - perfect for OpenBB Copilot
   - Endpoint: `/zeitgeist_report`
   
2. **Zeitgeist HTML Report** (Styled)
   - Beautiful HTML with gradients
   - Endpoint: `/zeitgeist_html`

## Local Setup

### Prerequisites

1. Install [uv](https://github.com/astral-sh/uv) (Python package manager)
2. Python 3.12 or higher

### Environment Variables

Create a `.env` file in the project root:

```env
# Choose one AI provider (Gemini is default)
GEMINI_API_KEY=your-gemini-api-key-here
# OR
OPENAI_API_KEY=your-openai-api-key-here

# Optional: News API
GNEWS_API_KEY=your-gnews-api-key-here
```

The app uses **Google Gemini 2.5 Pro** by default. You can switch to OpenAI GPT-4.1 using the "AI Model" dropdown in the widget.

### Running Locally

1. Clone the repository:
```bash
git clone https://github.com/DidierRLopes/obb-zeitgeist.git
cd obb-zeitgeist
```

2. Install dependencies:
```bash
uv sync
```

3. Run the server:
```bash
uv run uvicorn widget_server:app --reload --port 8000
```

You can still follow the steps above to integrate with OpenBB Workspace, but the URL should now be http://127.0.0.1:8000
