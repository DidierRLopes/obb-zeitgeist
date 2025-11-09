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
5. Click Add Authentication
6. Add OpenAI API key
   - **Key**: X-OPENAI-API-KEY
   - **Valye**: Get from https://platform.openai.com/api-keys
   - **Location**: Header
7. Add GNEWS API key
   - **Key**: X-GNEWS-API-KEY
   - **Valye**: Get at: https://gnews.io (100 requests/day free)
   - **Location**: Header
5. Click the **Test** button to verify the connection
6. If the test is successful, click the **Add** button

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

**Local Development (.env file):**
```env
OPENAI_API_KEY=your-openai-api-key-here
GNEWS_API_KEY=your-gnews-api-key-here
```
