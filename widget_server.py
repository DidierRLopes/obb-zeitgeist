import asyncio
import json
import os
import time
from datetime import date
from functools import wraps
from textwrap import dedent

import polars as pl
import requests
from dicttoxml import dicttoxml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from gnews import GNews
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Load environment variables from .env file
load_dotenv()

BATCH_SIZE = 25
RETRIES = 3
DEFAULT_OPENAI_MODEL = "openai:gpt-4.1-2025-04-14"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

MODEL_PROVIDERS = {
    "openai": DEFAULT_OPENAI_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL
}

WIDGETS = {}

# Simple in-memory cache
CACHE = {}
CACHE_DURATION = 3600  # 1 hours in seconds

def get_cached_response(key):
    if key in CACHE:
        data, timestamp = CACHE[key]
        if time.time() - timestamp < CACHE_DURATION:
            return data
        else:
            del CACHE[key]
    return None

def set_cache(key, data):
    CACHE[key] = (data, time.time())

def register_widget(widget_config):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        endpoint = widget_config.get("endpoint")
        if endpoint:
            if "widgetId" not in widget_config:
                widget_config["widgetId"] = endpoint
            widget_id = widget_config["widgetId"]
            WIDGETS[widget_id] = widget_config
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

app = FastAPI(
    title="Zeitgeist OpenBB Widget",
    description="Market insights from prediction markets for OpenBB Workspace",
    version="0.0.1"
)

origins = [
    "https://pro.openbb.co",
    "http://localhost:3000",
    "http://localhost:5050",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Info": "Zeitgeist Market Insights Widget Server"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/widgets.json")
def get_widgets():
    return WIDGETS

@app.get("/apps.json")
def get_apps():
    try:
        with open("apps.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def fetch_from_kalshi() -> pl.DataFrame:
    API_URL = "https://api.elections.kalshi.com/trade-api/v2"
    params = {"status": "open", "with_nested_markets": "true", "limit": 100, "cursor": None}
    predictions = []
    MAX_PREDICTIONS = 500  # Limit total predictions to avoid infinite loops
    
    def simple_prediction(e):
        bets = []
        for m in e.get("markets", []):
            if m.get("notional_value", 0) > 0:  # Avoid division by zero
                bets.append({"prompt": m.get("yes_sub_title", ""), "probability": m.get("last_price", 0) / m["notional_value"]})
        return {"id": f"kalshi-{e.get('event_ticker', '')}", "title": e.get("title", ""), "bets": bets}
    
    while len(predictions) < MAX_PREDICTIONS:
        print(f"Fetching from kalshi @ offset={len(predictions)} ...")
        try:
            resp = requests.get(f"{API_URL}/events", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events", [])
            if not events:  # No more data
                break
            predictions.extend(events)
            params["cursor"] = data.get("cursor")
            if not params["cursor"]:  # No more pages
                break
        except Exception as e:
            print(f"Error fetching from Kalshi: {e}")
            break
    
    print(f"Fetched {len(predictions)} from kalshi")
    return pl.DataFrame([simple_prediction(p) for p in predictions[:MAX_PREDICTIONS]])

def fetch_from_polymarket() -> pl.DataFrame:
    API_URL = "https://gamma-api.polymarket.com"
    predictions = []
    MAX_PREDICTIONS = 500  # Limit total predictions
    
    def simple_prediction(p):
        bets = []
        try:
            outcomes = json.loads(p.get("outcomes", "[]"))
            prices = json.loads(p.get("outcomePrices", "[]"))
            for prompt, probability in zip(outcomes, prices):
                bets.append({"prompt": prompt, "probability": float(probability)})
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing prediction data: {e}")
        return {"id": f"pm-{p.get('id', '')}", "title": p.get("question", ""), "bets": bets}
    
    while len(predictions) < MAX_PREDICTIONS:
        params = {"active": "true", "closed": "false", "limit": 100, "offset": len(predictions)}
        print(f"Fetching from polymarket @ offset={params['offset']} ...")
        try:
            resp = requests.get(f"{API_URL}/markets", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:  # No more data
                break
            predictions.extend(data)
        except Exception as e:
            print(f"Stopping because of error from Polymarket: {e}")
            break
    
    print(f"Fetched {len(predictions)} from polymarket")
    return pl.DataFrame([simple_prediction(p) for p in predictions[:MAX_PREDICTIONS]])

async def generate_report(investor_type: str = "equities", model_provider: str = "gemini", gnews_api_key: str = None) -> str:
    # Set up GNews API key in environment before importing
    original_gnews_key = None
    if gnews_api_key:
        original_gnews_key = os.environ.get("GNEWS_API_KEY")
        os.environ["GNEWS_API_KEY"] = gnews_api_key
    
    try:
        kalshi_predictions = fetch_from_kalshi()
        polymarket_predictions = fetch_from_polymarket()
        predictions = pl.concat([kalshi_predictions, polymarket_predictions])
        
        MAX_PREDICTIONS_TO_ANALYZE = 50
        if len(predictions) > MAX_PREDICTIONS_TO_ANALYZE:
            predictions = predictions.head(MAX_PREDICTIONS_TO_ANALYZE)
        
        gnews = GNews()
        news = gnews.get_top_news()
        
        investor_profiles = {
            "equities": dedent("""
                I am an American equities investor and I am interested in topics
                that would impact the market in the relatively short term or could change how I invest.
                Besides publicly listed equities, I can have exposure to broad indices (e.g. $SPY and $QQQ)
                sectors (e.g. defense: $XAR, healthcare: $XLV) and alternatives
                like gold, energy, commodities, crypto, bonds, TIPS, REITs, mortgage-backed securities etc
                through ETFs/vehicles like $IAU, $DBC, $BTC, $ZROZ, $TIPZ, $VNQ etc
                so pay particular attention to macroeconomic themes.
            """),
            "crypto": dedent("""
                I am a cryptocurrency investor focused on digital assets and blockchain technologies.
                I invest in major cryptocurrencies like Bitcoin, Ethereum, Solana, and emerging DeFi protocols.
                I'm interested in regulatory developments, institutional adoption, DeFi innovations,
                and macroeconomic factors that affect crypto markets.
            """),
            "commodities": dedent("""
                I am a commodities trader focused on physical assets and futures markets.
                I trade energy (oil, natural gas), precious metals (gold, silver), 
                agriculture (wheat, corn, soybeans), and industrial metals (copper, aluminum).
                I'm interested in supply/demand dynamics, weather patterns, geopolitical events,
                and currency movements that affect commodity prices.
            """)
        }
        
        about_me = f"""<about_me>
        {investor_profiles.get(investor_type, investor_profiles["equities"])}
        Some examples of things that are LIKELY to impact my investments:
          - Short term macroeconomic indicators like GDP, unemployment, CPI, trade deficit etc.
          - Public or private companies suing each other or M&A activities
          - Foreign politics that would affect USD rates with major international currencies like JPY,CNY,EUR etc
          - EV/climate legislation and goals in short term (<5 years)
          - US policies and outlook on debt, budget, tax laws, tariffs, healthcare, energy
          - General major geopolitical events that can happen near future (<5 years)
          - Specific public companies mentioned like Tesla, Apple, Nvidia etc
          - Major natural disasters, pandemics or crisis with high (>50%) probabilities
        FYI: today's date is {date.today().strftime('%d-%b-%Y')}
        General instructions:
        - Think deeply about second or third order effects
        - Don't restrict yourself or fixate on only the tickers or themes mentioned above
          since these are just examples I used to give you a general idea of how I can invest
        </about_me>"""
        
        class RelevantPrediction(BaseModel):
            id: str = Field(description="original id from input")
            topics: list[str] = Field(description="public companies or investment sectors or broad alternatives impacted")

        # Select the model based on provider
        selected_model = MODEL_PROVIDERS.get(model_provider, DEFAULT_GEMINI_MODEL)

        relevant_prediction_agent = Agent(
            model=selected_model,
            output_type=list[RelevantPrediction],
            system_prompt=(
                "<task>"
                "You will be provided an array of questions from an online betting market"
                "Your job is to return only the ids of questions relevant to me"
                "</task>"
                f"{about_me}"
                "Some examples of things that are UNLIKELY to impact (unless a good reason is provided):"
                "  - Celebrity gossips e.g. how many tweets would Elon tweet this month"
                "  - Sports related e.g. Would Ronaldo be traded this season"
                "  - Events far (10+ years) in the future: Would India host the Olympics by 2040"
                "  - Geography e.g. election results in Kiribati is unlikely to impact my investments"
                "    but major economies like Chinese, India, EU, MEA politics is likely to impact"
                "  - Media e.g. what song will be in top billboard this week"
                "  - Ignore memecoins and NFTs (but focus on major crypto themes like BTC, solana and ethereum etc)"
                "  - Ignore essentially gambling bets on short term prices e.g. what will be USD/JPY today at 12pm"
                "Examine each question and return a subset of ids and related topics they may impact"
                "Topics be few must be short strings like sectors or tickers"
                "or short phrases that would be impacted by this question"
                "Generally be lenient when possible to decide whether to include an id or not"
            ),
            retries=RETRIES,
        )
        
        async def tag_predictions(predictions: pl.DataFrame) -> pl.DataFrame:
            dfs = []
            for i, batch in enumerate(predictions.iter_slices(BATCH_SIZE)):
                try:
                    result = await relevant_prediction_agent.run(batch.write_json())
                    if result.output:
                        dfs.append(pl.DataFrame(result.output))
                except Exception as e:
                    print(f"Error processing batch {i+1}: {e}")
            
            if dfs:
                relevant_predictions = pl.concat(dfs)
                return predictions.join(relevant_predictions, on="id", how="left")
            return predictions
        
        tagged_predictions = await tag_predictions(predictions)
        
        def to_xml_str(input: dict) -> str:
            return dicttoxml(input, xml_declaration=False, root=False, attr_type=False, return_bytes=False)
        
        synthesizing_agent = Agent(
            model=selected_model,
            output_type=str,
            system_prompt=(
                f"{about_me}"
                "<task>"
                "You will be provided an array of questions and probabilities from an online betting market"
                "along with today's top news headlines"
                "Consolidate and summarize into a 1-pager investment guideline thesis report"
                "The provided topics column can serve as hints to explore but think deeply about 2nd and 3rd order effects"
                "Take into account the probabilities and the fact that the topic is being discussed in the first place"
                "but also keep in mind that prediction markets often have moonshot bias i.e."
                "people sometime tend to overweight extreme low-probability outcomes and underweight high-probability ones"
                "due to the non-linear probability weighting function in their model"
                "</task>"
                "<output_format>"
                "Present in a markdown format with sections and sub-sections"
                "Go from broad (e.g. macro) to narrow (e.g. sector) and finally individual names as top-level sections"
                "Also add and consolidate any important or relevant news items"
                "in simple bullets at the top in a separate news section"
                "This is intended to be consumed daily as a news memo"
                "So just use the title: Daily Memo (date)"
                "Things to avoid:"
                "  - Don't mention that your input was prediction markets; the reader is aware of that"
                "  - Avoid putting the exact probabilities from the input; just use plain English to describe the prospects"
                "  - Avoid general guidelines like 'review this quarterly'"
                "  - Unless it pertains to an individual company or currency"
                "  - avoid mentioning broad ETF tickers as I can figure that out from the sector or bond duration etc"
                "</output_format>"
            ),
            retries=RETRIES,
        )
        
        report_input = {
            "prediction_markets": tagged_predictions.select("title", "bets", "topics")
            .filter(pl.col("topics").is_not_null())
            .to_dicts(),
            "news_headlines": pl.DataFrame(news).select("title", "description").to_dicts() if news else [],
        }
        
        report = await synthesizing_agent.run(to_xml_str(report_input))
        return report.output
    except Exception as e:
        print(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if gnews_api_key:
            if original_gnews_key is not None:
                os.environ["GNEWS_API_KEY"] = original_gnews_key
            elif "GNEWS_API_KEY" in os.environ:
                del os.environ["GNEWS_API_KEY"]

@register_widget({
    "name": "Zeitgeist Market Insights",
    "description": "Daily investment memo based on prediction markets and news",
    "category": "Market Analysis",
    "type": "markdown",
    "endpoint": "zeitgeist_report",
    "gridData": {"w": 20, "h": 30},
    "source": "Kalshi & Polymarket",
    "runButton": True,
    "params": [
        {
            "paramName": "investor_type",
            "value": "equities",
            "label": "Investor Type",
            "type": "text",
            "description": "Select your investment focus",
            "options": [
                {"label": "Equities & ETFs", "value": "equities"},
                {"label": "Cryptocurrency", "value": "crypto"},
                {"label": "Commodities", "value": "commodities"}
            ]
        },
        {
            "paramName": "model_provider",
            "value": "gemini",
            "label": "AI Model",
            "type": "text",
            "description": "Select AI model provider",
            "options": [
                {"label": "Google Gemini 2.5 Pro", "value": "gemini"},
                {"label": "OpenAI GPT-4.1", "value": "openai"}
            ]
        }
    ]
})
@app.get("/zeitgeist_report")
async def zeitgeist_report(request: Request, response: Response, investor_type: str = "equities", model_provider: str = "gemini"):
    """Generate Zeitgeist market insights report"""
    cache_key = f"zeitgeist_report_{investor_type}_{model_provider}"

    # Check cache first
    cached_data = get_cached_response(cache_key)
    if cached_data:
        response.headers["Cache-Control"] = "public, max-age=7200"  # 2 hours
        return cached_data

    # Handle API keys based on model provider
    if model_provider == "openai":
        api_key = request.headers.get('X-OPENAI-API-KEY')
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="OpenAI API key required. Set OPENAI_API_KEY environment variable or add 'X-OPENAI-API-KEY' header when connecting backend to OpenBB Workspace."
            )

        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key
    else:  # gemini
        api_key = request.headers.get('X-GEMINI-API-KEY')
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Gemini API key required. Set GEMINI_API_KEY environment variable or add 'X-GEMINI-API-KEY' header when connecting backend to OpenBB Workspace."
            )

        original_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = api_key

    try:
        gnews_api_key = request.headers.get('X-GNEWS-API-KEY')
        if not gnews_api_key:
            gnews_api_key = os.environ.get("GNEWS_API_KEY")

        report = await generate_report(investor_type, model_provider, gnews_api_key)
        
        # Cache the response
        set_cache(cache_key, report)
        response.headers["Cache-Control"] = "public, max-age=7200"  # 2 hours
        
        return report
    finally:
        if model_provider == "openai":
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
        else:  # gemini
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

@register_widget({
    "name": "Zeitgeist HTML Report",
    "description": "Styled investment memo with market insights",
    "category": "Market Analysis",
    "type": "html",
    "endpoint": "zeitgeist_html",
    "gridData": {"w": 25, "h": 35},
    "source": "Kalshi & Polymarket",
    "runButton": True,
    "params": [
        {
            "paramName": "investor_type",
            "value": "equities",
            "label": "Investor Type",
            "type": "text",
            "description": "Select your investment focus",
            "options": [
                {"label": "Equities & ETFs", "value": "equities"},
                {"label": "Cryptocurrency", "value": "crypto"},
                {"label": "Commodities", "value": "commodities"}
            ]
        },
        {
            "paramName": "model_provider",
            "value": "gemini",
            "label": "AI Model",
            "type": "text",
            "description": "Select AI model provider",
            "options": [
                {"label": "Google Gemini 2.5 Pro", "value": "gemini"},
                {"label": "OpenAI GPT-4.1", "value": "openai"}
            ]
        }
    ]
})
@app.get("/zeitgeist_html", response_class=HTMLResponse)
async def zeitgeist_html(request: Request, response: Response, investor_type: str = "equities", model_provider: str = "gemini"):
    """Generate styled HTML version of Zeitgeist report"""
    cache_key = f"zeitgeist_html_{investor_type}_{model_provider}"

    # Check cache first
    cached_data = get_cached_response(cache_key)
    if cached_data:
        response.headers["Cache-Control"] = "public, max-age=7200"  # 2 hours
        return HTMLResponse(content=cached_data)

    # Handle API keys based on model provider
    if model_provider == "openai":
        api_key = request.headers.get('X-OPENAI-API-KEY')
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="OpenAI API key required. Set OPENAI_API_KEY environment variable or add 'X-OPENAI-API-KEY' header when connecting backend to OpenBB Workspace."
            )

        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key
    else:  # gemini
        api_key = request.headers.get('X-GEMINI-API-KEY')
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Gemini API key required. Set GEMINI_API_KEY environment variable or add 'X-GEMINI-API-KEY' header when connecting backend to OpenBB Workspace."
            )

        original_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = api_key

    try:
        gnews_api_key = request.headers.get('X-GNEWS-API-KEY')
        if not gnews_api_key:
            gnews_api_key = os.environ.get("GNEWS_API_KEY")

        markdown_report = await generate_report(investor_type, model_provider, gnews_api_key)
        html_content = MarkdownIt().render(markdown_report)
    except Exception as e:
        html_content = f"<h1>Error Generating Report</h1>\n<p>Error: {str(e)}</p>"
    finally:
        if model_provider == "openai":
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
        else:  # gemini
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
    
    today = date.today()
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Zeitgeist Report {today.strftime("%d-%b-%Y")}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #2d3748;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #4a5568;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #764ba2;
            padding-left: 10px;
        }}
        h3 {{
            color: #718096;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        p {{
            margin-bottom: 15px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        .source {{
            font-size: 0.9em;
            color: #718096;
            font-style: italic;
        }}
        .timestamp {{
            font-size: 0.85em;
            color: #a0aec0;
            text-align: right;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
        }}
        strong {{
            color: #2d3748;
        }}
        code {{
            background: #f7fafc;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
            color: #e53e3e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="source">Source: Kalshi & Polymarket Prediction Markets</div>
                <div class="source">Investor Profile: {investor_type.title()}</div>
            </div>
        </div>
        {html_content}
        <div class="timestamp">
            Generated: {today.strftime("%B %d, %Y at %H:%M")}
        </div>
    </div>
</body>
</html>"""
    
    # Cache the response
    set_cache(cache_key, html)
    response.headers["Cache-Control"] = "public, max-age=7200"  # 2 hours
    
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)