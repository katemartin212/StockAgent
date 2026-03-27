# -*- coding: utf-8 -*-
"""
ISOM 260: Build Your First AI Agent
Session 4 — Building AI Products | Suffolk University | Prof. Hasan Arslan

Local version — converted from Google Colab notebook.

Setup:
    pip install -r requirements.txt

    Then set your API key in one of two ways:
      Option A: Create a .env file with:   ANTHROPIC_API_KEY=your-key-here
      Option B: Export in your shell:      export ANTHROPIC_API_KEY=your-key-here

    Get your key at: https://console.anthropic.com
"""

# ============================================
# Part 1: Setup
# ============================================

import os
import json
import anthropic
import yfinance as yf

# Load API key from .env file if present, otherwise fall back to environment variable
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env var works too

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Add it to a .env file or export it in your shell."
    )

client = anthropic.Anthropic(api_key=api_key)

# Verify connection
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say 'Agent ready!' in exactly 2 words."}]
)

print(response.content[0].text)
print("\n✅ Connection successful! Using model: claude-sonnet-4-6")


# ============================================
# Part 3: Tool 1 — Calculator
# ============================================

calculator_tool = {
    "name": "calculator",
    "description": (
        "A precise mathematical calculator. Use this tool whenever you need to "
        "perform arithmetic calculations. It handles addition, subtraction, "
        "multiplication, division, and exponentiation with perfect accuracy. "
        "Always prefer this tool over mental math for any calculation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '2 + 2' or '(100 * 0.15) + 50'"
            }
        },
        "required": ["expression"]
    }
}


def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Only numbers and +-*/.() are allowed."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


print("Testing calculator:")
print(f"  2 + 2 = {calculator('2 + 2')}")
print(f"  7394 * 8261 = {calculator('7394 * 8261')}")
print(f"  (100 * 0.15) + 50 = {calculator('(100 * 0.15) + 50')}")
print("\n✅ Calculator tool is working!")


# ============================================
# The Agent Loop
# ============================================

def run_agent(user_message: str, tools: list, tool_functions: dict, verbose=True):
    """
    Run an AI agent that can use tools to answer questions.

    This is the same core pattern used by NanoClaw, OpenClaw,
    Claude Code, and every AI agent in production today.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🗣️  User: {user_message}")
        print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]
    step = 0

    while True:
        step += 1

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="You are a helpful assistant. Use the available tools whenever they would help you give a more accurate answer. Always show your reasoning.",
            tools=tools,
            messages=messages
        )

        if verbose:
            print(f"\n🔄 Step {step} | Stop reason: {response.stop_reason}")

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    if verbose:
                        snippet = block.text[:150] + "..." if len(block.text) > 150 else block.text
                        print(f"   🧠 Thinking: {snippet}")

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    if verbose:
                        print(f"   🛠️  Calling tool: {tool_name}")
                        print(f"      Input: {json.dumps(tool_input)}")

                    if tool_name in tool_functions:
                        result = tool_functions[tool_name](**tool_input)
                    else:
                        result = f"Error: Unknown tool '{tool_name}'"

                    if verbose:
                        print(f"      Result: {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            final_answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer += block.text

            if verbose:
                print(f"\n🤖 Agent Answer:\n{final_answer}")
                print(f"\n✅ Done in {step} step(s)")

            return final_answer

        if step > 10:
            return "Error: Agent exceeded maximum steps."


print("✅ Agent loop defined! Let's test it.")

# Test the calculator agent
tools = [calculator_tool]
tool_functions = {"calculator": calculator}

run_agent("What is 7,394 multiplied by 8,261?", tools, tool_functions)

run_agent(
    "I'm opening a coffee shop. My monthly rent is $4,500, ingredients cost $2,800, "
    "and staff costs $6,200. If I sell 3,400 cups per month at $5.75 each, "
    "what's my monthly profit? And how many months until I recoup a $45,000 startup investment?",
    tools, tool_functions
)

run_agent("What is the capital of France?", tools, tool_functions)


# ============================================
# Part 4: Multi-Tool Agent
# ============================================

stock_tool = {
    "name": "get_stock_price",
    "description": (
        "Look up the current stock price and key metrics for a publicly traded company. "
        "Provide the stock ticker symbol (e.g., AAPL for Apple, MSFT for Microsoft). "
        "Returns current price, daily change, 52-week high/low, and shares outstanding."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'"
            }
        },
        "required": ["ticker"]
    }
}


def get_stock_price(ticker: str) -> str:
    """Simulated stock data for classroom use."""
    stocks = {
        "AAPL": {"name": "Apple Inc.", "price": 242.58, "change": "+1.23%", "high_52w": 260.10, "low_52w": 164.08, "shares_outstanding": "15.12B"},
        "MSFT": {"name": "Microsoft Corp.", "price": 428.50, "change": "-0.45%", "high_52w": 468.35, "low_52w": 388.46, "shares_outstanding": "7.43B"},
        "GOOGL": {"name": "Alphabet Inc.", "price": 182.30, "change": "+0.87%", "high_52w": 201.42, "low_52w": 150.22, "shares_outstanding": "12.20B"},
        "AMZN": {"name": "Amazon.com Inc.", "price": 215.45, "change": "+2.10%", "high_52w": 242.52, "low_52w": 166.48, "shares_outstanding": "10.52B"},
        "TSLA": {"name": "Tesla Inc.", "price": 342.10, "change": "-3.21%", "high_52w": 488.54, "low_52w": 138.80, "shares_outstanding": "3.21B"},
        "NVDA": {"name": "NVIDIA Corp.", "price": 138.25, "change": "+1.95%", "high_52w": 153.13, "low_52w": 75.61, "shares_outstanding": "24.49B"},
    }
    ticker = ticker.upper().strip()
    if ticker in stocks:
        return json.dumps(stocks[ticker])
    return json.dumps({"error": f"Ticker '{ticker}' not found. Available: {', '.join(stocks.keys())}"})


company_tool = {
    "name": "get_company_info",
    "description": (
        "Get detailed information about a company including its sector, "
        "number of employees, founding year, CEO, and headquarters location. "
        "Use the stock ticker symbol to look up the company."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'AAPL'"
            }
        },
        "required": ["ticker"]
    }
}


def get_company_info(ticker: str) -> str:
    """Simulated company info for classroom use."""
    companies = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology", "employees": 164000, "founded": 1976, "ceo": "Tim Cook", "hq": "Cupertino, CA"},
        "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "employees": 228000, "founded": 1975, "ceo": "Satya Nadella", "hq": "Redmond, WA"},
        "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "employees": 182000, "founded": 1998, "ceo": "Sundar Pichai", "hq": "Mountain View, CA"},
        "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "employees": 1540000, "founded": 1994, "ceo": "Andy Jassy", "hq": "Seattle, WA"},
        "TSLA": {"name": "Tesla Inc.", "sector": "Automotive", "employees": 140000, "founded": 2003, "ceo": "Elon Musk", "hq": "Austin, TX"},
        "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology", "employees": 32000, "founded": 1993, "ceo": "Jensen Huang", "hq": "Santa Clara, CA"},
    }
    ticker = ticker.upper().strip()
    if ticker in companies:
        return json.dumps(companies[ticker])
    return json.dumps({"error": f"Company '{ticker}' not found."})


all_tools = [calculator_tool, stock_tool, company_tool]
all_functions = {
    "calculator": calculator,
    "get_stock_price": get_stock_price,
    "get_company_info": get_company_info,
}

print("✅ 3 tools registered:")
for t in all_tools:
    print(f"   🛠️  {t['name']}")

run_agent(
    "Compare Apple and Microsoft: which company has a higher market cap? "
    "Also tell me which company has more revenue per employee if Apple's "
    "annual revenue is $385 billion and Microsoft's is $245 billion.",
    all_tools, all_functions
)

run_agent(
    "I have $100,000 to invest. If I split it equally between NVIDIA and Tesla, "
    "how many shares of each could I buy at current prices? "
    "Which company is older, and what sectors are they in?",
    all_tools, all_functions
)


# ============================================
# Part 5: Custom Tool — IT Sector Financial Data (live via yfinance)
# ============================================
#
# This tool returns real-time institutional-grade metrics:
# Rule of 40, gross margin, EV/Revenue, P/E, analyst targets.
# No API key required — uses Yahoo Finance.

financial_data_tool = {
    "name": "get_financial_data",
    "description": (
        "Get real-time financial and valuation metrics for stocks. "
        "Returns institutional-grade metrics including revenue growth, gross margin, "
        "Rule of 40 score (revenue growth % + FCF margin % — a key SaaS health metric), "
        "EV/Revenue multiple, P/E ratio, and analyst price targets. "
        "Use this whenever you need to assess a company's financial health, "
        "compare valuations, or evaluate whether a stock price is justified by fundamentals. "
        "Always prefer this over get_stock_price when the question involves "
        "financial health or valuation analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'NVDA', 'MSFT', 'AAPL'"
            }
        },
        "required": ["ticker"]
    }
}


def get_financial_data(ticker: str) -> str:
    """
    Fetches real financial data via yfinance (Yahoo Finance).
    No API key required.
    Rule of 40 = Revenue Growth % + Free Cash Flow Margin %
    A score above 40 is considered healthy for a tech/SaaS company.
    """
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        info = stock.info

        # yfinance may return currentPrice or regularMarketPrice depending on market hours
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not info or price is None:
            return json.dumps({"error": f"Ticker '{ticker}' not found or market data unavailable."})

        # Revenue growth (year-over-year from annual income statement)
        revenue_growth = None
        try:
            financials = stock.financials
            if financials is not None and not financials.empty:
                revenues = financials.loc["Total Revenue"].dropna()
                if len(revenues) >= 2:
                    rev_now = revenues.iloc[0]
                    rev_prior = revenues.iloc[1]
                    revenue_growth = round(((rev_now - rev_prior) / abs(rev_prior)) * 100, 1)
        except Exception:
            pass

        # Gross margin
        gross_margin = round((info.get("grossMargins", 0) or 0) * 100, 1)

        # FCF margin (FCF / Revenue)
        fcf_margin = None
        try:
            cashflow = stock.cashflow
            if cashflow is not None and not cashflow.empty:
                fcf = cashflow.loc["Free Cash Flow"].iloc[0]
                rev = info.get("totalRevenue", 1) or 1
                fcf_margin = round((fcf / rev) * 100, 1)
        except Exception:
            pass

        # Rule of 40
        if revenue_growth is not None and fcf_margin is not None:
            rule_of_40 = round(revenue_growth + fcf_margin, 1)
        else:
            rule_of_40 = "N/A"

        result = {
            "ticker": ticker,
            "name": info.get("longName"),
            "price": price,
            "market_cap": info.get("marketCap"),
            "pe_ratio": round(info.get("trailingPE", 0) or 0, 1),
            "forward_pe": round(info.get("forwardPE", 0) or 0, 1),
            "ev_revenue_multiple": round(info.get("enterpriseToRevenue", 0) or 0, 1),
            "revenue_growth_pct": revenue_growth,
            "gross_margin_pct": gross_margin,
            "fcf_margin_pct": fcf_margin,
            "rule_of_40": rule_of_40,
            "short_interest_pct": round((info.get("shortPercentOfFloat", 0) or 0) * 100, 1),
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey", "N/A").replace("_", " "),
            "data_source": "Yahoo Finance (live)",
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# Quick test
print("Testing live data fetch...")
test = get_financial_data("NVDA")
parsed = json.loads(test)

if "error" in parsed:
    print(f"❌ Error: {parsed['error']}")
else:
    print(f"✅ Live data fetched for {parsed['ticker']} — {parsed['name']}!")
    print(f"   Price          : ${parsed['price']}")
    print(f"   Revenue growth : {parsed['revenue_growth_pct']}%")
    print(f"   Gross margin   : {parsed['gross_margin_pct']}%")
    print(f"   FCF margin     : {parsed['fcf_margin_pct']}%")
    print(f"   Rule of 40     : {parsed['rule_of_40']}")
    print(f"   EV/Revenue     : {parsed['ev_revenue_multiple']}x")
    print(f"   P/E ratio      : {parsed['pe_ratio']}x")
    print(f"   Analyst target : ${parsed['analyst_target']}")
    print(f"   Recommendation : {parsed['analyst_recommendation']}")


# Register all 4 tools
my_tools = [calculator_tool, stock_tool, company_tool, financial_data_tool]
my_functions = {
    "calculator": calculator,
    "get_stock_price": get_stock_price,
    "get_company_info": get_company_info,
    "get_financial_data": get_financial_data,
}

print("\n✅ 4 tools registered:")
for t in my_tools:
    print(f"   🛠️  {t['name']}")


# ============================================================
# Test 1: Valuation justification
# ============================================================

print("\n" + "="*60)
print("BUSINESS QUESTION 1: Valuation justification")
print("="*60)

run_agent(
    "Is NVIDIA's current P/E ratio justified given its revenue growth rate? "
    "Compare it to Microsoft and tell me which is the better value right now.",
    my_tools, my_functions
)


# ============================================================
# Test 2: Portfolio math + fundamentals comparison
# ============================================================

print("\n" + "="*60)
print("BUSINESS QUESTION 2: Best risk-adjusted opportunity")
print("="*60)

run_agent(
    "I have $50,000 to invest equally across Apple, Meta, and Amazon. "
    "How many shares of each can I buy at current prices? "
    "Which of the three has the strongest fundamentals based on "
    "Rule of 40 and gross margin?",
    my_tools, my_functions
)


# ============================================================
# Test 3: Narrative vs. reality
# ============================================================

print("\n" + "="*60)
print("BUSINESS QUESTION 3: Narrative vs. fundamentals check")
print("="*60)

run_agent(
    "Tesla trades at a high P/E multiple. Pull its actual revenue growth "
    "and gross margin. Is the valuation mathematically justified, and how "
    "does it compare to NVIDIA which is also a high-multiple stock? "
    "What does this tell us about the gap between narrative and reality?",
    my_tools, my_functions
)
