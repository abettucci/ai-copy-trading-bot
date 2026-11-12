"""
AI Copy Trading Lambda Handler

Paper trading engine powered by GPT-4o.
- Generates a diversified stock portfolio using an LLM
- Tracks P&L in real time via yfinance
- Compares performance against S&P 500 benchmark
- Stores portfolio state in DynamoDB
- Sends formatted updates to Telegram

Commands via telegram_bot:
  /test_copytrading  — If no portfolio: generate one. If exists: show status + P&L.
  /rebalancear       — Ask the AI to rebalance the portfolio.
"""

import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import get_config
from shared.db import DynamoDBClient
from shared.telegram import TelegramClient
from shared.utils import yahoo_finance_price, create_response, log_lambda_event, LambdaLogger

import requests

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CAPITAL = 100_000.0  # $100k paper trading


# ---------------------------------------------------------------------------
# Market data helpers (direct Yahoo Finance API — no yfinance library)
# ---------------------------------------------------------------------------

def fetch_price(ticker: str) -> float:
    data = yahoo_finance_price(ticker)
    return data.get('price', 0.0)


def fetch_prices(tickers: List[str]) -> Dict[str, float]:
    return {t: fetch_price(t) for t in tickers}


def get_market_context() -> str:
    indices = {'^GSPC': 'S&P 500', '^NDX': 'Nasdaq 100', '^VIX': 'VIX'}
    lines = []
    for symbol, name in indices.items():
        data = yahoo_finance_price(symbol)
        if data.get('price'):
            lines.append(f"{name}: {data['price']:.0f} ({data['change_pct']:+.1f}%)")
    return "\n".join(lines) if lines else "Market context unavailable"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_openai(prompt: str, api_key: str) -> Optional[str]:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un portfolio manager experto. "
                        "Responde UNICAMENTE con JSON valido, sin texto adicional, "
                        "sin bloques de codigo markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return None


def build_prompt(capital: float, current_portfolio: str = "") -> str:
    market_context = get_market_context()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return f"""Hoy es {today}. Eres un portfolio manager con ${capital:,.0f} USD de capital ficticio (paper trading).

Contexto de mercado actual:
{market_context}
{current_portfolio}

Crea un portfolio diversificado de acciones del mercado estadounidense para mantener durante el proximo mes.
- Selecciona entre 6 y 10 empresas solidas con buenas perspectivas.
- Reserva entre 3% y 6% en cash.
- Cada accion debe tener una tesis de inversion clara.

Responde UNICAMENTE con este JSON (sin texto adicional ni bloques de codigo):
{{
  "market_thesis": "Tu analisis del mercado actual en 1-2 oraciones",
  "cash_pct": 4.0,
  "allocations": [
    {{
      "ticker": "AAPL",
      "allocation_pct": 15.0,
      "reasoning": "Razon para elegir esta accion en 1 oracion"
    }}
  ]
}}

Regla: allocation_pct de todas las posiciones + cash_pct deben sumar exactamente 100.
"""


# ---------------------------------------------------------------------------
# Portfolio persistence (DynamoDB)
# ---------------------------------------------------------------------------

def load_portfolio(db: DynamoDBClient) -> Optional[dict]:
    state = db.get_state('COPY_TRADING', 'portfolio')
    if state and 'data' in state:
        return state['data']
    return None


def save_portfolio(db: DynamoDBClient, portfolio: dict):
    db.save_state(
        ticker='COPY_TRADING',
        data_type='portfolio',
        data=portfolio,
        ttl_hours=720,  # 30 days
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def generate_portfolio(db: DynamoDBClient, api_key: str, capital: float = DEFAULT_CAPITAL) -> Optional[dict]:
    """Ask GPT-4o to build a new portfolio."""
    prompt = build_prompt(capital)
    raw = call_openai(prompt, api_key)
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw: {raw}")
        return None

    try:
        allocations = data['allocations']
        cash_pct = float(data.get('cash_pct', 4.0))
        market_thesis = data.get('market_thesis', '')

        tickers = [a['ticker'] for a in allocations]
        prices = fetch_prices(tickers)
        sp500_price = fetch_price('^GSPC')

        positions = []
        for alloc in allocations:
            ticker = alloc['ticker']
            price = prices.get(ticker, 0.0)
            if price == 0.0:
                continue
            alloc_pct = float(alloc['allocation_pct'])
            alloc_value = capital * alloc_pct / 100.0
            shares = alloc_value / price
            positions.append({
                'ticker': ticker,
                'allocation_pct': alloc_pct,
                'shares': shares,
                'entry_price': price,
                'current_price': price,
                'value': alloc_value,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'reasoning': alloc.get('reasoning', ''),
            })

        now = datetime.utcnow().isoformat()
        portfolio = {
            'capital': capital,
            'cash': capital * cash_pct / 100.0,
            'positions': positions,
            'created_at': now,
            'last_updated': now,
            'last_rebalanced': now,
            'ai_model': 'gpt-4o',
            'market_thesis': market_thesis,
            'benchmark_start': sp500_price,
            'benchmark_current': sp500_price,
        }
        save_portfolio(db, portfolio)
        return portfolio

    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Error building portfolio: {e}")
        return None


def update_prices(db: DynamoDBClient, portfolio: dict) -> dict:
    """Refresh prices and recalculate P&L."""
    tickers = [p['ticker'] for p in portfolio['positions']]
    prices = fetch_prices(tickers)
    sp500_price = fetch_price('^GSPC')

    for pos in portfolio['positions']:
        current = prices.get(pos['ticker'], pos['current_price'])
        pos['current_price'] = current
        pos['value'] = pos['shares'] * current
        pos['pnl'] = pos['value'] - (pos['shares'] * pos['entry_price'])
        pos['pnl_pct'] = ((current - pos['entry_price']) / pos['entry_price'] * 100) if pos['entry_price'] else 0

    portfolio['benchmark_current'] = sp500_price
    portfolio['last_updated'] = datetime.utcnow().isoformat()
    save_portfolio(db, portfolio)
    return portfolio


def rebalance_portfolio(db: DynamoDBClient, api_key: str, portfolio: dict) -> Optional[dict]:
    """Ask GPT-4o to rebalance the existing portfolio."""
    current_summary = "\nPortfolio actual:\n" + "\n".join(
        f"- {p['ticker']}: {p['allocation_pct']:.0f}% (P&L: {p['pnl_pct']:+.1f}%)"
        for p in portfolio['positions']
    )
    prompt = build_prompt(portfolio['capital'], current_summary)
    raw = call_openai(prompt, api_key)
    if not raw:
        return None

    try:
        data = json.loads(raw)
        allocations = data['allocations']
        cash_pct = float(data.get('cash_pct', 4.0))
        market_thesis = data.get('market_thesis', '')

        tickers = [a['ticker'] for a in allocations]
        prices = fetch_prices(tickers)
        sp500_price = fetch_price('^GSPC')

        positions = []
        for alloc in allocations:
            ticker = alloc['ticker']
            price = prices.get(ticker, 0.0)
            if price == 0.0:
                continue
            alloc_pct = float(alloc['allocation_pct'])
            alloc_value = portfolio['capital'] * alloc_pct / 100.0
            shares = alloc_value / price
            positions.append({
                'ticker': ticker,
                'allocation_pct': alloc_pct,
                'shares': shares,
                'entry_price': price,
                'current_price': price,
                'value': alloc_value,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'reasoning': alloc.get('reasoning', ''),
            })

        now = datetime.utcnow().isoformat()
        portfolio['positions'] = positions
        portfolio['cash'] = portfolio['capital'] * cash_pct / 100.0
        portfolio['market_thesis'] = market_thesis
        portfolio['last_updated'] = now
        portfolio['last_rebalanced'] = now
        portfolio['benchmark_current'] = sp500_price
        save_portfolio(db, portfolio)
        return portfolio

    except Exception as e:
        logger.error(f"Error rebalancing: {e}")
        return None


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------

def format_portfolio_message(portfolio: dict) -> str:
    capital = portfolio['capital']
    total_value = sum(p['value'] for p in portfolio['positions']) + portfolio['cash']
    total_pnl = total_value - capital
    total_pnl_pct = (total_pnl / capital * 100) if capital else 0

    bench_start = portfolio.get('benchmark_start', 0)
    bench_current = portfolio.get('benchmark_current', 0)
    bench_pnl_pct = ((bench_current - bench_start) / bench_start * 100) if bench_start else 0
    vs_benchmark = total_pnl_pct - bench_pnl_pct

    pnl_emoji = "+" if total_pnl >= 0 else ""
    vs_emoji = "+" if vs_benchmark >= 0 else ""

    msg = "*AI Copy Trading - Paper Portfolio*\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"Capital: ${capital:,.0f}\n"
    msg += f"Valor total: *${total_value:,.0f}*\n"
    msg += f"P&L: *{pnl_emoji}${total_pnl:,.0f}* ({pnl_emoji}{total_pnl_pct:.1f}%)\n"
    msg += f"vs S&P 500: *{vs_emoji}{vs_benchmark:.1f}%* (S&P: {bench_pnl_pct:+.1f}%)\n"
    msg += f"\n_{portfolio.get('market_thesis', '')}_\n\n"

    msg += "*Posiciones:*\n"
    sorted_pos = sorted(portfolio['positions'], key=lambda p: p['value'], reverse=True)
    for p in sorted_pos:
        dot = "+" if p['pnl'] >= 0 else "-"
        sign = "+" if p['pnl_pct'] >= 0 else ""
        msg += f"  {p['ticker']} ({p['allocation_pct']:.0f}%) ${p['value']:,.0f} | {sign}{p['pnl_pct']:.1f}%\n"

    if portfolio['cash'] > 0:
        msg += f"  CASH ${portfolio['cash']:,.0f}\n"

    created = portfolio.get('created_at', '')[:10]
    updated = portfolio.get('last_updated', '')[:16].replace('T', ' ')
    rebalanced = portfolio.get('last_rebalanced', '')[:10]
    msg += f"\nCreado: {created}\n"
    msg += f"Actualizado: {updated} UTC\n"
    msg += f"Ultimo rebalanceo: {rebalanced}"

    return msg


def format_new_portfolio_message(portfolio: dict) -> str:
    msg = "*Nuevo portfolio creado por GPT-4o*\n\n"
    msg += f"_{portfolio.get('market_thesis', '')}_\n\n"

    for p in portfolio['positions']:
        msg += f"  *{p['ticker']}* — {p['allocation_pct']:.0f}% (${p['value']:,.0f})\n"
        msg += f"    _{p['reasoning']}_\n"

    msg += f"\n  CASH — ${portfolio['cash']:,.0f}"
    msg += f"\n\nBenchmark S&P 500: {portfolio['benchmark_start']:,.0f}"
    msg += "\n\nUsa /test\\_copytrading para ver el estado actualizado."
    return msg


# ---------------------------------------------------------------------------
# Lambda entrypoint
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    log_lambda_event(event, context)
    log = LambdaLogger('copy-trading', context.aws_request_id)

    config = get_config()

    db = DynamoDBClient(
        alerts_table=config.alerts_table,
        state_table=config.state_table,
        subscribers_table=config.subscribers_table,
        digest_queue_table=getattr(config, 'digest_queue_table', ''),
        region=config.region,
    )

    telegram = TelegramClient(config.telegram_bot_token)

    openai_key = os.environ.get('OPENAI_API_KEY', '')
    chat_id = event.get('triggered_by_chat_id')
    action = event.get('action', 'status')  # status | generate | rebalance

    if not openai_key:
        if chat_id:
            telegram.send_message(chat_id, "Error: OPENAI\\_API\\_KEY no configurada.")
        return create_response(500, {'error': 'Missing OPENAI_API_KEY'})

    if not chat_id:
        # Fallback: send to all subscribers
        subs = db.get_subscribers_by_type('all')
        if subs:
            chat_id = int(subs[0]['user_id'])
        else:
            return create_response(200, {'message': 'No chat_id and no subscribers'})

    portfolio = load_portfolio(db)

    if action == 'rebalance' and portfolio:
        telegram.send_message(chat_id, "Rebalanceando portfolio con GPT-4o...")
        portfolio = rebalance_portfolio(db, openai_key, portfolio)
        if portfolio:
            msg = format_new_portfolio_message(portfolio)
            telegram.send_message(chat_id, msg)
            log.info("Portfolio rebalanced")
        else:
            telegram.send_message(chat_id, "Error al rebalancear. Revisa los logs.")

    elif portfolio:
        # Existing portfolio: update prices and show status
        telegram.send_message(chat_id, "Actualizando precios del portfolio...")
        portfolio = update_prices(db, portfolio)
        msg = format_portfolio_message(portfolio)
        telegram.send_message(chat_id, msg)
        log.info("Portfolio prices updated")

    else:
        # No portfolio: generate one
        telegram.send_message(chat_id, "Generando portfolio con GPT-4o... (puede tardar 15-20s)")
        portfolio = generate_portfolio(db, openai_key)
        if portfolio:
            msg = format_new_portfolio_message(portfolio)
            telegram.send_message(chat_id, msg)
            log.info(f"New portfolio created with {len(portfolio['positions'])} positions")
        else:
            telegram.send_message(chat_id, "Error generando el portfolio. Revisa los logs.")

    return create_response(200, {
        'message': 'Copy trading complete',
        'action': action,
        'has_portfolio': portfolio is not None,
    })
