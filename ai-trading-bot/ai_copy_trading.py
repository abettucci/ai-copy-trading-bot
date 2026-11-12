"""
AI Trading Engine

Simulates a portfolio managed by an LLM (paper trading).
The AI generates stock allocations, tracks P&L, and compares against S&P 500.

Usage:
    engine = AICopyTradingEngine(openai_api_key="sk-...")
    engine.generate_portfolio()       # Ask LLM to build a portfolio
    engine.update_prices()            # Refresh current prices
    msg = engine.format_portfolio_message()   # Telegram-ready message
"""

import json
import os
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = "ai_portfolio.json"
DEFAULT_CAPITAL = 100_000.0  # $100k paper trading


@dataclass
class Position:
    ticker: str
    allocation_pct: float   # target allocation %
    shares: float           # number of shares held
    entry_price: float      # price at time of purchase
    current_price: float    # latest fetched price
    value: float            # current market value
    pnl: float              # unrealized P&L in dollars
    pnl_pct: float          # unrealized P&L %
    reasoning: str          # LLM explanation for this pick


@dataclass
class AIPortfolio:
    capital: float
    cash: float
    positions: List[Position]
    created_at: str
    last_updated: str
    last_rebalanced: str
    ai_model: str
    market_thesis: str
    benchmark_start: float   # S&P 500 value at portfolio creation
    benchmark_current: float # S&P 500 latest value


class AICopyTradingEngine:
    def __init__(self, openai_api_key: str = None, capital: float = DEFAULT_CAPITAL):
        self.capital = capital
        self.openai_api_key = openai_api_key
        self.portfolio_file = PORTFOLIO_FILE
        self.portfolio: Optional[AIPortfolio] = None
        self._load_portfolio()

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _load_portfolio(self):
        if not os.path.exists(self.portfolio_file):
            return
        try:
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)
            positions = [Position(**p) for p in data.get('positions', [])]
            self.portfolio = AIPortfolio(
                capital=data['capital'],
                cash=data['cash'],
                positions=positions,
                created_at=data['created_at'],
                last_updated=data['last_updated'],
                last_rebalanced=data['last_rebalanced'],
                ai_model=data['ai_model'],
                market_thesis=data['market_thesis'],
                benchmark_start=data['benchmark_start'],
                benchmark_current=data['benchmark_current'],
            )
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")

    def _save_portfolio(self):
        if not self.portfolio:
            return
        data = {
            'capital': self.portfolio.capital,
            'cash': self.portfolio.cash,
            'positions': [asdict(p) for p in self.portfolio.positions],
            'created_at': self.portfolio.created_at,
            'last_updated': self.portfolio.last_updated,
            'last_rebalanced': self.portfolio.last_rebalanced,
            'ai_model': self.portfolio.ai_model,
            'market_thesis': self.portfolio.market_thesis,
            'benchmark_start': self.portfolio.benchmark_start,
            'benchmark_current': self.portfolio.benchmark_current,
        }
        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f, indent=2)

    # ─── Market Data ─────────────────────────────────────────────────────────

    def _fetch_price(self, ticker: str) -> float:
        try:
            hist = yf.Ticker(ticker).history(period='2d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
        return 0.0

    def _fetch_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: self._fetch_price(t) for t in tickers}

    def _get_market_context(self) -> str:
        """Fetch key indices to give the LLM market awareness."""
        indices = {'^GSPC': 'S&P 500', '^NDX': 'Nasdaq 100', '^VIX': 'VIX'}
        lines = []
        for symbol, name in indices.items():
            try:
                hist = yf.Ticker(symbol).history(period='5d')
                if not hist.empty and len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change_pct = (current - prev) / prev * 100
                    lines.append(f"{name}: {current:.0f} ({change_pct:+.1f}%)")
            except Exception:
                pass
        return "\n".join(lines) if lines else "Contexto de mercado no disponible"

    # ─── LLM Call ────────────────────────────────────────────────────────────

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI GPT-4o to generate the portfolio."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un portfolio manager experto. "
                            "Responde ÚNICAMENTE con JSON válido, sin texto adicional, "
                            "sin bloques de código markdown."
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

    def _build_prompt(self, market_context: str) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""Hoy es {today}. Eres un portfolio manager con ${self.capital:,.0f} USD de capital ficticio (paper trading).

Contexto de mercado actual:
{market_context}

Crea un portfolio diversificado de acciones del mercado estadounidense para mantener durante el próximo mes.
- Selecciona entre 6 y 10 empresas sólidas con buenas perspectivas.
- Reserva entre 3% y 6% en cash.
- Cada acción debe tener una tesis de inversión clara.

Responde ÚNICAMENTE con este JSON (sin texto adicional ni bloques de código):
{{
  "market_thesis": "Tu análisis del mercado actual en 1-2 oraciones",
  "cash_pct": 4.0,
  "allocations": [
    {{
      "ticker": "AAPL",
      "allocation_pct": 15.0,
      "reasoning": "Razón para elegir esta acción en 1 oración"
    }}
  ]
}}

Regla: allocation_pct de todas las posiciones + cash_pct deben sumar exactamente 100.
"""

    # ─── Portfolio Generation & Rebalancing ──────────────────────────────────

    def generate_portfolio(self) -> bool:
        """Ask the LLM to create a new portfolio from scratch."""
        market_context = self._get_market_context()
        prompt = self._build_prompt(market_context)
        raw = self._call_openai(prompt)
        if not raw:
            return False
        return self._apply_llm_response(raw, rebalance=False)

    def rebalance_portfolio(self) -> bool:
        """Ask the LLM to re-evaluate and rebuild the portfolio."""
        market_context = self._get_market_context()
        current_summary = ""
        if self.portfolio:
            current_summary = "\nPortfolio actual:\n" + "\n".join(
                f"- {p.ticker}: {p.allocation_pct:.0f}% (P&L: {p.pnl_pct:+.1f}%)"
                for p in self.portfolio.positions
            )
        prompt = self._build_prompt(market_context + current_summary)
        raw = self._call_openai(prompt)
        if not raw:
            return False
        return self._apply_llm_response(raw, rebalance=True)

    def _apply_llm_response(self, raw: str, rebalance: bool) -> bool:
        """Parse LLM JSON and build/update the portfolio."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw: {raw}")
            return False

        try:
            allocations = data['allocations']
            cash_pct = float(data.get('cash_pct', 4.0))
            market_thesis = data.get('market_thesis', '')

            tickers = [a['ticker'] for a in allocations]
            prices = self._fetch_prices(tickers)
            sp500_price = self._fetch_price('^GSPC')

            positions = []
            for alloc in allocations:
                ticker = alloc['ticker']
                price = prices.get(ticker, 0.0)
                if price == 0.0:
                    logger.warning(f"Skipping {ticker}: price not available")
                    continue

                alloc_pct = float(alloc['allocation_pct'])
                alloc_value = self.capital * alloc_pct / 100.0
                shares = alloc_value / price

                positions.append(Position(
                    ticker=ticker,
                    allocation_pct=alloc_pct,
                    shares=shares,
                    entry_price=price,
                    current_price=price,
                    value=alloc_value,
                    pnl=0.0,
                    pnl_pct=0.0,
                    reasoning=alloc.get('reasoning', ''),
                ))

            now = datetime.now().isoformat()
            benchmark_start = (
                self.portfolio.benchmark_start if rebalance and self.portfolio
                else sp500_price
            )

            self.portfolio = AIPortfolio(
                capital=self.capital,
                cash=self.capital * cash_pct / 100.0,
                positions=positions,
                created_at=self.portfolio.created_at if rebalance and self.portfolio else now,
                last_updated=now,
                last_rebalanced=now,
                ai_model="gpt-4o",
                market_thesis=market_thesis,
                benchmark_start=benchmark_start,
                benchmark_current=sp500_price,
            )
            self._save_portfolio()
            return True

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error applying LLM response: {e}")
            return False

    # ─── Price Updates & Metrics ─────────────────────────────────────────────

    def update_prices(self):
        """Refresh current prices and recalculate P&L for all positions."""
        if not self.portfolio:
            return

        tickers = [p.ticker for p in self.portfolio.positions]
        prices = self._fetch_prices(tickers)
        sp500_price = self._fetch_price('^GSPC')

        for pos in self.portfolio.positions:
            current = prices.get(pos.ticker, pos.current_price)
            pos.current_price = current
            pos.value = pos.shares * current
            pos.pnl = pos.value - (pos.shares * pos.entry_price)
            pos.pnl_pct = (current - pos.entry_price) / pos.entry_price * 100 if pos.entry_price else 0

        self.portfolio.benchmark_current = sp500_price
        self.portfolio.last_updated = datetime.now().isoformat()
        self._save_portfolio()

    def get_total_value(self) -> float:
        if not self.portfolio:
            return 0.0
        return sum(p.value for p in self.portfolio.positions) + self.portfolio.cash

    def get_total_pnl(self) -> float:
        return self.get_total_value() - self.capital

    def get_total_pnl_pct(self) -> float:
        if self.capital == 0:
            return 0.0
        return self.get_total_pnl() / self.capital * 100

    def get_benchmark_pnl_pct(self) -> float:
        if not self.portfolio or self.portfolio.benchmark_start == 0:
            return 0.0
        return (
            (self.portfolio.benchmark_current - self.portfolio.benchmark_start)
            / self.portfolio.benchmark_start * 100
        )

    # ─── Formatting ──────────────────────────────────────────────────────────

    def format_portfolio_message(self) -> str:
        """Return a Telegram-ready Markdown message with the portfolio status."""
        if not self.portfolio:
            return (
                "❌ No hay portfolio activo.\n\n"
                "Usa el botón *Crear Portfolio* para que la IA construya uno."
            )

        self.update_prices()

        total_value = self.get_total_value()
        total_pnl = self.get_total_pnl()
        total_pnl_pct = self.get_total_pnl_pct()
        benchmark_pnl_pct = self.get_benchmark_pnl_pct()
        vs_benchmark = total_pnl_pct - benchmark_pnl_pct

        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        vs_emoji = "✅" if vs_benchmark >= 0 else "❌"

        # ── Header ──
        msg = "🤖 *AI Trading — Paper Portfolio*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"💼 Valor total: *${total_value:,.0f}*\n"
        msg += f"{pnl_emoji} P&L total: *${total_pnl:+,.0f}* ({total_pnl_pct:+.1f}%)\n"
        msg += f"{vs_emoji} vs S\\&P 500: *{vs_benchmark:+.1f}%* (S&P: {benchmark_pnl_pct:+.1f}%)\n"
        msg += f"\n💡 _{self.portfolio.market_thesis}_\n\n"

        # ── Positions ──
        msg += "*Posiciones:*\n"
        sorted_pos = sorted(self.portfolio.positions, key=lambda p: p.value, reverse=True)
        for p in sorted_pos:
            arrow = "▲" if p.pnl >= 0 else "▼"
            dot = "🟢" if p.pnl >= 0 else "🔴"
            msg += (
                f"{dot} *{p.ticker}* ({p.allocation_pct:.0f}%) "
                f"${p.value:,.0f} | {arrow}{abs(p.pnl_pct):.1f}%\n"
            )

        if self.portfolio.cash > 0:
            msg += f"💵 *CASH* — ${self.portfolio.cash:,.0f}\n"

        # ── Footer ──
        updated = datetime.fromisoformat(self.portfolio.last_updated)
        rebalanced = datetime.fromisoformat(self.portfolio.last_rebalanced)
        msg += f"\n⏰ Actualizado: {updated.strftime('%d/%m %H:%M')}"
        msg += f"\n🔄 Último rebalanceo: {rebalanced.strftime('%d/%m/%Y')}"

        return msg

    def format_position_detail(self, ticker: str) -> str:
        """Return detail for a single position."""
        if not self.portfolio:
            return "No hay portfolio activo."

        pos = next((p for p in self.portfolio.positions if p.ticker == ticker), None)
        if not pos:
            return f"No se encontró {ticker} en el portfolio."

        arrow = "▲" if pos.pnl >= 0 else "▼"
        msg = f"📊 *{pos.ticker} — Detalle*\n"
        msg += f"Asignación: {pos.allocation_pct:.1f}%\n"
        msg += f"Shares: {pos.shares:.4f}\n"
        msg += f"Precio entrada: ${pos.entry_price:.2f}\n"
        msg += f"Precio actual: ${pos.current_price:.2f}\n"
        msg += f"Valor: ${pos.value:,.2f}\n"
        msg += f"P&L: {arrow}${abs(pos.pnl):,.2f} ({pos.pnl_pct:+.1f}%)\n\n"
        msg += f"💡 _{pos.reasoning}_"
        return msg

    def has_portfolio(self) -> bool:
        return self.portfolio is not None and len(self.portfolio.positions) > 0
