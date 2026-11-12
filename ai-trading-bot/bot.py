"""
AI Trading Bot — Telegram bot standalone

Comandos:
  /start      — Bienvenida + menú
  /portfolio  — Ver portfolio actual con P&L actualizado
  /crear      — Pedir a la IA que genere un portfolio nuevo
  /rebalancear — Rebalancear con GPT-4o
  /actualizar — Refrescar precios sin rebalancear
  /detalle TICKER — Ver detalle de una posición
"""

import os
import logging
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from ai_copy_trading import AICopyTradingEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BOT_TOKEN      = os.environ['TELEGRAM_BOT_TOKEN']
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

bot    = telebot.TeleBot(BOT_TOKEN)
engine = AICopyTradingEngine(openai_api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        InlineKeyboardButton("🔄 Actualizar",    callback_data="at_refresh"),
        InlineKeyboardButton("⚖️ Rebalancear",   callback_data="at_rebalance"),
    )
    return kb


def _detail_keyboard() -> InlineKeyboardMarkup:
    if not engine.portfolio:
        return InlineKeyboardMarkup()
    kb = InlineKeyboardMarkup(row_width=4)
    buttons = [
        InlineKeyboardButton(p.ticker, callback_data=f"at_detail_{p.ticker}")
        for p in engine.portfolio.positions
    ]
    kb.add(*buttons)
    return kb


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@bot.message_handler(commands=['start', 'help'])
def cmd_start(message):
    has = engine.has_portfolio()
    kb  = InlineKeyboardMarkup(row_width=1)
    if has:
        kb.add(
            InlineKeyboardButton("📊 Ver Portfolio",      callback_data="at_view"),
            InlineKeyboardButton("🔄 Actualizar Precios", callback_data="at_refresh"),
            InlineKeyboardButton("⚖️ Rebalancear (IA)",   callback_data="at_rebalance"),
        )
    else:
        kb.add(InlineKeyboardButton("🚀 Crear Portfolio con IA", callback_data="at_create"))

    text = (
        "🤖 *AI Trading Bot*\n\n"
        "Portfolio de paper trading gestionado por GPT-4o.\n"
        "Capital ficticio: *$100,000 USD* — sin dinero real.\n\n"
        "Comandos:\n"
        "• /portfolio — Ver estado\n"
        "• /crear — Nuevo portfolio\n"
        "• /rebalancear — Rebalancear con IA\n"
        "• /actualizar — Refrescar precios\n"
        "• /detalle TICKER — Detalles de una posición\n"
    )
    text += "\n_Portfolio activo_ ✅" if has else "\n_Sin portfolio activo todavía._"
    bot.send_message(message.chat.id, text, parse_mode='Markdown', reply_markup=kb)


@bot.message_handler(commands=['portfolio'])
def cmd_portfolio(message):
    bot.send_message(message.chat.id, "⏳ Actualizando precios...")
    _send_portfolio(message.chat.id)


@bot.message_handler(commands=['crear'])
def cmd_crear(message):
    bot.send_message(
        message.chat.id,
        "⏳ Consultando a la IA para construir el portfolio...\n_Esto puede tardar unos segundos._",
        parse_mode='Markdown',
    )
    _run_create(message.chat.id)


@bot.message_handler(commands=['rebalancear'])
def cmd_rebalancear(message):
    bot.send_message(
        message.chat.id,
        "⚖️ Consultando a la IA para rebalancear...\n_La IA evaluará el contexto actual del mercado._",
        parse_mode='Markdown',
    )
    _run_rebalance(message.chat.id)


@bot.message_handler(commands=['actualizar'])
def cmd_actualizar(message):
    bot.send_message(message.chat.id, "🔄 Actualizando precios de mercado...")
    _run_refresh(message.chat.id)


@bot.message_handler(commands=['detalle'])
def cmd_detalle(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.send_message(message.chat.id, "Uso: /detalle TICKER\nEjemplo: /detalle AAPL")
        return
    ticker = parts[1].upper()
    msg = engine.format_position_detail(ticker)
    bot.send_message(message.chat.id, msg, parse_mode='Markdown')


# ---------------------------------------------------------------------------
# Inline callbacks
# ---------------------------------------------------------------------------

@bot.callback_query_handler(func=lambda c: c.data.startswith('at_'))
def handle_callback(call):
    chat_id = call.message.chat.id
    bot.answer_callback_query(call.id)

    if call.data == 'at_view':
        bot.send_message(chat_id, "⏳ Actualizando precios...")
        _send_portfolio(chat_id)

    elif call.data == 'at_create':
        bot.send_message(
            chat_id,
            "⏳ Generando portfolio con GPT-4o...\n_Puede tardar unos segundos._",
            parse_mode='Markdown',
        )
        _run_create(chat_id)

    elif call.data == 'at_refresh':
        bot.send_message(chat_id, "🔄 Actualizando precios...")
        _run_refresh(chat_id)

    elif call.data == 'at_rebalance':
        bot.send_message(
            chat_id,
            "⚖️ Rebalanceando con GPT-4o...\n_Esto puede tardar unos segundos._",
            parse_mode='Markdown',
        )
        _run_rebalance(chat_id)

    elif call.data.startswith('at_detail_'):
        ticker = call.data.replace('at_detail_', '')
        msg = engine.format_position_detail(ticker)
        bot.send_message(chat_id, msg, parse_mode='Markdown')


# ---------------------------------------------------------------------------
# Acciones internas
# ---------------------------------------------------------------------------

def _send_portfolio(chat_id: int):
    try:
        msg = engine.format_portfolio_message()
        bot.send_message(chat_id, msg, parse_mode='Markdown', reply_markup=_action_keyboard())
        if engine.portfolio:
            bot.send_message(chat_id, "👇 Tocá una acción para ver el detalle:", reply_markup=_detail_keyboard())
    except Exception as e:
        logger.error(f"Error mostrando portfolio: {e}")
        bot.send_message(chat_id, f"❌ Error: {e}")


def _run_create(chat_id: int):
    try:
        ok = engine.generate_portfolio()
        if ok:
            msg = engine.format_portfolio_message()
            bot.send_message(chat_id, msg, parse_mode='Markdown', reply_markup=_action_keyboard())
        else:
            bot.send_message(
                chat_id,
                "❌ No se pudo generar el portfolio.\n"
                "Verificá que `OPENAI_API_KEY` esté configurado correctamente.",
            )
    except Exception as e:
        logger.error(f"Error creando portfolio: {e}")
        bot.send_message(chat_id, f"❌ Error: {e}")


def _run_refresh(chat_id: int):
    try:
        engine.update_prices()
        msg = engine.format_portfolio_message()
        bot.send_message(chat_id, msg, parse_mode='Markdown', reply_markup=_action_keyboard())
    except Exception as e:
        logger.error(f"Error actualizando precios: {e}")
        bot.send_message(chat_id, f"❌ Error: {e}")


def _run_rebalance(chat_id: int):
    try:
        ok = engine.rebalance_portfolio()
        if ok:
            msg = engine.format_portfolio_message()
            bot.send_message(chat_id, msg, parse_mode='Markdown', reply_markup=_action_keyboard())
        else:
            bot.send_message(
                chat_id,
                "❌ No se pudo rebalancear.\n"
                "Verificá que `OPENAI_API_KEY` esté configurado.",
            )
    except Exception as e:
        logger.error(f"Error rebalanceando: {e}")
        bot.send_message(chat_id, f"❌ Error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logger.info("AI Trading Bot iniciado.")
    bot.infinity_polling()
