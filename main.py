import os
import asyncio
import logging
from collections import deque
from typing import Dict, List, Optional

import socks
from dotenv import load_dotenv
from telethon import TelegramClient, events
import aiohttp

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================== Чтение конфигурации из .env ==================
API_ID = int(os.getenv('API_ID', 0))
API_HASH = os.getenv('API_HASH', '')
SESSION_NAME = os.getenv('SESSION_NAME', 'my_session')
PROXY_TYPE = os.getenv('PROXY_TYPE', 'socks5').lower()
PROXY_ADDR = os.getenv('PROXY_ADDR')
PROXY_PORT = int(os.getenv('PROXY_PORT', 0)) if os.getenv('PROXY_PORT') else None
PROXY_USERNAME = os.getenv('PROXY_USERNAME')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD')
LM_STUDIO_URL = os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1/chat/completions')
SYSTEM_PROMPT = os.getenv(
    'SYSTEM_PROMPT',
    'Ты — мой друг, мы отдыхаем на Острове Тенерифе. '
    'Отвечай коротко и дружелюбно, как в обычном чате. '
    'Не пиши никаких размышлений, анализов или пояснений. '
    'Не используй нумерацию, списки, звёздочки или дефисы. '
    'Просто дай прямой ответ, не более 30 символов.'
)
MAX_HISTORY_MESSAGES = int(os.getenv('MAX_HISTORY_MESSAGES', 10))

# ================== Настройка прокси для Telethon ==================
proxy = None
if PROXY_ADDR and PROXY_PORT:
    proxy_type = socks.SOCKS5 if PROXY_TYPE == 'socks5' else socks.SOCKS4
    proxy = (proxy_type, PROXY_ADDR, PROXY_PORT, True, PROXY_USERNAME, PROXY_PASSWORD)
    logger.info(f"Прокси настроен: {PROXY_TYPE}://{PROXY_ADDR}:{PROXY_PORT}")
else:
    logger.info("Прокси не используется")

# ================== Хранилище истории сообщений для каждого пользователя ==================
# user_id -> deque of dicts (role, content)
user_histories: Dict[int, deque] = {}

def get_user_history(user_id: int) -> deque:
    """Возвращает историю пользователя, создаёт новую, если её нет."""
    if user_id not in user_histories:
        user_histories[user_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
    return user_histories[user_id]

def add_to_history(user_id: int, role: str, content: str):
    """Добавляет сообщение в историю пользователя."""
    history = get_user_history(user_id)
    history.append({"role": role, "content": content})

def clear_history(user_id: int):
    """Очищает историю пользователя."""
    if user_id in user_histories:
        user_histories[user_id].clear()
        logger.info(f"История для пользователя {user_id} очищена.")

def build_messages_for_api(user_id: int, new_message: str) -> List[dict]:
    """
    Строит список сообщений для отправки в API:
    - системный промпт
    - затем последние сообщения из истории (без системного)
    - затем текущее сообщение пользователя
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    history = get_user_history(user_id)
    messages.extend(list(history))
    messages.append({"role": "user", "content": new_message})
    return messages

# ================== Функция запроса к GLM-4.7-Flash через LM Studio ==================
async def get_ai_response(user_id: int, user_message: str) -> Optional[str]:
    """
    Отправляет запрос к LM Studio (GLM-4.7-Flash) с отключённым мышлением.
    Возвращает ответ или None при ошибке.
    """
    messages = build_messages_for_api(user_id, user_message)
    headers = {"Content-Type": "application/json"}
    
    # Параметры запроса: отключаем режим мышления для GLM-4.7
    payload = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
        "enable_thinking": False,          # <-- отключаем мышление
        # Дополнительный параметр для надёжности (по документации)
        "thinking": {"type": "disabled"}
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LM_STUDIO_URL, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    choices = data.get('choices', [])
                    if choices:
                        message = choices[0].get('message', {})
                        content = message.get('content', '').strip()
                        # Если в ответе есть reasoning_content (мышление), мы его игнорируем
                        return content
                    else:
                        logger.error("Неожиданный формат ответа от LM Studio: %s", data)
                else:
                    logger.error("Ошибка HTTP %d при запросе к LM Studio", resp.status)
                    error_text = await resp.text()
                    logger.error("Тело ошибки: %s", error_text)
    except Exception as e:
        logger.exception("Исключение при вызове LM Studio: %s", e)
    return None

# ================== Основная функция ==================
async def main():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH, proxy=proxy)
    await client.start()
    logger.info("Клиент Telegram запущен. Аккаунт: %s", await client.get_me().username)

    @client.on(events.NewMessage(incoming=True))
    async def handler(event):
        # Реагируем только на личные сообщения и не на свои
        if not event.is_private or event.out:
            return

        # Обработка команд
        text = event.text.strip()
        if text == '/start':
            await event.reply("Привет! Я твой друг с Тенерифе. Просто напиши мне что-нибудь :)")
            return
        if text == '/reset':
            clear_history(event.sender_id)
            await event.reply("История нашего разговора очищена. Начнём заново!")
            return

        sender = await event.get_sender()
        user_id = sender.id
        logger.info("Новое сообщение от %s (%d): %s", sender.first_name, user_id, text[:50])

        # Добавляем сообщение пользователя в историю
        add_to_history(user_id, "user", text)

        # Показываем статус "печатает"
        async with client.action(event.chat_id, 'typing'):
            ai_response = await get_ai_response(user_id, text)

        if ai_response:
            await event.reply(ai_response)
            logger.info("Отправлен ответ для %d: %s", user_id, ai_response)
            add_to_history(user_id, "assistant", ai_response)
        else:
            await event.reply("...")
            logger.warning("Ответ нейросети не получен для %d, отправлена заглушка")
            # Можно удалить последнее сообщение пользователя из истории, чтобы не портить контекст
            # (опционально, раскомментируйте при необходимости)
            # history = get_user_history(user_id)
            # if history and history[-1]["role"] == "user":
            #     history.pop()

    logger.info("Обработчик сообщений зарегистрирован. Ожидаем сообщения...")
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())