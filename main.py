import os
import asyncio
import logging
from collections import deque
from typing import Dict, List, Optional

import socks
from dotenv import load_dotenv
from telethon import TelegramClient, events
import aiohttp

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Конфигурация
API_ID = int(os.getenv('API_ID', 0))
API_HASH = os.getenv('API_HASH', '')
SESSION_NAME = os.getenv('SESSION_NAME', 'my_session')
PROXY_TYPE = os.getenv('PROXY_TYPE', 'socks5').lower()
PROXY_ADDR = os.getenv('PROXY_ADDR')
PROXY_PORT = int(os.getenv('PROXY_PORT', 0)) if os.getenv('PROXY_PORT') else None
PROXY_USERNAME = os.getenv('PROXY_USERNAME')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD')
LM_STUDIO_URL = os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1/chat/completions')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'Ты отдыхающий друг на Острове Тенерифе. Общайся дружелюбно, кратко. Отвечай не более 30 символов.')
MAX_RESPONSE_LENGTH = 30
MAX_HISTORY_MESSAGES = 10  # Сколько последних сообщений хранить для каждого пользователя

# Хранилище историй: user_id -> deque of dicts (role, content)
user_histories: Dict[int, deque] = {}

# Прокси (как и раньше)
proxy = None
if PROXY_ADDR and PROXY_PORT:
    proxy_type = socks.SOCKS5 if PROXY_TYPE == 'socks5' else socks.SOCKS4
    proxy = (proxy_type, PROXY_ADDR, PROXY_PORT, True, PROXY_USERNAME, PROXY_PASSWORD)
    logger.info(f"Прокси настроен: {PROXY_TYPE}://{PROXY_ADDR}:{PROXY_PORT}")
else:
    logger.info("Прокси не используется")

def get_user_history(user_id: int) -> deque:
    """Возвращает историю пользователя, создаёт новую, если её нет."""
    if user_id not in user_histories:
        # Используем deque с максимальной длиной для автоматического удаления старых сообщений
        user_histories[user_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
    return user_histories[user_id]

def add_to_history(user_id: int, role: str, content: str):
    """Добавляет сообщение в историю пользователя."""
    history = get_user_history(user_id)
    history.append({"role": role, "content": content})

def build_messages_for_api(user_id: int, new_message: str) -> List[dict]:
    """
    Строит список сообщений для отправки в API:
    - начинается с системного промпта
    - затем последние сообщения из истории (кроме системного)
    - затем текущее сообщение пользователя (оно ещё не в истории, передаём отдельно)
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Добавляем сохранённую историю (без системного промпта)
    history = get_user_history(user_id)
    messages.extend(list(history))  # deque -> list
    # Добавляем новое сообщение пользователя (оно пока не в истории)
    messages.append({"role": "user", "content": new_message})
    return messages

async def get_ai_response(user_id: int, user_message: str) -> Optional[str]:
    """
    Отправляет запрос к локальной нейросети, используя историю пользователя.
    Возвращает ответ или None при ошибке.
    """
    messages = build_messages_for_api(user_id, user_message)
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False
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
                        # Обрезаем до максимальной длины
                        if len(content) > MAX_RESPONSE_LENGTH:
                            content = content[:MAX_RESPONSE_LENGTH].rstrip() + '…'
                        return content
                    else:
                        logger.error("Неожиданный формат ответа от LM Studio: %s", data)
                else:
                    logger.error("Ошибка HTTP %d при запросе к LM Studio", resp.status)
    except Exception as e:
        logger.exception("Исключение при вызове LM Studio: %s", e)
    return None

async def main():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH, proxy=proxy)
    await client.start()
    logger.info("Клиент Telegram запущен")

    @client.on(events.NewMessage(incoming=True))
    async def handler(event):
        if not event.is_private or event.out:
            return

        sender = await event.get_sender()
        user_id = sender.id
        logger.info("Новое сообщение от %s: %s", user_id, event.text[:50])

        # Добавляем сообщение пользователя в историю (до отправки запроса, чтобы модель его учла)
        add_to_history(user_id, "user", event.text)

        # Показываем статус "печатает"
        async with client.action(event.chat_id, 'typing'):
            ai_response = await get_ai_response(user_id, event.text)

        if ai_response:
            await event.reply(ai_response)
            logger.info("Отправлен ответ %s: %s", user_id, ai_response)
            # Добавляем ответ ассистента в историю
            add_to_history(user_id, "assistant", ai_response)
        else:
            # Если ответ не получен, можно отправить заглушку, но в историю её не добавляем
            await event.reply("…")
            logger.warning("Ответ нейросети не получен для %s, отправлена заглушка")
            # Также удаляем последнее сообщение пользователя из истории? Решайте сами.
            # Можно оставить как есть, но тогда пользователь может повторно отправить то же сообщение.
            # Лучше удалить, чтобы история не засорялась:
            # history = get_user_history(user_id)
            # if history and history[-1]["role"] == "user":
            #     history.pop()
            # Но проще ничего не делать, ошибки редки.

    logger.info("Обработчик сообщений зарегистрирован, ожидаем сообщения...")
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())