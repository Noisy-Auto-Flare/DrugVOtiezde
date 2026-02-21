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
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========== Конфигурация из .env ==========
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

# ========== Прокси ==========
proxy = None
if PROXY_ADDR and PROXY_PORT:
    proxy_type = socks.SOCKS5 if PROXY_TYPE == 'socks5' else socks.SOCKS4
    proxy = (proxy_type, PROXY_ADDR, PROXY_PORT, True, PROXY_USERNAME, PROXY_PASSWORD)
    logger.info(f"Прокси настроен: {PROXY_TYPE}://{PROXY_ADDR}:{PROXY_PORT}")
else:
    logger.info("Прокси не используется")

# ========== Хранилище истории ==========
user_histories: Dict[int, deque] = {}

def get_user_history(user_id: int) -> deque:
    if user_id not in user_histories:
        user_histories[user_id] = deque(maxlen=MAX_HISTORY_MESSAGES)
    return user_histories[user_id]

def add_to_history(user_id: int, role: str, content: str):
    history = get_user_history(user_id)
    history.append({"role": role, "content": content})

def clear_history(user_id: int):
    if user_id in user_histories:
        user_histories[user_id].clear()
        logger.info(f"История для пользователя {user_id} очищена.")

def build_messages_for_api(history: deque, new_message: str) -> List[dict]:
    """
    Формирует список сообщений для отправки в API.
    - системный промпт
    - вся история (уже чередующиеся user/assistant)
    - новое сообщение пользователя (роль user)
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(list(history))
    messages.append({"role": "user", "content": new_message})
    return messages

# ========== Запрос к LM Studio (GLM-4.7) ==========
async def get_ai_response_from_messages(messages: List[dict]) -> Optional[str]:
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
        "enable_thinking": False,
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
                        return content
                    else:
                        logger.error("Неожиданный формат ответа: %s", data)
                else:
                    logger.error("Ошибка HTTP %d", resp.status)
                    error_text = await resp.text()
                    logger.error("Тело ошибки: %s", error_text)
    except Exception as e:
        logger.exception("Исключение при запросе: %s", e)
    return None

# ========== Основная функция ==========
async def main():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH, proxy=proxy)
    await client.start()
    me = await client.get_me()
    logger.info("Клиент Telegram запущен. Аккаунт: %s", me.username or me.first_name)

    @client.on(events.NewMessage(incoming=True))
    async def handler(event):
        if not event.is_private or event.out:
            return

        text = event.text.strip()
        user_id = event.sender_id

        # Команды
        if text == '/start':
            await event.reply("Привет! Я твой друг с Тенерифе. Просто напиши мне что-нибудь :)")
            return
        if text == '/reset':
            clear_history(user_id)
            await event.reply("История нашего разговора очищена. Начнём заново!")
            return

        logger.info("Новое сообщение от %d: %s", user_id, text[:50])

        # Получаем историю пользователя
        history = get_user_history(user_id)

        # Формируем сообщения для API (история + новое сообщение)
        messages_for_api = build_messages_for_api(history, text)

        # Показываем статус "печатает"
        async with client.action(event.chat_id, 'typing'):
            ai_response = await get_ai_response_from_messages(messages_for_api)

        if ai_response:
            # Сохраняем и сообщение пользователя, и ответ ассистента в историю
            add_to_history(user_id, "user", text)
            add_to_history(user_id, "assistant", ai_response)
            await event.reply(ai_response)
            logger.info("Отправлен ответ для %d: %s", user_id, ai_response)
        else:
            await event.reply("...")
            logger.warning("Ответ нейросети не получен для %d, отправлена заглушка")

    logger.info("Обработчик сообщений зарегистрирован. Ожидаем сообщения...")
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())