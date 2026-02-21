import os
import asyncio
import logging
from typing import Optional

import socks
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.tl.types import PeerUser
import aiohttp

# Загружаем переменные окружения из .env
load_denv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация из переменных окружения
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
MAX_RESPONSE_LENGTH = 30  # максимальная длина ответа в символах

# Формируем proxy-параметр для Telethon, если заданы адрес и порт
proxy = None
if PROXY_ADDR and PROXY_PORT:
    proxy_type = socks.SOCKS5 if PROXY_TYPE == 'socks5' else socks.SOCKS4
    proxy = (proxy_type, PROXY_ADDR, PROXY_PORT, True, PROXY_USERNAME, PROXY_PASSWORD)
    logger.info(f"Прокси настроен: {PROXY_TYPE}://{PROXY_ADDR}:{PROXY_PORT}")
else:
    logger.info("Прокси не используется")

async def get_ai_response(user_message: str) -> Optional[str]:
    """
    Отправляет запрос к локальной нейросети (LM Studio) и возвращает ответ.
    """
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 50,          # ограничиваем токены, но символы могут быть длиннее
        "temperature": 0.7,
        "stream": False
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LM_STUDIO_URL, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Извлекаем текст ответа (формат OpenAI)
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
    # Создаём клиента Telethon с поддержкой прокси
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH, proxy=proxy)
    await client.start()
    logger.info("Клиент Telegram запущен")

    @client.on(events.NewMessage(incoming=True))
    async def handler(event):
        # Реагируем только на личные сообщения (не из групп/каналов)
        if not event.is_private:
            return

        # Игнорируем собственные сообщения (если аккаунт отвечает сам себе)
        if event.out:
            return

        sender = await event.get_sender()
        logger.info("Новое сообщение от %s: %s", sender.id, event.text[:50])

        # Показываем статус "печатает"
        async with client.action(event.chat_id, 'typing'):
            # Получаем ответ от нейросети
            ai_response = await get_ai_response(event.text)
            if ai_response:
                await event.reply(ai_response)
                logger.info("Отправлен ответ: %s", ai_response)
            else:
                # Если не удалось получить ответ, можно отправить что-то по умолчанию
                await event.reply("…")
                logger.warning("Ответ нейросети не получен, отправлено заглушка")

    logger.info("Обработчик сообщений зарегистрирован, ожидаем сообщения...")
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())