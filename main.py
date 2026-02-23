import os
import asyncio
import logging
import base64
from collections import deque
from typing import Dict, List, Optional, Union, Any

import socks
from dotenv import load_dotenv
from telethon import TelegramClient, events, functions
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import aiohttp

load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========== Чтение конфигурации ==========
API_ID = int(os.getenv('API_ID', 0))
API_HASH = os.getenv('API_HASH', '')
SESSION_NAME = os.getenv('SESSION_NAME', 'my_session')
PROXY_TYPE = os.getenv('PROXY_TYPE', 'socks5').lower()
PROXY_ADDR = os.getenv('PROXY_ADDR')
PROXY_PORT = int(os.getenv('PROXY_PORT', 0)) if os.getenv('PROXY_PORT') else None
PROXY_USERNAME = os.getenv('PROXY_USERNAME')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD')
# Добавляем чтение RAW_MODE
RAW_MODE = os.getenv('RAW_MODE', 'false').lower() == 'true'
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
ENABLE_THINKING = os.getenv('ENABLE_THINKING', 'false').lower() == 'true'
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 100))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
ONLINE_TIMEOUT = int(os.getenv('ONLINE_TIMEOUT', 30))  # секунд
VISION_ENABLED = os.getenv('VISION_ENABLED', 'true').lower() == 'true'
ENABLE_TYPING_DELAY= os.getenv('ENABLE_TYPING_DELAY', 'false').lower() == 'true'
DELAY_PER_CHAR = float(os.getenv('DELAY_PER_CHAR', 0.05))  # секунд на символ
MIN_DELAY = float(os.getenv('MIN_DELAY', 1.0))  # минимальная задержка
MAX_DELAY = float(os.getenv('MAX_DELAY', 5.0))  #

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

def add_to_history(user_id: int, role: str, content: Union[str, List[dict]]):
    """
    Добавляет сообщение в историю. content может быть строкой (текст) или списком (multimodal).
    Для совместимости храним как есть, но для построения запроса будем использовать исходный формат.
    """
    history = get_user_history(user_id)
    history.append({"role": role, "content": content})

def clear_history(user_id: int):
    if user_id in user_histories:
        user_histories[user_id].clear()
        logger.info(f"История для пользователя {user_id} очищена.")

def build_messages_for_api(history: deque, new_message_content: Union[str, List[dict]]) -> List[dict]:
    """
    Формирует список сообщений для отправки в API.
    Если RAW_MODE=True, системный промпт НЕ добавляется.
    """
    messages = []
    if not RAW_MODE:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(list(history))
    messages.append({"role": "user", "content": new_message_content})
    return messages

# ========== Вспомогательные функции для изображений ==========
async def extract_image_content(event) -> Optional[bytes]:
    """
    Извлекает изображение из сообщения, возвращает байты или None.
    Поддерживает фото и документы-изображения.
    """
    if event.photo:
        # Фото
        return await event.download_media(bytes)
    elif event.document and event.document.mime_type and event.document.mime_type.startswith('image/'):
        # Документ с изображением
        return await event.download_media(bytes)
    return None

def create_multimodal_content(text: str, image_bytes: bytes) -> List[dict]:
    """
    Создаёт content в формате multimodal: текстовый элемент и изображение в base64.
    """
    # Кодируем изображение в base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    # Определяем MIME-тип (можно упрощённо взять image/jpeg, но лучше определить)
    # Для простоты будем считать, что это JPEG; если не подойдёт, модель может не понять.
    # Можно попытаться определить по магическим числам, но для большинства случаев JPEG подойдёт.
    mime_type = "image/jpeg"  # можно улучшить
    content = [
        {"type": "text", "text": text} if text else None,
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
    ]
    # Убираем пустой текстовый элемент, если текст отсутствует
    return [item for item in content if item is not None]

# ========== Запрос к LM Studio ==========
async def get_ai_response_from_messages(messages: List[dict]) -> Optional[str]:
    headers = {"Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    # Добавляем параметры отключения мышления только если НЕ raw_mode и мышление выключено
    if not RAW_MODE and not ENABLE_THINKING:
        payload["enable_thinking"] = False
        payload["thinking"] = {"type": "disabled"}
    # В raw_mode эти параметры не добавляем, оставляя модель в её естественном состоянии

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
                    logger.error("Ошибка HTTP %d при запросе к LM Studio", resp.status)
                    error_text = await resp.text()
                    logger.error("Тело ошибки: %s", error_text)
    except Exception as e:
        logger.exception("Исключение при запросе: %s", e)
    return None

# ========== Управление статусом онлайн ==========
class OnlineStatusManager:
    def __init__(self, client: TelegramClient, timeout: int):
        self.client = client
        self.timeout = timeout
        self._offline_task: Optional[asyncio.Task] = None

    async def set_online(self):
        """Включает статус онлайн и отменяет запланированное отключение."""
        try:
            await self.client(functions.account.UpdateStatusRequest(offline=False))
            logger.debug("Статус установлен: онлайн")
        except Exception as e:
            logger.error("Не удалось установить онлайн: %s", e)

        # Отменяем предыдущий таймер отключения, если был
        if self._offline_task and not self._offline_task.done():
            self._offline_task.cancel()
            try:
                await self._offline_task
            except asyncio.CancelledError:
                pass

        # Запускаем новый таймер
        self._offline_task = asyncio.create_task(self._auto_offline())

    async def _auto_offline(self):
        """Через timeout секунд переводит статус в офлайн."""
        try:
            await asyncio.sleep(self.timeout)
            await self.client(functions.account.UpdateStatusRequest(offline=True))
            logger.debug("Статус установлен: офлайн (по таймауту)")
        except asyncio.CancelledError:
            logger.debug("Таймер отключения отменён")
        except Exception as e:
            logger.error("Не удалось установить офлайн: %s", e)

    async def shutdown(self):
        """При завершении работы принудительно ставим офлайн."""
        if self._offline_task and not self._offline_task.done():
            self._offline_task.cancel()
            try:
                await self._offline_task
            except asyncio.CancelledError:
                pass
        try:
            await self.client(functions.account.UpdateStatusRequest(offline=True))
            logger.info("Статус установлен: офлайн (завершение работы)")
        except Exception as e:
            logger.error("Не удалось установить офлайн при завершении: %s", e)

# ========== Основная функция ==========
async def main():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH, proxy=proxy)
    await client.start()
    me = await client.get_me()
    logger.info("Клиент Telegram запущен. Аккаунт: %s", me.username or me.first_name)

    # Создаём менеджер статуса онлайн
    online_manager = OnlineStatusManager(client, ONLINE_TIMEOUT)

    @client.on(events.NewMessage(incoming=True))
    async def handler(event):
        if not event.is_private or event.out:
            return
        
        try:
            await client.send_read_acknowledge(event.chat_id)
        except Exception as e:
            logger.warning("Не удалось пометить сообщение как прочитанное: %s", e)

        text = event.text.strip()
        user_id = event.sender_id

        # Команды
        if text == '/start':
            await event.reply("Привет)")
            return
        if text == '/reset':
            clear_history(user_id)
            await event.reply("История нашего разговора очищена. Начнём заново!")
            return

        logger.info("Новое сообщение от %d: %s", user_id, text[:50])

        # Устанавливаем статус онлайн и сбрасываем таймер
        await online_manager.set_online()

        # Проверяем наличие изображения
        image_bytes = None
        if VISION_ENABLED and (event.photo or (event.document and event.document.mime_type.startswith('image/'))):
            try:
                image_bytes = await extract_image_content(event)
                logger.info("Изображение получено, размер: %d байт", len(image_bytes) if image_bytes else 0)
            except Exception as e:
                logger.error("Ошибка при скачивании изображения: %s", e)

        # Формируем content для нового сообщения
        if image_bytes and VISION_ENABLED:
            # Мультимодальное сообщение (текст + картинка)
            new_content = create_multimodal_content(text, image_bytes)
        else:
            # Только текст
            new_content = text

        # Получаем историю пользователя
        history = get_user_history(user_id)
        # Строим сообщения для API (история + новое сообщение)
        messages_for_api = build_messages_for_api(history, new_content)

        start_time = asyncio.get_event_loop().time()

        # Показываем статус "печатает" (будет активен всё время, пока генерируется ответ и пока длится задержка)
        async with client.action(event.chat_id, 'typing'):
            ai_response = await get_ai_response_from_messages(messages_for_api)

            if ai_response and ENABLE_TYPING_DELAY:
                elapsed = asyncio.get_event_loop().time() - start_time
                delay_needed = len(ai_response) * DELAY_PER_CHAR
                delay_needed = max(MIN_DELAY, min(MAX_DELAY, delay_needed))
                wait_time = max(0, delay_needed - elapsed)
                if wait_time > 0:
                    logger.debug("Дополнительная задержка перед отправкой: %.2f сек", wait_time)
                    await asyncio.sleep(wait_time)

        if ai_response:
        # Сохраняем в историю
            add_to_history(user_id, "user", new_content)
            add_to_history(user_id, "assistant", ai_response)
            # Отправляем с разбиением
            await send_long_message(event, ai_response)
            logger.info("Отправлен ответ для %d (длина %d символов, общая задержка %.2f сек)",
                    user_id, len(ai_response), asyncio.get_event_loop().time() - start_time)
        else:
            await event.reply("...")
            logger.warning("Ответ нейросети не получен для %d", user_id)
        
    logger.info("Обработчик сообщений зарегистрирован. Ожидаем сообщения...")
    try:
        await client.run_until_disconnected()
    finally:
        # При выходе выключаем онлайн
        await online_manager.shutdown()

if __name__ == '__main__':
    asyncio.run(main())