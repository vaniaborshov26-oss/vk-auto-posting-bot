import os 
import io
import json
import time
import base64

import requests
import telebot
import google.generativeai as genai
from PIL import Image


# ============================================================
# НАСТРОЙКИ
# ============================================================

# ОБЯЗАТЕЛЬНО ВСТАВЬ НОВЫЕ ПЕРЕВЫПУЩЕННЫЕ КЛЮЧИ
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN")

VK_GROUP_ID = os.getenv("VK_GROUP_ID")

# Сохраняем модель из твоего рабочего скрипта
GEMINI_MODEL = "gemini-3.6-flash"

# Сохраняем версию VK из рабочего скрипта
VK_API_VERSION = "5.131"

# Таймауты
GEMINI_TIMEOUT = 120
VK_TIMEOUT = 120

# Повторные попытки Gemini
GEMINI_RETRIES = 3

# Паузы между попытками
RETRY_DELAYS = [3, 7, 15]


# ============================================================
# GEMINI
# ============================================================

genai.configure(
    api_key=GEMINI_API_KEY
)

gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config={
        "response_mime_type": "application/json"
    }
)


# ============================================================
# TELEGRAM
# ============================================================

telebot.apihelper.RETRY_ON_ERROR = True

bot = telebot.TeleBot(
    TELEGRAM_BOT_TOKEN
)


# ============================================================
# БЕЗОПАСНЫЙ JSON
# ============================================================

def safe_json_response(response, source_name):
    try:
        return response.json()
    except ValueError:
        raise Exception(
            f"{source_name} вернул не JSON.\n"
            f"HTTP: {response.status_code}\n"
            f"Ответ:\n{response.text[:3000]}"
        )


# ============================================================
# VK API
# ============================================================

def vk_call(method_name, params=None, data=None):
    """
    Сохраняем логику старого рабочего скрипта:

    params -> GET
    data   -> POST
    """

    clean_method = (
        method_name
        .strip()
        .split("/")[-1]
        .replace(".json", "")
    )

    endpoint_url = (
        "https://api.vk.com/method/"
        + clean_method
    )

    try:

        if data is not None:

            response = requests.post(
                endpoint_url,
                data=data,
                timeout=VK_TIMEOUT
            )

        else:

            response = requests.get(
                endpoint_url,
                params=params or {},
                timeout=VK_TIMEOUT
            )

    except requests.exceptions.Timeout:

        raise Exception(
            f"VK timeout в {clean_method}: "
            f"сервер не ответил за "
            f"{VK_TIMEOUT} секунд."
        )

    except requests.exceptions.ConnectionError as e:

        raise Exception(
            f"Ошибка соединения с VK "
            f"в {clean_method}: {e}"
        )

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"HTTP ошибка VK "
            f"в {clean_method}: {e}"
        )

    result = safe_json_response(
        response,
        f"VK {clean_method}"
    )

    print(
        f"[VK] {clean_method} "
        f"HTTP={response.status_code}"
    )

    return result


# ============================================================
# ОЧИСТКА JSON GEMINI
# ============================================================

def clean_json_response(raw_text):

    if not raw_text:
        raise Exception(
            "Gemini вернул пустой ответ."
        )

    cleaned = raw_text.strip()

    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


# ============================================================
# GEMINI: АНАЛИЗ ФОТО
# ============================================================

def analyze_photo_with_gemini(image_bytes):

    system_prompt = """
Проанализируй прикреплённое изображение и составь подробное
текстовое описание кадра.

Выдай ответ СТРОГО в формате JSON без вводных слов,
без markdown и без блока ```json.

Используй следующие ключи:

{
  "photo_title": "Ёмкое, красивое и точное название, которое отражает конкретный сюжет, кто изображён и где происходит действие",
  "photo_style": "Определи стиль фото по самому изображению",
  "camera_and_settings": "Предполагаемая камера, объектив и параметры съёмки",
  "shot_type_and_pose_intro": "Ракурс, ориентация кадра, положение тела и взгляд",
  "hairstyle_and_makeup": "Детальное описание причёски, бровей, макияжа и кожи",
  "outfit": "Детальное описание одежды, обуви и аксессуаров",
  "pose_details": "Детальное описание позы, пластики тела, рук и пальцев",
  "lighting": "Описание освещения, источников света и теней",
  "background": "Описание фона и окружения",
  "color_grading_and_style": "Описание цветокоррекции, контраста и палитры",
  "quality_and_style_tags": "Параметры качества и стилистические теги",
  "hashtags": "5-8 целевых хэштегов через пробел"
}

Названия и описания делай максимально полезными
для последующего создания промпта для генерации изображения.

camera_and_settings:
если точные параметры неизвестны, укажи реалистичные
предполагаемые значения.

hashtags должны описывать сюжет, персонажа,
место и стиль.
"""

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    last_error = None

    for attempt in range(
        GEMINI_RETRIES
    ):

        try:

            print(
                f"[Gemini] Попытка "
                f"{attempt + 1}/{GEMINI_RETRIES}"
            )

            response = gemini_model.generate_content(
                [
                    system_prompt,
                    image
                ],
                request_options={
                    "timeout": GEMINI_TIMEOUT
                }
            )

            clean_text = clean_json_response(
                response.text
            )

            data = json.loads(
                clean_text
            )

            required_keys = [
                "photo_title",
                "photo_style",
                "camera_and_settings",
                "shot_type_and_pose_intro",
                "hairstyle_and_makeup",
                "outfit",
                "pose_details",
                "lighting",
                "background",
                "color_grading_and_style",
                "quality_and_style_tags",
                "hashtags"
            ]

            missing = [
                key
                for key in required_keys
                if key not in data
            ]

            if missing:

                raise Exception(
                    "Gemini не вернул поля: "
                    + ", ".join(missing)
                )

            print(
                "[Gemini] Анализ успешно получен."
            )

            return data

        except Exception as e:

            last_error = e

            print(
                f"[Gemini] Ошибка: {e}"
            )

            if attempt < GEMINI_RETRIES - 1:

                delay = RETRY_DELAYS[
                    attempt
                ]

                print(
                    f"[Gemini] Повтор "
                    f"через {delay} сек."
                )

                time.sleep(delay)

    raise Exception(
        "Gemini не смог обработать "
        f"фотографию после "
        f"{GEMINI_RETRIES} попыток.\n"
        f"Последняя ошибка: {last_error}"
    )


# ============================================================
# ТЕКСТ ПОСТА
# ============================================================

def build_wall_post_text(data):

    title = data.get(
        "photo_title",
        "Нейрофотосессия"
    )

    hashtags = data.get(
        "hashtags",
        "#нейрофото #промпт #нейросеть"
    )

    return f"""✨ {title}

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{hashtags}"""


# ============================================================
# ТЕКСТ ПРОМПТА
# ============================================================

def build_comment_prompt_text(data):

    return f"""📌 Промпт для генерации:

Внешность должна полностью соответствовать прикреплённому референсу:
идентичные черты лица, возраст, рост, форма лица, цвет глаз,
цвет и длина волос, причёска, телосложение, пропорции, макияж,
выражение лица и общее визуальное впечатление.

Любые изменения внешности, стилизация под другого человека
или искажение типажа недопустимы.

Стиль:
{data.get('photo_style', '')}

Камера и параметры:
{data.get('camera_and_settings', '')}

{data.get('shot_type_and_pose_intro', '')}

Причёска, кожа и макияж:
{data.get('hairstyle_and_makeup', '')}

Образ:
{data.get('outfit', '')}

Поза:
{data.get('pose_details', '')}

Освещение:
{data.get('lighting', '')}

Фон:
{data.get('background', '')}

Цветокор:
{data.get('color_grading_and_style', '')}

Качество и стиль:
{data.get(
    'quality_and_style_tags',
    'Максимальный фотореализм, высокая детализация, '
    'естественная анатомия, реалистичная кожа, '
    'без CGI, без мультяшности, без артефактов.'
)}"""


# ============================================================
# VK: ПУБЛИКАЦИЯ
# ============================================================

def post_to_vk(
    image_bytes,
    wall_text,
    comment_text
):

    group_id = int(
        str(VK_GROUP_ID).replace(
            "-",
            ""
        )
    )

    # ========================================================
    # 1. Получаем upload_url
    # ========================================================

    print(
        "[VK] Получаю upload_url..."
    )

    server_res = vk_call(
        "photos.getWallUploadServer",
        params={
            "group_id": group_id,
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": VK_API_VERSION
        }
    )

    print(
        "[VK] getWallUploadServer:"
    )
    print(server_res)

    if "error" in server_res:

        raise Exception(
            "Ошибка VK getWallUploadServer: "
            f"{server_res['error']}"
        )

    if "response" not in server_res:

        raise Exception(
            "VK не вернул response:\n"
            f"{server_res}"
        )

    if "upload_url" not in server_res["response"]:

        raise Exception(
            "VK не вернул upload_url:\n"
            f"{server_res}"
        )

    raw_upload_url = (
        server_res["response"]["upload_url"]
    )

    # Сохраняем очистку URL из рабочего кода
    clean_upload_url = (
        raw_upload_url.split("](")[-1].rstrip(")")
        if "](" in raw_upload_url
        else raw_upload_url.strip("[]() ")
    )

    print(
        "[VK] upload_url получен."
    )

    # ========================================================
    # 2. Загружаем фотографию
    # ========================================================

    files = {
        "photo": (
            "photo.jpg",
            image_bytes,
            "image/jpeg"
        )
    }

    print(
        "[VK] Загружаю фотографию..."
    )

    try:

        upload_response = requests.post(
            clean_upload_url,
            files=files,
            timeout=VK_TIMEOUT
        )

    except requests.exceptions.Timeout:

        raise Exception(
            f"VK upload server не ответил "
            f"за {VK_TIMEOUT} секунд."
        )

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"Ошибка загрузки фото в VK: {e}"
        )

    # ========================================================
    # 3. Диагностика upload server
    # ========================================================

    print(
        "\n"
        "================ VK UPLOAD ================\n"
    )

    print(
        "HTTP STATUS:",
        upload_response.status_code
    )

    print(
        "CONTENT TYPE:",
        upload_response.headers.get(
            "content-type"
        )
    )

    print(
        "RAW RESPONSE:"
    )

    print(
        upload_response.text
    )

    print(
        "\n"
        "============================================\n"
    )

    upload_res = safe_json_response(
        upload_response,
        "VK upload server"
    )

    # ========================================================
    # 4. Проверяем upload
    # ========================================================

    if not upload_res:

        raise Exception(
            "VK upload server "
            "вернул пустой ответ."
        )

    if "error" in upload_res:

        raise Exception(
            "Ошибка VK upload server:\n"
            f"{upload_res['error']}"
        )

    server = upload_res.get(
        "server"
    )

    photo = upload_res.get(
        "photo"
    )

    vk_hash = upload_res.get(
        "hash"
    )

    print(
        "[VK] server =",
        server
    )

    print(
        "[VK] photo =",
        repr(photo)
    )

    print(
        "[VK] hash =",
        vk_hash
    )

    if not server:

        raise Exception(
            "VK upload server не вернул "
            "server:\n"
            f"{upload_res}"
        )

    if not vk_hash:

        raise Exception(
            "VK upload server не вернул "
            "hash:\n"
            f"{upload_res}"
        )

    # Главная проверка твоей ошибки
    if (
        photo is None
        or str(photo).strip() == ""
        or photo == "[]"
        or photo == "undefined"
        or str(photo).lower() == "null"
    ):

        raise Exception(
            "VK upload server вернул "
            "ПУСТОЙ photo.\n\n"
            "Фото не удалось передать "
            "на этап photos.saveWallPhoto.\n\n"
            f"Полный ответ VK:\n{upload_res}"
        )

    print(
        "[VK] Фотография успешно загружена."
    )

    # ========================================================
    # 5. Сохраняем фото в VK
    # ========================================================

    print(
        "[VK] Сохраняю фото..."
    )

    save_res = vk_call(
        "photos.saveWallPhoto",
        data={
            "group_id": group_id,
            "server": server,
            "photo": photo,
            "hash": vk_hash,
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": VK_API_VERSION
        }
    )

    print(
        "[VK] saveWallPhoto:"
    )
    print(save_res)

    if "error" in save_res:

        raise Exception(
            "Ошибка VK saveWallPhoto: "
            f"{save_res['error']}"
        )

    try:

        photo_info = (
            save_res["response"][0]
        )

        owner_id = (
            photo_info["owner_id"]
        )

        photo_id = (
            photo_info["id"]
        )

    except Exception as e:

        raise Exception(
            "VK не вернул данные "
            "сохранённой фотографии:\n"
            f"{save_res}"
        ) from e

    print(
        f"[VK] Фото сохранено: "
        f"photo{owner_id}_{photo_id}"
    )

    # ========================================================
    # 6. wall.post
    # ========================================================

    print(
        "[VK] Публикую пост..."
    )

    post_res = vk_call(
        "wall.post",
        data={
            "owner_id": -group_id,
            "from_group": 1,
            "message": wall_text,
            "attachments": (
                f"photo{owner_id}_{photo_id}"
            ),
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": VK_API_VERSION
        }
    )

    print(
        "[VK] wall.post:"
    )
    print(post_res)

    if "error" in post_res:

        raise Exception(
            "Ошибка VK wall.post: "
            f"{post_res['error']}"
        )

    if (
        "response" not in post_res
        or "post_id" not in post_res["response"]
    ):

        raise Exception(
            "VK не вернул post_id:\n"
            f"{post_res}"
        )

    post_id = (
        post_res["response"]["post_id"]
    )

    print(
        f"[VK] Пост опубликован: {post_id}"
    )

    # ========================================================
    # 7. Комментарий
    # ========================================================

    print(
        "[VK] Публикую комментарий..."
    )

    comment_res = vk_call(
        "wall.createComment",
        data={
            "owner_id": -group_id,
            "post_id": post_id,
            "from_group": group_id,
            "message": comment_text,
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": VK_API_VERSION
        }
    )

    # Если from_group=group_id не сработал
    if "error" in comment_res:

        print(
            "[VK] Первая попытка комментария "
            "не удалась."
        )

        print(
            comment_res
        )

        print(
            "[VK] Пробую from_group=1..."
        )

        comment_res = vk_call(
            "wall.createComment",
            data={
                "owner_id": -group_id,
                "post_id": post_id,
                "from_group": 1,
                "message": comment_text,
                "access_token": VK_ACCESS_TOKEN.strip(),
                "v": VK_API_VERSION
            }
        )

    if "error" in comment_res:

        print(
            "⚠️ Ошибка публикации комментария:"
        )

        print(
            comment_res["error"]
        )

    else:

        print(
            "✅ Промпт опубликован "
            "в комментариях."
        )

    # ========================================================
    # 8. Ссылка
    # ========================================================

    return (
        f"https://vk.com/wall-"
        f"{group_id}_{post_id}"
    )


# ============================================================
# TELEGRAM / START
# ============================================================

@bot.message_handler(
    commands=["start"]
)
def send_welcome(message):

    bot.reply_to(
        message,
        "👋 Привет!\n\n"
        "Отправь мне фотографию, и я:\n"
        "1️⃣ Проанализирую её через Gemini\n"
        "2️⃣ Создам название и хэштеги\n"
        "3️⃣ Сформирую промпт\n"
        "4️⃣ Опубликую фото в VK\n"
        "5️⃣ Оставлю промпт первым комментарием"
    )


# ============================================================
# TELEGRAM / ФОТО
# ============================================================

@bot.message_handler(
    content_types=[
        "photo",
        "document"
    ]
)
def handle_photo(message):

    status_msg = bot.reply_to(
        message,
        "⏳ Анализирую сюжет фото через Gemini..."
    )

    try:

        # ----------------------------------------------------
        # 1. Получаем Telegram file_id
        # ----------------------------------------------------

        if message.photo:

            file_id = (
                message.photo[-1].file_id
            )

        elif message.document:

            file_id = (
                message.document.file_id
            )

        else:

            raise Exception(
                "Фото не найдено."
            )

        # ----------------------------------------------------
        # 2. Скачиваем файл
        # ----------------------------------------------------

        print(
            "[Telegram] Скачиваю фото..."
        )

        file_info = bot.get_file(
            file_id
        )

        downloaded_file = (
            bot.download_file(
                file_info.file_path
            )
        )

        if not downloaded_file:

            raise Exception(
                "Не удалось скачать "
                "фотографию из Telegram."
            )

        print(
            "[Telegram] Фото получено: "
            f"{len(downloaded_file)} байт"
        )

        # ----------------------------------------------------
        # 3. Проверяем/нормализуем изображение
        # ----------------------------------------------------

        try:

            image = Image.open(
                io.BytesIO(downloaded_file)
            )

            print(
                "[Image] Формат:",
                image.format
            )

            print(
                "[Image] Размер:",
                image.size
            )

            # Для VK делаем гарантированный JPEG.
            # Это особенно полезно, если пользователь
            # прислал документ PNG/WebP и т.д.

            rgb_image = image.convert(
                "RGB"
            )

            normalized_buffer = (
                io.BytesIO()
            )

            rgb_image.save(
                normalized_buffer,
                format="JPEG",
                quality=95
            )

            vk_image_bytes = (
                normalized_buffer.getvalue()
            )

            print(
                "[Image] JPEG для VK:",
                len(vk_image_bytes),
                "байт"
            )

        except Exception as image_error:

            raise Exception(
                f"Не удалось обработать "
                f"изображение: {image_error}"
            )

        # ----------------------------------------------------
        # 4. Gemini
        # ----------------------------------------------------

        json_data = (
            analyze_photo_with_gemini(
                downloaded_file
            )
        )

        wall_text = (
            build_wall_post_text(
                json_data
            )
        )

        comment_text = (
            build_comment_prompt_text(
                json_data
            )
        )

        # ----------------------------------------------------
        # 5. Меняем статус
        # ----------------------------------------------------

        bot.edit_message_text(
            "🚀 Анализ готов.\n\n"
            "Загружаю фото и публикую "
            "пост в VK...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        # ----------------------------------------------------
        # 6. VK
        # ----------------------------------------------------

        post_link = post_to_vk(
            vk_image_bytes,
            wall_text,
            comment_text
        )

        # ----------------------------------------------------
        # 7. Ответ пользователю
        # ----------------------------------------------------

        bot.reply_to(
            message,
            f"✅ Пост опубликован!\n\n"
            f"🔗 Ссылка на запись:\n"
            f"{post_link}\n\n"
            f"📝 Текст поста:\n"
            f"{wall_text}\n\n"
            f"💬 Промпт:\n\n"
            f"{comment_text}"
        )

    except Exception as e:

        error_text = str(e)

        print(
            "\n"
            "================ ERROR ================\n"
        )

        print(
            error_text
        )

        print(
            "========================================\n"
        )

        try:

            bot.edit_message_text(
                f"❌ Произошла ошибка:\n\n"
                f"{error_text}",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

        except Exception:

            bot.reply_to(
                message,
                f"❌ Произошла ошибка:\n\n"
                f"{error_text}"
            )


# ============================================================
# ЗАПУСК
# ============================================================

print(
    "================================================"
)

print(
    "✅ БОТ ЗАПУЩЕН"
)

print(
    f"Gemini: {GEMINI_MODEL}"
)

print(
    f"Gemini timeout: {GEMINI_TIMEOUT} сек."
)

print(
    f"VK API: {VK_API_VERSION}"
)

print(
    f"VK timeout: {VK_TIMEOUT} сек."
)

print(
    "Ожидаю фотографию..."
)

print(
    "================================================"
)


# ============================================================
# TELEGRAM POLLING
# ============================================================

while True:

    try:

        bot.infinity_polling(
            timeout=120,
            long_polling_timeout=120,
            skip_pending=True
        )

    except Exception as e:

        print(
            "[Telegram] Ошибка polling:"
        )

        print(
            e
        )

        print(
            "Переподключение через 5 секунд..."
        )

        time.sleep(5)
