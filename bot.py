import os
import io
import json
import time

import requests
import telebot
import google.generativeai as genai
from PIL import Image


# ============================================================
# НАСТРОЙКИ
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN")
VK_GROUP_ID = os.getenv("VK_GROUP_ID")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN")

if not GEMINI_API_KEY:
    raise RuntimeError("Не задан GEMINI_API_KEY")

if not VK_ACCESS_TOKEN:
    raise RuntimeError("Не задан VK_ACCESS_TOKEN")

if not VK_GROUP_ID:
    raise RuntimeError("Не задан VK_GROUP_ID")


# ============================================================
# GEMINI
# ============================================================

genai.configure(
    api_key=GEMINI_API_KEY
)

gemini_model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
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
# VK API
# ============================================================

VK_API_VERSION = "5.131"
VK_TIMEOUT = 120


def vk_call(method_name, params=None, data=None):
    """
    Рабочая логика:
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

    try:

        result = response.json()

    except ValueError:

        raise Exception(
            f"VK вернул не JSON в {clean_method}:\n"
            f"{response.text[:3000]}"
        )

    return result


# ============================================================
# ОЧИСТКА JSON
# ============================================================

def clean_json_response(raw_text):

    if not raw_text:
        raise Exception(
            "Gemini вернул пустой ответ."
        )

    cleaned = raw_text.strip()

    if cleaned.startswith("```"):

        lines = cleaned.split("\n")

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].startswith("```"):
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

Выдай ответ СТРОГО в формате JSON без вводных слов
и без разметки markdown.

Используй следующие ключи:

{
  "photo_title": "Ёмкое, красивое и точное название кратко только суть, которое отражает конкретный сюжет, кто изображён и где происходит действие"⚠️Берешь промпт — обязательно ставь лайк ❤️ и оставляй свой результат генерации этого фото в комментарии! 
  "photo_style": "Определи стиль фото по самому изображению",
  "camera_and_settings": "Предполагаемая камера, объектив и параметры съёмки",
  "shot_type_and_pose_intro": "Ракурс, ориентация кадра, положение тела и взгляд",
  "hairstyle_and_makeup": "Детальное описание причёски, бровей, макияжа и состояния кожи",
  "outfit": "Детальное описание одежды, кроя, фасона, ткани, обуви и аксессуаров",
  "pose_details": "Детальное описание позы, пластики тела, расположения рук и пальцев",
  "lighting": "Описание освещения, типа света, источников и теней",
  "background": "Описание фона и окружения",
  "color_grading_and_style": "Описание цветокоррекции, контраста, палитры и обработки",
  "quality_and_style_tags": "Параметры качества и стилистические теги",
  "hashtags": "5-8 целевых хэштегов через пробел"
}

photo_title:
Красивое краткое и точное название изображения.

photo_style:
Определи визуальный стиль фотографии.

camera_and_settings:
Укажи реалистичные предполагаемые параметры камеры,
объектива, диафрагмы, ISO и выдержки.

shot_type_and_pose_intro:
Укажи тип кадра, ориентацию, ракурс,
положение тела и направление взгляда.

hairstyle_and_makeup:
Опиши волосы, причёску, брови, глаза, губы,
макияж и состояние кожи.

outfit:
Опиши одежду максимально подробно:
цвет, материал, фасон, крой, обувь и аксессуары.

pose_details:
Опиши положение головы, корпуса, рук,
пальцев, ног и тела.

lighting:
Подробно опиши свет, его направление,
жёсткость, источники и тени.

background:
Подробно опиши окружение и фон.

color_grading_and_style:
Опиши цветовую палитру, контраст,
насыщенность и цветокоррекцию.

quality_and_style_tags:
Укажи степень реализма, детализацию,
естественную анатомию, реалистичность кожи,
отсутствие или присутствие CGI и артефактов.

hashtags:
5-8 релевантный актуальных хэштегов через пробел.

Если на изображении есть текст,
обязательно укажи:
- сам текст;
- язык;
- расположение;
- стиль текста.

Не придумывай элементы, которых нет на изображении.
"""

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

    except Exception as e:

        raise Exception(
            f"Не удалось открыть изображение: {e}"
        )

    last_error = None

    for attempt in range(3):

        try:

            print(
                f"[Gemini] Анализ "
                f"{attempt + 1}/3"
            )

            response = gemini_model.generate_content(
                [
                    system_prompt,
                    image
                ],
                request_options={
                    "timeout": 120
                }
            )

            clean_text = clean_json_response(
                response.text
            )

            data = json.loads(
                clean_text
            )

            print(
                "[Gemini] Анализ успешно получен."
            )

            return data

        except Exception as e:

            last_error = e

            print(
                "[Gemini] Ошибка:",
                e
            )

            if attempt < 2:

                time.sleep(
                    [3, 7, 15][attempt]
                )

    raise Exception(
        "Gemini не смог проанализировать "
        f"фотографию: {last_error}"
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

    return f"""⚠️Берешь промпт — обязательно ставь лайк ❤️ на пост и делись результом вашей генерации в комментарии!

КАК СОЗДАТЬ ФОТО С ПОМОЩЬЮ БОТОВ 🖤

🔹 БОТ 1 ВК — GPTron Nano Banana Pro 🍌✅
1️⃣ Переходим в бот:
https://vk.com/write-236453790?ref=pp53aacd7d52

🔹 БОТ 2 ВК — Lexy Nano Banana Pro 🍌✅
Переходим в бот:
https://vk.com/write-233546714?ref=84372609_add

Отправляем своё фото.
Выбираем модель генерации NANA BANANA PRO
Перед отправкой вставляем нужный промт в комментариях.

❗ Промт всегда можно и нужно менять под себя:
цвет волос, глаз, одежду, позу, настроение и т.д.

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{hashtags}"""


# ============================================================
# ПРОМПТ
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
    'Степень реализма, высокая детализация, '
    'естественная анатомия, реалистичность кожи, '
    'без или с CGI, присутствие мультяшности, без артефактов.'
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

    print(
        "[VK] Получаю upload_url..."
    )

    # --------------------------------------------------------
    # 1. Получаем сервер загрузки
    # --------------------------------------------------------

    server_res = vk_call(
        "photos.getWallUploadServer",
        params={
            "group_id": group_id,
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": VK_API_VERSION
        }
    )

    print(
        "[VK] getWallUploadServer:",
        server_res
    )

    if "error" in server_res:

        raise Exception(
            "Ошибка VK getWallUploadServer: "
            f"{server_res['error']}"
        )

    try:

        raw_upload_url = (
            server_res["response"]["upload_url"]
        )

    except (KeyError, TypeError):

        raise Exception(
            "VK не вернул upload_url:\n"
            f"{server_res}"
        )

    # Сохраняем рабочую обработку URL
    clean_upload_url = (
        raw_upload_url.split("](")[-1].rstrip(")")
        if "](" in raw_upload_url
        else raw_upload_url.strip("[]() ")
    )

    # --------------------------------------------------------
    # 2. Загружаем фото
    # --------------------------------------------------------

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
            f"Таймаут загрузки фото в VK "
            f"({VK_TIMEOUT} сек.)"
        )

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"Ошибка загрузки фото в VK: {e}"
        )

    try:

        upload_res = upload_response.json()

    except ValueError:

        raise Exception(
            "VK upload server вернул "
            "не JSON:\n"
            f"{upload_response.text[:3000]}"
        )

    print(
        "[VK] Upload response:",
        upload_res
    )

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

    # --------------------------------------------------------
    # 3. Проверяем photo/server/hash
    # --------------------------------------------------------

    server = upload_res.get(
        "server"
    )

    photo = upload_res.get(
        "photo"
    )

    vk_hash = upload_res.get(
        "hash"
    )

    if not server:

        raise Exception(
            "VK upload не вернул server:\n"
            f"{upload_res}"
        )

    if not vk_hash:

        raise Exception(
            "VK upload не вернул hash:\n"
            f"{upload_res}"
        )

    if (
        photo is None
        or str(photo).strip() == ""
        or photo == "[]"
        or photo == "undefined"
        or str(photo).lower() == "null"
    ):

        raise Exception(
            "VK upload не вернул корректный photo:\n"
            f"{upload_res}"
        )

    # --------------------------------------------------------
    # 4. Сохраняем фото
    # --------------------------------------------------------

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
        "[VK] saveWallPhoto:",
        save_res
    )

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

    except Exception:

        raise Exception(
            "VK не вернул данные "
            "сохранённой фотографии:\n"
            f"{save_res}"
        )

    # --------------------------------------------------------
    # 5. Публикуем пост
    # --------------------------------------------------------

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
        "[VK] wall.post:",
        post_res
    )

    if "error" in post_res:

        raise Exception(
            "Ошибка VK wall.post: "
            f"{post_res['error']}"
        )

    try:

        post_id = (
            post_res["response"]["post_id"]
        )

    except Exception:

        raise Exception(
            "VK не вернул post_id:\n"
            f"{post_res}"
        )

    # --------------------------------------------------------
    # 6. Публикуем комментарий
    # --------------------------------------------------------

    print(
        "[VK] Публикую промпт..."
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

    # Если from_group=group_id не работает
    # пробуем from_group=1

    if "error" in comment_res:

        print(
            "[VK] Первая попытка комментария "
            "не удалась:"
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
            "✅ Промпт успешно опубликован "
            "в комментарии."
        )

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
        "Отправь мне фотографию, и я:\n\n"
        "1️⃣ Проанализирую её через Gemini 3.6 Flash\n"
        "2️⃣ Создам название и хэштеги\n"
        "3️⃣ Сформирую готовый промпт\n"
        "4️⃣ Опубликую фото в VK\n"
        "5️⃣ Оставлю промпт первым комментарием"
    )


# ============================================================
# TELEGRAM / PHOTO
# ============================================================

@bot.message_handler(
    content_types=["photo"]
)
def handle_photo(message):

    status_msg = bot.reply_to(
        message,
        "⏳ Анализирую сюжет фото "
        "через Gemini 3.6 Flash..."
    )

    try:

        # ----------------------------------------------------
        # 1. Получаем file_id
        # ----------------------------------------------------

        file_id = (
            message.photo[-1].file_id
        )

        # ----------------------------------------------------
        # 2. Скачиваем фото
        # ----------------------------------------------------

        print(
            "[Telegram] Получаю фотографию..."
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
                "фотографию."
            )

        print(
            "[Telegram] Фото получено:",
            len(downloaded_file),
            "байт"
        )

        # ----------------------------------------------------
        # 3. Gemini
        # ----------------------------------------------------

        json_data = (
            analyze_photo_with_gemini(
                downloaded_file
            )
        )

        # ----------------------------------------------------
        # 4. Формируем тексты
        # ----------------------------------------------------

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
        # 5. Статус
        # ----------------------------------------------------

        bot.edit_message_text(
            "🚀 Анализ готов.\n\n"
            "Публикую фото и промпт "
            "в VK...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        # ----------------------------------------------------
        # 6. VK
        # ----------------------------------------------------

        post_link = post_to_vk(
            downloaded_file,
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
            f"💬 Промпт в комментарии:\n\n"
            f"{comment_text}"
        )

    except Exception as e:

        print(
            "\n"
            "================ ERROR ================\n"
        )

        print(
            e
        )

        print(
            "========================================\n"
        )

        try:

            bot.edit_message_text(
                "❌ Произошла ошибка:\n\n"
                f"{e}",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

        except Exception:

            bot.reply_to(
                message,
                f"❌ Произошла ошибка:\n\n{e}"
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
    "Gemini: gemini-3.6-flash"
)

print(
    f"VK API: {VK_API_VERSION}"
)

print(
    f"Gemini timeout: 120 сек."
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

        print(
            "📡 Подключение к Telegram..."
        )

        bot.infinity_polling(
            timeout=30,
            long_polling_timeout=30,
            skip_pending=True
        )

    except Exception as e:

        print(
            "[Telegram] Ошибка polling:",
            e
        )

        print(
            "🔄 Повторное подключение "
            "через 5 секунд..."
        )

        time.sleep(5)
