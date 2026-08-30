import os
import io
import json
import time
import base64
import traceback

import requests
import telebot

from PIL import Image
from google import genai
from google.genai import types


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
# МОДЕЛИ
# ============================================================

# Текст / анализ / JSON / сравнение
TEXT_MODEL = "gemini-3.6-flash"

# Nano Banana Pro
IMAGE_MODEL = "gemini-3-pro-image"


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

GEMINI_TIMEOUT = 180
VK_TIMEOUT = 120

# Максимальное количество черновиков
MAX_DRAFT_ATTEMPTS = 3

# Минимальное соответствие для перехода к 2K
MATCH_THRESHOLD = 90

# Паузы после временной ошибки
RETRY_DELAYS = [3, 7, 15]


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=GEMINI_TIMEOUT * 1000
    )
)


# ============================================================
# TELEGRAM
# ============================================================

telebot.apihelper.RETRY_ON_ERROR = True

bot = telebot.TeleBot(
    TELEGRAM_BOT_TOKEN
)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def clean_json_response(raw_text):
    """
    Удаляет markdown-обёртку ```json ... ```
    """

    if not raw_text:
        raise Exception(
            "Gemini вернул пустой ответ."
        )

    text = raw_text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def parse_json(text):
    """
    Преобразование строки Gemini в Python dict.
    """

    cleaned = clean_json_response(text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as e:

        raise Exception(
            "Gemini вернул некорректный JSON.\n\n"
            f"{cleaned[:6000]}"
        ) from e


def get_image_mime_type(image_bytes):
    """
    Определяет MIME-тип изображения.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        fmt = (
            image.format or "JPEG"
        ).upper()

        mapping = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }

        return mapping.get(
            fmt,
            "image/jpeg"
        )

    except Exception:

        return "image/jpeg"


def normalize_image_for_ai(image_bytes):
    """
    Преобразует входное изображение
    в качественный JPEG.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        print(
            "[IMAGE] Формат:",
            image.format
        )

        print(
            "[IMAGE] Размер:",
            image.size
        )

        rgb = image.convert("RGB")

        buffer = io.BytesIO()

        rgb.save(
            buffer,
            format="JPEG",
            quality=95
        )

        result = buffer.getvalue()

        print(
            "[IMAGE] JPEG:",
            len(result),
            "байт"
        )

        return result

    except Exception as e:

        raise Exception(
            f"Ошибка обработки изображения: {e}"
        ) from e


def get_aspect_ratio(image_bytes):
    """
    Определяет ближайшее стандартное соотношение сторон.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        width, height = image.size

        ratio = width / height

        candidates = {
            "1:1": 1.0,
            "4:5": 0.8,
            "3:4": 0.75,
            "2:3": 0.6667,
            "9:16": 0.5625,
            "5:4": 1.25,
            "4:3": 1.3333,
            "3:2": 1.5,
            "16:9": 1.7778,
            "21:9": 2.3333
        }

        best = min(
            candidates.items(),
            key=lambda item: abs(
                item[1] - ratio
            )
        )

        return best[0]

    except Exception:

        return "4:5"


# ============================================================
# GEMINI 3.6
# АНАЛИЗ РЕФЕРЕНСА
# ============================================================

def analyze_reference(image_bytes):

    prompt = """
Ты — профессиональный visual analyst и prompt engineer
для фотореалистичной генерации изображений.

Проанализируй прикреплённое изображение максимально подробно.

Главная задача:
не придумать новое изображение,
а восстановить максимально точное техническое описание
того, что уже находится в референсе.

Особенно внимательно анализируй:

1. Композицию.
2. Кадрирование.
3. Соотношение сторон.
4. Размер и положение главного объекта.
5. Положение человека или людей.
6. Направление взгляда.
7. Положение головы.
8. Положение корпуса.
9. Положение рук.
10. Положение пальцев.
11. Положение ног.
12. Причёску.
13. Выражение лица.
14. Макияж.
15. Одежду.
16. Материалы одежды.
17. Аксессуары.
18. Фон.
19. Предметы вокруг.
20. Перспективу.
21. Глубину резкости.
22. Освещение.
23. Направление света.
24. Тени.
25. Цветовую палитру.
26. Цветокоррекцию.
27. Атмосферу.
28. Предполагаемую камеру и объектив.
29. Возможные параметры съёмки.
30. Текст, логотипы и надписи.

Если точные параметры камеры неизвестны,
укажи реалистичное предположение,
но не выдавай предположение за достоверный факт.

Если есть текст,
обязательно укажи:
- содержание;
- язык;
- расположение;
- визуальный стиль текста.

НЕ придумывай элементов,
которых нет на изображении.

Верни ТОЛЬКО JSON.
"""

    mime_type = get_image_mime_type(
        image_bytes
    )

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    last_error = None

    for attempt in range(3):

        try:

            print(
                f"[Gemini 3.6] Анализ "
                f"{attempt + 1}/3"
            )

            response = (
                client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=[
                        image_part,
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            )

            data = parse_json(
                response.text
            )

            print(
                "[Gemini 3.6] Анализ готов."
            )

            return data

        except Exception as e:

            last_error = e

            print(
                "[Gemini 3.6] Ошибка:",
                e
            )

            if attempt < 2:

                time.sleep(
                    RETRY_DELAYS[attempt]
                )

    raise Exception(
        "Не удалось получить анализ референса.\n"
        f"Последняя ошибка: {last_error}"
    )


# ============================================================
# ПОСТРОЕНИЕ СТАНДАРТНОГО ПРОМПТА
# ============================================================

def build_standard_prompt(data):

    composition = data.get(
        "composition",
        {}
    )

    subject = data.get(
        "subject",
        {}
    )

    face = data.get(
        "face_and_expression",
        {}
    )

    hair = data.get(
        "hair",
        {}
    )

    outfit = data.get(
        "outfit",
        {}
    )

    pose = data.get(
        "pose",
        {}
    )

    environment = data.get(
        "environment",
        {}
    )

    lighting = data.get(
        "lighting",
        {}
    )

    camera = data.get(
        "camera",
        {}
    )

    color = data.get(
        "color",
        {}
    )

    text_info = data.get(
        "text_in_image",
        {}
    )

    prompt = f"""
📌 ПРОМПТ ДЛЯ ГЕНЕРАЦИИ

Внешность человека должна максимально точно соответствовать
прикреплённому референсу.

Сохрани индивидуальный визуальный образ человека:
форму лица, пропорции, глаза, нос, губы, кожу, волосы,
возрастное впечатление, мимику и общее визуальное восприятие.

Не идеализируй лицо.
Не меняй типаж.
Не делай человека моложе или старше.
Не добавляй новые черты.

КЛЮЧЕВАЯ ЗАДАЧА:
максимально точно воспроизвести референс,
а не создать просто похожую сцену.

СЮЖЕТ:
{data.get("photo_title", "")}

КОМПОЗИЦИЯ:
Соотношение сторон: {composition.get("aspect_ratio", "")}
Тип кадра: {composition.get("shot_type", "")}
Кадрирование: {composition.get("framing", "")}
Ракурс камеры: {composition.get("camera_angle", "")}
Положение объекта: {composition.get("subject_position", "")}
Перспектива: {composition.get("perspective", "")}

ГЛАВНЫЙ ОБЪЕКТ:
Количество: {subject.get("count", "")}
Описание: {subject.get("description", "")}
Положение: {subject.get("position", "")}
Размер в кадре: {subject.get("scale_in_frame", "")}

ЛИЦО И ВЫРАЖЕНИЕ:
Положение головы: {face.get("head_position", "")}
Взгляд: {face.get("gaze", "")}
Выражение лица: {face.get("expression", "")}
Макияж: {face.get("makeup", "")}
Кожа: {face.get("skin", "")}

ВОЛОСЫ:
Цвет: {hair.get("color", "")}
Длина: {hair.get("length", "")}
Причёска: {hair.get("style", "")}
Детали: {hair.get("details", "")}

ОДЕЖДА:
Описание: {outfit.get("description", "")}
Цвета: {outfit.get("colors", "")}
Материалы: {outfit.get("materials", "")}
Обувь: {outfit.get("shoes", "")}
Аксессуары: {outfit.get("accessories", "")}

ПОЗА:
Корпус: {pose.get("body", "")}
Голова: {pose.get("head", "")}
Левая рука: {pose.get("left_arm", "")}
Правая рука: {pose.get("right_arm", "")}
Левая кисть: {pose.get("left_hand", "")}
Правая кисть: {pose.get("right_hand", "")}
Ноги: {pose.get("legs", "")}
Стопы: {pose.get("feet", "")}

ОКРУЖЕНИЕ:
Локация: {environment.get("location", "")}
Фон: {environment.get("background", "")}
Передний план: {environment.get("foreground", "")}
Предметы: {environment.get("objects", "")}

ОСВЕЩЕНИЕ:
Тип: {lighting.get("type", "")}
Источник: {lighting.get("source", "")}
Направление: {lighting.get("direction", "")}
Жёсткость: {lighting.get("hardness", "")}
Тени: {lighting.get("shadows", "")}
Контровой свет: {lighting.get("rim_light", "")}

КАМЕРА:
Тип: {camera.get("camera_type", "")}
Объектив: {camera.get("lens", "")}
Диафрагма: {camera.get("aperture", "")}
ISO: {camera.get("iso", "")}
Выдержка: {camera.get("shutter_speed", "")}
Глубина резкости: {camera.get("depth_of_field", "")}

ЦВЕТОКОРРЕКЦИЯ:
Палитра: {color.get("palette", "")}
Грейдинг: {color.get("grading", "")}
Контраст: {color.get("contrast", "")}
Насыщенность: {color.get("saturation", "")}
Баланс белого: {color.get("white_balance", "")}

ТЕКСТ НА ИЗОБРАЖЕНИИ:
Есть: {text_info.get("present", False)}
Язык: {text_info.get("language", "")}
Содержание: {text_info.get("content", "")}
Положение: {text_info.get("position", "")}
Стиль: {text_info.get("style", "")}

ОБЩИЙ СТИЛЬ:
{data.get("style", "")}

КАЧЕСТВО:
{data.get("quality", "")}

СОХРАНИТЬ ОБЯЗАТЕЛЬНО:
- исходную композицию;
- исходное кадрирование;
- исходный ракурс;
- исходную перспективу;
- положение объектов;
- позу;
- одежду;
- фон;
- освещение;
- цветовую логику;
- масштаб объекта в кадре.

Не добавляй элементы,
которых нет на референсе.

Не удаляй важные элементы.

Максимальный фотореализм.
Реалистичная кожа.
Естественная анатомия.
Высокая детализация.
Без CGI.
Без мультяшности.
Без деформаций.
Без лишних пальцев.
Без лишних конечностей.
"""

    return prompt.strip()


# ============================================================
# NANO BANANA PRO
# ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЯ
# ============================================================

def generate_image_nano_banana(
    reference_bytes,
    prompt,
    image_size="1K"
):
    """
    Nano Banana Pro.
    Используем актуальный Interactions API.
    """

    aspect_ratio = get_aspect_ratio(
        reference_bytes
    )

    mime_type = get_image_mime_type(
        reference_bytes
    )

    encoded_reference = (
        base64.b64encode(
            reference_bytes
        ).decode("utf-8")
    )

    generation_prompt = f"""
Используй прикреплённое изображение
как ГЛАВНЫЙ визуальный референс.

Создай новое изображение,
максимально точно воспроизводящее исходный референс.

Главный приоритет:
визуальное соответствие оригиналу.

Сохрани:
- композицию;
- кадрирование;
- ракурс;
- перспективу;
- положение человека;
- позу;
- руки;
- пальцы;
- одежду;
- волосы;
- фон;
- предметы;
- освещение;
- тени;
- цвет;
- масштаб объекта.

Не добавляй новые предметы.
Не удаляй важные предметы.
Не меняй сцену без необходимости.

ПРОМПТ:

{prompt}

Создай максимально реалистичную фотографию.
"""

    last_error = None

    for attempt in range(3):

        try:

            print(
                f"[Nano Banana Pro] "
                f"{image_size}, "
                f"попытка {attempt + 1}/3"
            )

            interaction = (
                client.interactions.create(
                    model=IMAGE_MODEL,
                    input=[
                        {
                            "type": "text",
                            "text": generation_prompt
                        },
                        {
                            "type": "image",
                            "data": encoded_reference,
                            "mime_type": mime_type
                        }
                    ],
                    response_format={
                        "type": "image",
                        "mime_type": "image/jpeg",
                        "aspect_ratio": aspect_ratio,
                        "image_size": image_size
                    },
                    timeout=GEMINI_TIMEOUT
                )
            )

            generated = (
                interaction.output_image
            )

            if not generated:
                raise Exception(
                    "Nano Banana Pro "
                    "не вернул output_image."
                )

            if not generated.data:
                raise Exception(
                    "Nano Banana Pro "
                    "вернул пустые данные изображения."
                )

            image_bytes = (
                base64.b64decode(
                    generated.data
                )
            )

            if not image_bytes:
                raise Exception(
                    "Получено пустое изображение."
                )

            print(
                "[Nano Banana Pro] "
                f"Изображение получено: "
                f"{len(image_bytes)} байт"
            )

            return image_bytes

        except Exception as e:

            last_error = e

            print(
                "[Nano Banana Pro] Ошибка:",
                e
            )

            if attempt < 2:

                time.sleep(
                    RETRY_DELAYS[attempt]
                )

    raise Exception(
        "Nano Banana Pro не смог создать "
        f"изображение.\n"
        f"Последняя ошибка: {last_error}"
    )


# ============================================================
# GEMINI 3.6
# СРАВНЕНИЕ РЕФЕРЕНСА И ЧЕРНОВИКА
# ============================================================

def compare_reference_and_draft(
    reference_bytes,
    draft_bytes,
    current_prompt
):

    reference_mime = get_image_mime_type(
        reference_bytes
    )

    draft_mime = get_image_mime_type(
        draft_bytes
    )

    reference_part = types.Part.from_bytes(
        data=reference_bytes,
        mime_type=reference_mime
    )

    draft_part = types.Part.from_bytes(
        data=draft_bytes,
        mime_type=draft_mime
    )

    comparison_prompt = f"""
Ты — эксперт по визуальному контролю качества
AI-generated изображений.

IMAGE 1 = ОРИГИНАЛЬНЫЙ РЕФЕРЕНС.
IMAGE 2 = СГЕНЕРИРОВАННЫЙ ЧЕРНОВИК.

Сравни изображения.

Главная задача:
понять, насколько IMAGE 2 воспроизводит IMAGE 1.

Оцени:

1. Общую композицию.
2. Кадрирование.
3. Соотношение сторон.
4. Положение главного объекта.
5. Размер объекта в кадре.
6. Ракурс.
7. Перспективу.
8. Положение головы.
9. Положение корпуса.
10. Руки.
11. Пальцы.
12. Ноги.
13. Одежду.
14. Волосы.
15. Выражение лица.
16. Фон.
17. Предметы.
18. Освещение.
19. Тени.
20. Цветовую палитру.
21. Цветокоррекцию.
22. Текст и логотипы.
23. Общую геометрию.

Не штрафуй за микроскопические различия,
если главная композиция и сцена совпадают.

Верни строго JSON:

{{
    "overall_score": 0,
    "composition_score": 0,
    "framing_score": 0,
    "pose_score": 0,
    "clothing_score": 0,
    "background_score": 0,
    "lighting_score": 0,
    "color_score": 0,
    "detail_score": 0,
    "errors": [],
    "corrections": [],
    "improved_prompt": ""
}}

Каждая оценка от 0 до 100.

overall_score:
общая степень визуального соответствия.

errors:
только реально заметные расхождения.

corrections:
точные исправления.

improved_prompt:
полный обновлённый промпт.
Не удаляй уже правильные детали.
Исправляй только найденные несоответствия.

ТЕКУЩИЙ ПРОМПТ:

{current_prompt}
"""

    last_error = None

    for attempt in range(3):

        try:

            print(
                f"[Gemini 3.6] Проверка "
                f"{attempt + 1}/3"
            )

            response = (
                client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=[
                        reference_part,
                        draft_part,
                        comparison_prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            )

            result = parse_json(
                response.text
            )

            print(
                "[Gemini 3.6] "
                "Проверка завершена. "
                f"Оценка: "
                f"{result.get('overall_score')}"
            )

            return result

        except Exception as e:

            last_error = e

            print(
                "[Gemini 3.6] "
                f"Ошибка проверки: {e}"
            )

            if attempt < 2:
                time.sleep(
                    RETRY_DELAYS[attempt]
                )

    raise Exception(
        "Не удалось проверить "
        f"черновик: {last_error}"
    )


# ============================================================
# VK API
# ============================================================

def safe_json_response(
    response,
    source_name
):

    try:

        return response.json()

    except ValueError:

        raise Exception(
            f"{source_name} вернул не JSON.\n"
            f"HTTP: {response.status_code}\n"
            f"Ответ:\n"
            f"{response.text[:3000]}"
        )


def vk_call(
    method_name,
    params=None,
    data=None
):

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
            f"VK timeout в {clean_method}"
        )

    except requests.exceptions.ConnectionError as e:

        raise Exception(
            f"Ошибка соединения с VK: {e}"
        )

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"HTTP ошибка VK: {e}"
        )

    return safe_json_response(
        response,
        f"VK {clean_method}"
    )


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

    # --------------------------------------------------------
    # 1. Получаем upload_url
    # --------------------------------------------------------

    print(
        "[VK] Получаю upload_url..."
    )

    server_res = vk_call(
        "photos.getWallUploadServer",
        params={
            "group_id": group_id,
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": "5.131"
        }
    )

    print(
        "[VK] getWallUploadServer:",
        server_res
    )

    if "error" in server_res:

        raise Exception(
            f"VK getWallUploadServer: "
            f"{server_res['error']}"
        )

    if (
        "response" not in server_res
        or "upload_url" not in server_res["response"]
    ):

        raise Exception(
            "VK не вернул upload_url:\n"
            f"{server_res}"
        )

    upload_url = (
        server_res["response"]["upload_url"]
    )

    # --------------------------------------------------------
    # 2. Загружаем фотографию
    # --------------------------------------------------------

    files = {
        "photo": (
            "photo.jpg",
            image_bytes,
            "image/jpeg"
        )
    }

    print(
        "[VK] Загружаю изображение..."
    )

    try:

        upload_response = requests.post(
            upload_url,
            files=files,
            timeout=VK_TIMEOUT
        )

    except requests.exceptions.Timeout:

        raise Exception(
            "VK upload timeout."
        )

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"Ошибка загрузки фото VK: {e}"
        )

    upload_res = safe_json_response(
        upload_response,
        "VK upload server"
    )

    print(
        "[VK] Upload response:",
        upload_res
    )

    if "error" in upload_res:

        raise Exception(
            f"VK upload error: "
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
            "VK upload не вернул корректный photo.\n\n"
            f"Ответ VK:\n{upload_res}"
        )

    # --------------------------------------------------------
    # 3. Save photo
    # --------------------------------------------------------

    print(
        "[VK] Сохраняю фотографию..."
    )

    save_res = vk_call(
        "photos.saveWallPhoto",
        data={
            "group_id": group_id,
            "server": server,
            "photo": photo,
            "hash": vk_hash,
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": "5.131"
        }
    )

    print(
        "[VK] saveWallPhoto:",
        save_res
    )

    if "error" in save_res:

        raise Exception(
            f"VK saveWallPhoto: "
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
            f"сохранённой фотографии:\n"
            f"{save_res}"
        ) from e

    # --------------------------------------------------------
    # 4. wall.post
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
            "v": "5.131"
        }
    )

    print(
        "[VK] wall.post:",
        post_res
    )

    if "error" in post_res:

        raise Exception(
            f"VK wall.post: "
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

    # --------------------------------------------------------
    # 5. Комментарий
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
            "v": "5.131"
        }
    )

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
                "v": "5.131"
            }
        )

    if "error" in comment_res:

        print(
            "⚠️ Комментарий не опубликован:",
            comment_res["error"]
        )

    else:

        print(
            "✅ Промпт опубликован."
        )

    return (
        f"https://vk.com/wall-"
        f"{group_id}_{post_id}"
    )


# ============================================================
# ПОСТ VK
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

ИНСТРУКЦИЯ

КАК СОЗДАТЬ ФОТО С ПОМОЩЬЮ БОТОВ 🖤

🔹 БОТ 1 ВК — GPTron Nano Banana Pro 🍌✅
1️⃣ Переходим в бот:
https://vk.com/write-236453790?ref=pp53aacd7d52

🔹 БОТ 2 ВК — Lexy Nano Banana Pro 🍌✅
Переходим в бот:
https://vk.com/write-233546714?ref=84372609_add

Отправляем своё фото.

Перед отправкой вставляем нужный промт из комментария.

❗ Промт можно и нужно менять под себя:
цвет волос, глаз, одежду, позу, настроение и т.д.

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{hashtags}"""


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
        "Отправь мне понравившееся фото.\n\n"
        "Я:\n"
        "1️⃣ Изучу референс\n"
        "2️⃣ Создам JSON-анализ\n"
        "3️⃣ Сформирую стандартный промпт\n"
        "4️⃣ Создам черновик Nano Banana Pro 1K\n"
        "5️⃣ Проверю точность\n"
        "6️⃣ При необходимости исправлю промпт\n"
        "7️⃣ Создам финал Nano Banana Pro 2K\n"
        "8️⃣ Опубликую результат в VK"
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
        "⏳ Получил референс.\n"
        "Начинаю анализ..."
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
        # 2. Скачать
        # ----------------------------------------------------

        print(
            "[Telegram] Скачиваю изображение..."
        )

        file_info = bot.get_file(
            file_id
        )

        original_bytes = (
            bot.download_file(
                file_info.file_path
            )
        )

        if not original_bytes:

            raise Exception(
                "Не удалось скачать "
                "изображение."
            )

        print(
            "[Telegram] Получено:",
            len(original_bytes),
            "байт"
        )

        # ----------------------------------------------------
        # 3. Нормализация
        # ----------------------------------------------------

        reference_bytes = (
            normalize_image_for_ai(
                original_bytes
            )
        )

        # ----------------------------------------------------
        # 4. ANALYSIS
        # ----------------------------------------------------

        bot.edit_message_text(
            "🔍 Анализирую референс "
            "через Gemini 3.6 Flash...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        analysis_json = (
            analyze_reference(
                reference_bytes
            )
        )

        print(
            "[SYSTEM] JSON получен:"
        )

        print(
            json.dumps(
                analysis_json,
                ensure_ascii=False,
                indent=2
            )
        )

        # ----------------------------------------------------
        # 5. STANDARD PROMPT
        # ----------------------------------------------------

        bot.edit_message_text(
            "📝 JSON готов.\n"
            "Формирую стандартный промпт...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        current_prompt = (
            build_standard_prompt(
                analysis_json
            )
        )

        # ----------------------------------------------------
        # 6. Показываем JSON
        # ----------------------------------------------------

        json_text = json.dumps(
            analysis_json,
            ensure_ascii=False,
            indent=2
        )

        if len(json_text) <= 3900:

            bot.send_message(
                message.chat.id,
                "📦 JSON-анализ:\n\n"
                f"{json_text}"
            )

        else:

            bot.send_message(
                message.chat.id,
                "📦 JSON-анализ создан.\n"
                "Полный JSON сохранён в логах Bothost."
            )

        # ----------------------------------------------------
        # 7. Показываем промпт
        # ----------------------------------------------------

        if len(current_prompt) <= 3900:

            bot.send_message(
                message.chat.id,
                "📝 Стандартный промпт:\n\n"
                f"{current_prompt}"
            )

        # ----------------------------------------------------
        # 8. DRAFT LOOP
        # ----------------------------------------------------

        successful_draft = None
        successful_check = None
        successful_prompt = (
            current_prompt
        )

        for draft_number in range(
            1,
            MAX_DRAFT_ATTEMPTS + 1
        ):

            bot.edit_message_text(
                f"🎨 Создаю черновик "
                f"Nano Banana Pro 1K...\n\n"
                f"Попытка "
                f"{draft_number}/"
                f"{MAX_DRAFT_ATTEMPTS}",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

            # ------------------------------------------------
            # Generate 1K
            # ------------------------------------------------

            draft_bytes = (
                generate_image_nano_banana(
                    reference_bytes,
                    current_prompt,
                    image_size="1K"
                )
            )

            # ------------------------------------------------
            # Отправляем черновик в Telegram
            # ------------------------------------------------

            bot.send_photo(
                message.chat.id,
                draft_bytes,
                caption=(
                    f"🖼 Черновик 1K\n"
                    f"Попытка {draft_number}"
                )
            )

            # ------------------------------------------------
            # Проверяем
            # ------------------------------------------------

            bot.edit_message_text(
                f"🔎 Проверяю черновик "
                f"{draft_number}...",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

            check_result = (
                compare_reference_and_draft(
                    reference_bytes,
                    draft_bytes,
                    current_prompt
                )
            )

            print(
                "[CHECK]"
            )

            print(
                json.dumps(
                    check_result,
                    ensure_ascii=False,
                    indent=2
                )
            )

            score = int(
                check_result.get(
                    "overall_score",
                    0
                )
            )

            # ------------------------------------------------
            # Отчёт
            # ------------------------------------------------

            bot.send_message(
                message.chat.id,
                "📊 Результат проверки\n\n"
                f"Общее: {score}/100\n"
                f"Композиция: "
                f"{check_result.get('composition_score', 0)}/100\n"
                f"Кадрирование: "
                f"{check_result.get('framing_score', 0)}/100\n"
                f"Поза: "
                f"{check_result.get('pose_score', 0)}/100\n"
                f"Одежда: "
                f"{check_result.get('clothing_score', 0)}/100\n"
                f"Фон: "
                f"{check_result.get('background_score', 0)}/100\n"
                f"Свет: "
                f"{check_result.get('lighting_score', 0)}/100\n"
                f"Цвет: "
                f"{check_result.get('color_score', 0)}/100"
            )

            # ------------------------------------------------
            # УСПЕХ
            # ------------------------------------------------

            if score >= MATCH_THRESHOLD:

                successful_draft = (
                    draft_bytes
                )

                successful_check = (
                    check_result
                )

                successful_prompt = (
                    current_prompt
                )

                print(
                    f"[SYSTEM] "
                    f"Черновик принят: "
                    f"{score}/100"
                )

                break

            # ------------------------------------------------
            # НЕУДАЧА
            # ------------------------------------------------

            if draft_number < MAX_DRAFT_ATTEMPTS:

                improved_prompt = (
                    check_result
                    .get(
                        "improved_prompt",
                        ""
                    )
                    .strip()
                )

                if improved_prompt:

                    current_prompt = (
                        improved_prompt
                    )

                else:

                    corrections = (
                        check_result.get(
                            "corrections",
                            []
                        )
                    )

                    correction_text = (
                        "\n".join(
                            f"- {item}"
                            for item in corrections
                        )
                    )

                    current_prompt = (
                        current_prompt
                        + "\n\n"
                        "КРИТИЧЕСКИЕ "
                        "ИСПРАВЛЕНИЯ:\n"
                        + correction_text
                    )

                bot.send_message(
                    message.chat.id,
                    "🔄 Соответствие недостаточное.\n\n"
                    f"Результат: {score}/100\n"
                    "Исправляю промпт и "
                    "создаю следующий черновик."
                )

        # ====================================================
        # НЕ ПРОШЁЛ ПРОВЕРКУ
        # ====================================================

        if successful_draft is None:

            bot.edit_message_text(
                f"⚠️ Не удалось достичь "
                f"{MATCH_THRESHOLD}/100 "
                f"за {MAX_DRAFT_ATTEMPTS} попытки.\n\n"
                "Финальный 2K не создаю "
                "и в VK не публикую.",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

            if len(current_prompt) <= 3900:

                bot.send_message(
                    message.chat.id,
                    "📝 Последний вариант промпта:\n\n"
                    f"{current_prompt}"
                )

            return

        # ====================================================
        # FINAL 2K
        # ====================================================

        final_score = (
            successful_check.get(
                "overall_score",
                0
            )
        )

        bot.edit_message_text(
            "✅ Черновик принят.\n\n"
            f"Соответствие: "
            f"{final_score}/100\n\n"
            "🍌 Создаю финальное изображение "
            "Nano Banana Pro 2K...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        final_bytes = (
            generate_image_nano_banana(
                reference_bytes,
                successful_prompt,
                image_size="2K"
            )
        )

        # ----------------------------------------------------
        # Отправляем финальный результат
        # ----------------------------------------------------

        bot.send_photo(
            message.chat.id,
            final_bytes,
            caption=(
                "✅ Финал Nano Banana Pro 2K\n"
                f"Проверка черновика: "
                f"{final_score}/100"
            )
        )

        # ====================================================
        # VK
        # ====================================================

        wall_text = (
            build_wall_post_text(
                analysis_json
            )
        )

        comment_text = (
            "📌 Промпт для генерации:\n\n"
            + successful_prompt
        )

        bot.edit_message_text(
            "🚀 Финал готов.\n\n"
            "Публикую изображение "
            "и промпт в VK...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        post_link = post_to_vk(
            final_bytes,
            wall_text,
            comment_text
        )

        # ====================================================
        # ГОТОВО
        # ====================================================

        bot.edit_message_text(
            "🎉 Готово!\n\n"
            "✅ Референс проанализирован\n"
            "✅ JSON создан\n"
            "✅ Стандартный промпт создан\n"
            "✅ Черновик 1K создан\n"
            f"✅ Проверка: {final_score}/100\n"
            "✅ Финал 2K создан\n"
            "✅ Опубликовано в VK\n\n"
            f"🔗 {post_link}",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
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

        traceback.print_exc()

        print(
            "========================================\n"
        )

        try:

            bot.edit_message_text(
                "❌ Произошла ошибка:\n\n"
                f"{error_text}",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

        except Exception:

            bot.reply_to(
                message,
                "❌ Произошла ошибка:\n\n"
                f"{error_text}"
            )


# ============================================================
# ЗАПУСК
# ============================================================

print(
    "=================================================="
)

print(
    "✅ БОТ ЗАПУЩЕН"
)

print(
    f"Text model: {TEXT_MODEL}"
)

print(
    f"Image model: {IMAGE_MODEL}"
)

print(
    f"Gemini timeout: "
    f"{GEMINI_TIMEOUT} сек."
)

print(
    f"VK timeout: "
    f"{VK_TIMEOUT} сек."
)

print(
    f"Порог соответствия: "
    f"{MATCH_THRESHOLD}/100"
)

print(
    "Ожидаю референс..."
)

print(
    "=================================================="
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
            skip_pending=True,
            allowed_updates=["message"]
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
