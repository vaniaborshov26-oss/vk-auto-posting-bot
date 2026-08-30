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

TEXT_MODEL = "gemini-3.6-flash"

# Nano Banana Pro
IMAGE_MODEL = "gemini-3-pro-image"

# ============================================================
# ПАРАМЕТРЫ
# ============================================================

GEMINI_TIMEOUT = 180
VK_TIMEOUT = 120

MAX_DRAFT_ATTEMPTS = 3

# Минимальный балл для перехода к финальной генерации
MATCH_THRESHOLD = 90

# Пауза между повторными запросами
RETRY_DELAYS = [3, 7, 15]


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
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
    Очищает ответ Gemini от markdown-обёртки.
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
    Парсит JSON и выдаёт понятную ошибку.
    """

    cleaned = clean_json_response(text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as e:

        raise Exception(
            "Gemini вернул некорректный JSON.\n\n"
            f"{cleaned[:5000]}"
        ) from e


def get_image_mime_type(image_bytes):
    """
    Определяет MIME изображения через PIL.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        fmt = (image.format or "JPEG").upper()

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
    Приводит референс к JPEG.
    Это уменьшает проблемы с WEBP/PNG и Telegram documents.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        print(
            "[IMAGE] Исходный формат:",
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
            "[IMAGE] Нормализованный JPEG:",
            len(result),
            "байт"
        )

        return result

    except Exception as e:

        raise Exception(
            f"Не удалось подготовить изображение: {e}"
        ) from e


def get_aspect_ratio(image_bytes):
    """
    Определяет примерное соотношение сторон.
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
        }

        closest = min(
            candidates.items(),
            key=lambda item: abs(item[1] - ratio)
        )

        return closest[0]

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

Проанализируй прикреплённый референс максимально подробно.

Главная задача:
не придумать новое изображение,
а восстановить максимально точное техническое описание
того, ЧТО УЖЕ находится в референсе.

Особенно внимательно анализируй:

1. Композицию.
2. Кадрирование.
3. Соотношение сторон.
4. Размер и положение главного объекта.
5. Положение человека/людей.
6. Направление взгляда.
7. Положение головы.
8. Положение корпуса.
9. Положение рук.
10. Положение пальцев.
11. Положение ног.
12. Причёску.
13. Черты внешнего образа без выдумывания новых деталей.
14. Макияж.
15. Одежду.
16. Материалы и текстуры одежды.
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
30. Любой текст, логотипы или надписи на изображении.

Если точные параметры камеры неизвестны,
укажи реалистичное предположение,
но НЕ выдавай предположение за достоверный факт.

Если в изображении присутствует текст,
обязательно укажи его содержание,
язык, расположение и визуальное оформление.

Верни ТОЛЬКО JSON.

Структура:

{
  "photo_title": "",
  "composition": {
    "aspect_ratio": "",
    "shot_type": "",
    "framing": "",
    "camera_angle": "",
    "subject_position": "",
    "perspective": ""
  },
  "subject": {
    "count": 0,
    "description": "",
    "position": "",
    "scale_in_frame": ""
  },
  "face_and_expression": {
    "head_position": "",
    "gaze": "",
    "expression": "",
    "makeup": "",
    "skin": ""
  },
  "hair": {
    "color": "",
    "length": "",
    "style": "",
    "details": ""
  },
  "outfit": {
    "description": "",
    "colors": "",
    "materials": "",
    "shoes": "",
    "accessories": ""
  },
  "pose": {
    "body": "",
    "head": "",
    "left_arm": "",
    "right_arm": "",
    "left_hand": "",
    "right_hand": "",
    "legs": "",
    "feet": ""
  },
  "environment": {
    "location": "",
    "background": "",
    "foreground": "",
    "objects": ""
  },
  "lighting": {
    "type": "",
    "source": "",
    "direction": "",
    "hardness": "",
    "shadows": "",
    "rim_light": ""
  },
  "camera": {
    "camera_type": "",
    "lens": "",
    "aperture": "",
    "iso": "",
    "shutter_speed": "",
    "depth_of_field": ""
  },
  "color": {
    "palette": "",
    "grading": "",
    "contrast": "",
    "saturation": "",
    "white_balance": ""
  },
  "text_in_image": {
    "present": false,
    "language": "",
    "content": "",
    "position": "",
    "style": ""
  },
  "style": "",
  "quality": "",
  "hashtags": ""
}
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
                f"[Gemini 3.6] Анализ: "
                f"{attempt + 1}/3"
            )

            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[
                    image_part,
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            data = parse_json(
                response.text
            )

            print(
                "[Gemini 3.6] JSON анализа получен."
            )

            return data

        except Exception as e:

            last_error = e

            print(
                "[Gemini 3.6] Ошибка анализа:",
                e
            )

            if attempt < 2:
                time.sleep(
                    RETRY_DELAYS[attempt]
                )

    raise Exception(
        f"Не удалось проанализировать референс: "
        f"{last_error}"
    )


# ============================================================
# СТАНДАРТНЫЙ ПРОМПТ
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
а не создать похожую сцену.

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
Количество объектов/людей: {subject.get("count", "")}
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
{outfit.get("description", "")}

Цвета одежды:
{outfit.get("colors", "")}

Материалы и фактура:
{outfit.get("materials", "")}

Обувь:
{outfit.get("shoes", "")}

Аксессуары:
{outfit.get("accessories", "")}

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
Тип света: {lighting.get("type", "")}
Источник: {lighting.get("source", "")}
Направление: {lighting.get("direction", "")}
Жёсткость: {lighting.get("hardness", "")}
Тени: {lighting.get("shadows", "")}
Контровой свет: {lighting.get("rim_light", "")}

КАМЕРА:
Камера: {camera.get("camera_type", "")}
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
Наличие: {text_info.get("present", False)}
Язык: {text_info.get("language", "")}
Содержание: {text_info.get("content", "")}
Положение: {text_info.get("position", "")}
Стиль текста: {text_info.get("style", "")}

СТИЛЬ:
{data.get("style", "")}

КАЧЕСТВО:
{data.get("quality", "")}

Сохрани оригинальную композицию, геометрию,
ракурс, кадрирование, расположение объектов,
позу, освещение, фон, масштаб объекта в кадре
и цветовую логику референса.

Не добавляй элементы, которых нет в референсе.
Не удаляй важные элементы референса.

Максимальный фотореализм.
Естественная анатомия.
Реалистичная кожа.
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

    aspect_ratio = get_aspect_ratio(
        reference_bytes
    )

    mime_type = get_image_mime_type(
        reference_bytes
    )

    image_part = types.Part.from_bytes(
        data=reference_bytes,
        mime_type=mime_type
    )

    generation_prompt = f"""
Используй прикреплённое изображение как ГЛАВНЫЙ визуальный референс.

Задача:
создать новое изображение, которое максимально точно
воспроизводит референс.

Не интерпретируй сцену свободно.
Не меняй композицию без необходимости.
Не меняй ракурс.
Не меняй положение главного объекта.
Не меняй позу.
Не меняй одежду.
Не меняй фон.
Не меняй освещение.
Не меняй цветовую логику.

Особенно точно сохрани:
- геометрию кадра;
- положение тела;
- положение головы;
- руки и пальцы;
- размеры объекта в кадре;
- кадрирование;
- перспективу;
- фон;
- предметы;
- свет;
- цвет.

ИНСТРУКЦИЯ:

{prompt}

Создай фотореалистичную фотографию.
"""

    last_error = None

    for attempt in range(3):

        try:

            print(
                f"[Nano Banana Pro] "
                f"Генерация {image_size}: "
                f"{attempt + 1}/3"
            )

            response = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=[
                    image_part,
                    generation_prompt
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    response_format={
                        "image": {
                            "aspect_ratio": aspect_ratio,
                            "image_size": image_size
                        }
                    }
                )
            )

            for part in response.parts:

                if part.inline_data:

                    image = part.as_image()

                    buffer = io.BytesIO()

                    image.save(
                        buffer,
                        format="JPEG",
                        quality=95
                    )

                    result = (
                        buffer.getvalue()
                    )

                    print(
                        "[Nano Banana Pro] "
                        f"Получено изображение: "
                        f"{len(result)} байт"
                    )

                    return result

            raise Exception(
                "Nano Banana Pro не вернул "
                "изображение."
            )

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
        "Nano Banana Pro не смог "
        f"создать изображение: {last_error}"
    )


# ============================================================
# GEMINI 3.6
# ПРОВЕРКА РЕЗУЛЬТАТА
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

    prompt = f"""
Ты — эксперт по визуальному контролю качества
AI-generated изображений.

Ниже два изображения:

IMAGE 1 = ОРИГИНАЛЬНЫЙ РЕФЕРЕНС
IMAGE 2 = СГЕНЕРИРОВАННЫЙ РЕЗУЛЬТАТ

Сравни их именно как изображения,
а не как художественные произведения.

Главная цель —
определить, насколько IMAGE 2 воспроизводит IMAGE 1.

Проверь:

1. Общую композицию.
2. Кадрирование.
3. Соотношение сторон.
4. Положение главного объекта.
5. Масштаб объекта.
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
22. Текст/логотипы.
23. Общую геометрию сцены.

Важно:
не штрафуй результат за естественные микроскопические
отличия деталей, если композиция и смысл сцены совпадают.

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

Шкала каждого показателя:
0-100.

overall_score:
общая оценка воспроизведения референса.

errors:
только реально заметные расхождения.

corrections:
конкретные инструкции, что изменить
в следующей генерации.

improved_prompt:
полный обновлённый промпт.
Сохрани в нём все уже правильные элементы
и измени только то, что необходимо.

ТЕКУЩИЙ ПРОМПТ:

{current_prompt}
"""

    last_error = None

    for attempt in range(3):

        try:

            print(
                f"[Gemini 3.6] "
                f"Проверка результата "
                f"{attempt + 1}/3"
            )

            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[
                    reference_part,
                    draft_part,
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            result = parse_json(
                response.text
            )

            print(
                "[Gemini 3.6] "
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
        f"Не удалось проверить результат: "
        f"{last_error}"
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
            f"Ответ: {response.text[:3000]}"
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

    except requests.exceptions.RequestException as e:

        raise Exception(
            f"Ошибка HTTP VK "
            f"{clean_method}: {e}"
        )

    return safe_json_response(
        response,
        f"VK {clean_method}"
    )


# ============================================================
# VK POST
# ============================================================

def post_to_vk(
    image_bytes,
    wall_text,
    comment_text
):

    group_id = int(
        str(VK_GROUP_ID).replace("-", "")
    )

    # --------------------------------------------------------
    # 1. upload server
    # --------------------------------------------------------

    server_res = vk_call(
        "photos.getWallUploadServer",
        params={
            "group_id": group_id,
            "access_token": VK_ACCESS_TOKEN.strip(),
            "v": "5.131"
        }
    )

    if "error" in server_res:

        raise Exception(
            "VK getWallUploadServer: "
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
    # 2. upload photo
    # --------------------------------------------------------

    files = {
        "photo": (
            "photo.jpg",
            image_bytes,
            "image/jpeg"
        )
    }

    try:

        upload_response = requests.post(
            upload_url,
            files=files,
            timeout=VK_TIMEOUT
        )

    except requests.exceptions.Timeout:

        raise Exception(
            "VK upload timeout"
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
        "[VK] Upload:",
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
            f"VK upload не вернул server:\n"
            f"{upload_res}"
        )

    if not vk_hash:
        raise Exception(
            f"VK upload не вернул hash:\n"
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
            "VK upload не вернул корректный photo.\n"
            f"Ответ:\n{upload_res}"
        )

    # --------------------------------------------------------
    # 3. save photo
    # --------------------------------------------------------

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

        owner_id = photo_info[
            "owner_id"
        ]

        photo_id = photo_info[
            "id"
        ]

    except Exception as e:

        raise Exception(
            "VK не вернул данные "
            f"сохранённой фотографии:\n"
            f"{save_res}"
        ) from e

    # --------------------------------------------------------
    # 4. wall.post
    # --------------------------------------------------------

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

    post_id = (
        post_res["response"]["post_id"]
    )

    # --------------------------------------------------------
    # 5. comment
    # --------------------------------------------------------

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
            "[VK] Повтор комментария "
            "с from_group=1"
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
            "✅ Промпт опубликован "
            "в комментарии."
        )

    return (
        f"https://vk.com/wall-"
        f"{group_id}_{post_id}"
    )


# ============================================================
# TELEGRAM START
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
        "5️⃣ Проверю его на соответствие\n"
        "6️⃣ При необходимости исправлю промпт\n"
        "7️⃣ Создам финал Nano Banana Pro 2K\n"
        "8️⃣ Опубликую результат в VK"
    )


# ============================================================
# TELEGRAM PHOTO HANDLER
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
        "Начинаю глубокий анализ..."
    )

    try:

        # ----------------------------------------------------
        # 1. Telegram file_id
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
                "Фотография не найдена."
            )

        # ----------------------------------------------------
        # 2. Download
        # ----------------------------------------------------

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
                "фото из Telegram."
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
            "🔍 Анализирую референс через "
            "Gemini 3.6 Flash...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )

        analysis_json = (
            analyze_reference(
                reference_bytes
            )
        )

        print(
            "[SYSTEM] JSON анализа:\n",
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
        # 6. Отправляем JSON пользователю
        # ----------------------------------------------------

        json_text = json.dumps(
            analysis_json,
            ensure_ascii=False,
            indent=2
        )

        # Telegram message limit
        if len(json_text) <= 3900:

            bot.send_message(
                message.chat.id,
                "📦 JSON-анализ:\n\n"
                f"{json_text}"
            )

        # ----------------------------------------------------
        # 7. Отправляем текущий промпт
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
        successful_prompt = current_prompt

        for draft_number in range(
            1,
            MAX_DRAFT_ATTEMPTS + 1
        ):

            bot.edit_message_text(
                f"🎨 Создаю черновик "
                f"Nano Banana Pro 1K...\n\n"
                f"Попытка: "
                f"{draft_number}/{MAX_DRAFT_ATTEMPTS}",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

            draft_bytes = (
                generate_image_nano_banana(
                    reference_bytes,
                    current_prompt,
                    image_size="1K"
                )
            )

            # ------------------------------------------------
            # Отправляем черновик пользователю
            # ------------------------------------------------

            bot.send_photo(
                message.chat.id,
                draft_bytes,
                caption=(
                    f"🖼 Черновик 1K — "
                    f"попытка {draft_number}"
                )
            )

            # ------------------------------------------------
            # CHECK
            # ------------------------------------------------

            bot.edit_message_text(
                f"🔎 Проверяю точность "
                f"черновика {draft_number}...",
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
                "[CHECK]",
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

            bot.send_message(
                message.chat.id,
                "📊 Проверка черновика\n\n"
                f"Общее соответствие: "
                f"{score}/100\n\n"
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
            # УСПЕШНЫЙ РЕЗУЛЬТАТ
            # ------------------------------------------------

            if score >= MATCH_THRESHOLD:

                successful_draft = draft_bytes
                successful_check = check_result
                successful_prompt = current_prompt

                print(
                    f"[SYSTEM] Прошёл порог: "
                    f"{score}/100"
                )

                break

            # ------------------------------------------------
            # НЕ УСПЕЛ
            # ------------------------------------------------

            if draft_number < MAX_DRAFT_ATTEMPTS:

                improved_prompt = (
                    check_result.get(
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

                    correction_text = "\n".join(
                        f"- {item}"
                        for item in corrections
                    )

                    current_prompt = (
                        current_prompt
                        + "\n\n"
                        "КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ "
                        "ДЛЯ СЛЕДУЮЩЕЙ ГЕНЕРАЦИИ:\n"
                        + correction_text
                    )

                bot.send_message(
                    message.chat.id,
                    "🔄 Результат недостаточно точный.\n\n"
                    f"Соответствие: {score}/100\n"
                    "Исправляю промпт и создаю "
                    "следующий черновик."
                )

        # ====================================================
        # ЕСЛИ НЕ ДОСТИГЛИ ПОРОГА
        # ====================================================

        if successful_draft is None:

            bot.edit_message_text(
                "⚠️ Не удалось достичь "
                f"порога {MATCH_THRESHOLD}/100 "
                "за 3 попытки.\n\n"
                "Финальный 2K не публикую, "
                "чтобы не отправлять в VK "
                "неточный результат.",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )

            bot.send_message(
                message.chat.id,
                "💡 Последний промпт:\n\n"
                f"{current_prompt}"
            )

            return

        # ====================================================
        # FINAL 2K
        # ====================================================

        bot.edit_message_text(
            "✅ Черновик прошёл проверку.\n\n"
            f"Соответствие: "
            f"{successful_check.get('overall_score')}/100\n\n"
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
        # Отправляем финал пользователю
        # ----------------------------------------------------

        bot.send_photo(
            message.chat.id,
            final_bytes,
            caption=(
                "✅ Финальное изображение "
                "Nano Banana Pro 2K"
            )
        )

        # ----------------------------------------------------
        # Пост VK
        # ----------------------------------------------------

        wall_text = build_wall_post_text(
            analysis_json
        )

        # Для VK комментария используем финальный промпт
        comment_text = (
            "📌 Промпт для генерации:\n\n"
            + successful_prompt
        )

        bot.edit_message_text(
            "🚀 Финал 2K готов.\n\n"
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

        # ----------------------------------------------------
        # Успешное завершение
        # ----------------------------------------------------

        bot.edit_message_text(
            "🎉 Готово!\n\n"
            "✅ Референс проанализирован\n"
            "✅ JSON создан\n"
            "✅ Промпт создан\n"
            "✅ Черновик 1K проверен\n"
            "✅ Финал 2K создан\n"
            "✅ Пост опубликован в VK\n"
            "✅ Промпт добавлен в комментарий\n\n"
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
# WALL POST TEXT
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

КАК СОЗДАТЬ ФОТО С ПОМОЩЬЮ БОТОВ 🖤🖤🖤🖤

🔹 БОТ 1 ВК — GPTron Nano Banana Pro🍌✅
1️⃣ Переходим в бот ➡️https://vk.com/write-236453790?ref=pp53aacd7d52

🔹 БОТ 2 ВК — Lexy Nano Banana Pro🍌✅
Переходим в бот ➡️https://vk.com/write-233546714?ref=84372609_add

Отправляем своё фото.

Перед отправкой вставляем нужный промт в комментариях.

ВАЖНО ПРОЧИТАТЬ

❗ Промт всегда можно и нужно менять под себя:
цвет волос, глаз, одежду, позу, настроение и т.д.

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{hashtags}"""


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
    f"Gemini Text: {TEXT_MODEL}"
)

print(
    f"Nano Banana Pro: {IMAGE_MODEL}"
)

print(
    f"VK API: 5.131"
)

print(
    f"Match threshold: {MATCH_THRESHOLD}"
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

        bot.infinity_polling(
            timeout=120,
            long_polling_timeout=120,
            skip_pending=True
        )

    except Exception as e:

        print(
            "[Telegram] Ошибка polling:",
            e
        )

        time.sleep(5)
