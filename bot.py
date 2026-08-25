import io
import json
import os
import threading
from flask import Flask
import google.generativeai as genai
from PIL import Image
import telebot
import requests

# ================= НАСТРОЙКИ =================
TELEGRAM_BOT_TOKEN = "8879421117:AAEIwaZy0Md6vvPJqRzjzYuS_Sg3SPUeiKA"
GEMINI_API_KEY = "AQ.Ab8RN6JVP4ezCQ-9IgCEdOVzMVrhfQZwoVKom_v950fBOD0MZw"  # Ваш актуальный ключ Gemini
VK_ACCESS_TOKEN = "vk1.a.z0Ee0Bg1TDR16EYVXeCQMPwGKyoQeJp_ofHI9QbrSZJO9KUerGCjupKBNhwBgRKRzWFGz7Oz-NqtJyFpVjyDJu6EFyCPq3AkzZ05yWXrwooEjBUgA300Q_9wf2ixgZo9vy2NMYijPbNUqca0IGgxJqzbGcbhn-jKZSDW3oIcofGgQ7hkEqIlj7aFrPbjNd9Xd56hkyy2z93k8hfvpyz6qA"  # Ваш токен ВКонтакте
VK_GROUP_ID = "240635290"  # ID группы
# =============================================

# Фоновый мини веб-сервер для бесплатного тарифа Render
app = Flask(__name__)


@app.route("/")
def home():
  return "Бот активен и работает 24/7!"


def run_web_server():
  port = int(os.environ.get("PORT", 8080))
  app.run(host="0.0.0.0", port=port)


# Инициализируем Gemini 3.6 Flash
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
    generation_config={"response_mime_type": "application/json"},
)

# Инициализируем Telegram-бота
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


def vk_call(method_name, params=None, data=None):
  clean_method = method_name.strip().split("/")[-1].replace(".json", "")
  endpoint_url = "https://" + "api.vk.com/method/" + clean_method
  if data:
    response = requests.post(endpoint_url, data=data, timeout=20)
  else:
    response = requests.get(endpoint_url, params=params, timeout=20)
  return response.json()


def clean_json_response(raw_text):
  cleaned = raw_text.strip()
  if cleaned.startswith("```"):
    lines = cleaned.split("\n")
    if lines[0].startswith("```"):
      lines = lines[1:]
    if lines and lines[-1].startswith("```"):
      lines = lines[:-1]
    cleaned = "\n".join(lines).strip()
  return cleaned


def analyze_photo_with_gemini(image_bytes):
  system_prompt = """Проанализируй прикреплённое изображение и составь подробное текстовое описание кадра. Выдай ответ СТРОГО в формате JSON без вводных слов и разметки markdown со следующими ключами на русском языке:
{
  "photo_title": "Ёмкое, красивое и точное название, которое отражает конкретный сюжет, кто изображён и где происходит действие (например: 'Первоклассник за партой с бумажным самолётиком', 'Влюбленная пара на закате у моря', 'Девушка за столиком в уютном кафе')",
  "photo_style": "Определи стиль фото по самому изображению (например: Фотореалистичный портрет, Студийный эдиториал, Кинематографичный кадр, Ретро-фотография и т.д.)",
  "camera_and_settings": "Предполагаемая камера, объектив и параметры съёмки (например: Снято на Sony A7R V, 85mm f/1.4, f/2.0, ISO 100, 1/250s)",
  "shot_type_and_pose_intro": "Ракурс, ориентация кадра (например: вертикальный кадр 4:5), положение тела и взгляд",
  "hairstyle_and_makeup": "Детальное описание причёски, бровей, макияжа глаз, губ и состояния кожи",
  "outfit": "Детальное описание одежды, кроя, фасона, деталей кроя, ткани, обуви и аксессуаров",
  "pose_details": "Детальное описание позы, пластики тела, расположения рук и пальцев",
  "lighting": "Описание освещения, типа света (жёсткий/мягкий, теплый солнечный из окна), источников и теней",
  "background": "Описание фона и окружения (школьный класс, классная доска, парта, студия и т.д.)",
  "color_grading_and_style": "Описание цветокоррекции, контраста, палитры и общей обработки",
  "quality_and_style_tags": "Параметры качества под стиль фото (например: Максимальный фотореализм, высокая детализация, естественная анатомия, реалистичная кожа, без CGI, без мультяшности, без артефактов)",
  "hashtags": "5-8 целевых хэштегов строго через решётку #. Хэштеги должны отражать: сюжет, персонажа, место и стиль. Примеры: #1сентября #школьник #школа #детскийпортрет #нейрофото #промпт; если пара: #пара #влюбленные #lovestory #нейрофото #промпт"
}"""

  image = Image.open(io.BytesIO(image_bytes))
  response = gemini_model.generate_content([system_prompt, image])
  clean_text = clean_json_response(response.text)
  return json.loads(clean_text)


def build_wall_post_text(data):
  title = data.get("photo_title", "Нейрофотосессия")
  hashtags = data.get("hashtags", "#нейрофото #промпт #нейросеть")
  return f"""✨ {title}

👇 Забирай готовый промпт для генерации в комментариях к этому посту!

{hashtags}"""


def build_comment_prompt_text(data):
  return f"""📌 Промпт для генерации:

Внешность должна полностью соответствовать прикреплённому референсу: идентичные черты лица, возраст, рост, форма лица, цвет глаз, цвет и длина волос, причёска, телосложение, пропорции, макияж, выражение лица и общее визуальное впечатление. Любые изменения внешности, стилизация под другого человека или искажение типажа недопустимы.

Стиль: {data.get('photo_style', '')}
Камера и параметры: {data.get('camera_and_settings', '')}

{data.get('shot_type_and_pose_intro', '')}

Причёска, кожа и макияж: {data.get('hairstyle_and_makeup', '')}
Образ: {data.get('outfit', '')}
Поза: {data.get('pose_details', '')}
Освещение: {data.get('lighting', '')}
Фон: {data.get('background', '')}
Цветокор: {data.get('color_grading_and_style', '')}

Качество и стиль: {data.get('quality_and_style_tags', 'Максимальный фотореализм, высокая детализация, естественная анатомия, реалистичная кожа, без CGI, без мультяшности, без артефактов.')}"""


def post_to_vk(image_bytes, wall_text, comment_text):
  group_id = int(str(VK_GROUP_ID).replace("-", ""))

  # 1. Запрос адреса сервера загрузки
  server_res = vk_call(
      "photos.getWallUploadServer",
      params={
          "group_id": group_id,
          "access_token": VK_ACCESS_TOKEN,
          "v": "5.131",
      },
  )

  if "error" in server_res:
    raise Exception(f"Ошибка VK getWallUploadServer: {server_res['error']}")

  raw_upload_url = server_res["response"]["upload_url"]
  clean_upload_url = (
      raw_upload_url.split("](")[-1].rstrip(")")
      if "](" in raw_upload_url
      else raw_upload_url.strip("[]() ")
  )

  # 2. Загрузка фото на сервер ВК
  files = {"photo": ("photo.jpg", image_bytes, "image/jpeg")}
  upload_res = requests.post(clean_upload_url, files=files, timeout=30).json()

  if not upload_res or "photo" not in upload_res or upload_res["photo"] == "[]":
    raise Exception(f"Ошибка загрузки фото в ВК: {upload_res}")

  # 3. Сохранение фото
  save_res = vk_call(
      "photos.saveWallPhoto",
      data={
          "group_id": group_id,
          "server": upload_res["server"],
          "photo": upload_res["photo"],
          "hash": upload_res["hash"],
          "access_token": VK_ACCESS_TOKEN,
          "v": "5.131",
      },
  )

  if "error" in save_res:
    raise Exception(f"Ошибка VK saveWallPhoto: {save_res['error']}")

  photo_info = save_res["response"][0]
  owner_id = photo_info["owner_id"]
  photo_id = photo_info["id"]

  # 4. Публикация поста на стене
  post_res = vk_call(
      "wall.post",
      data={
          "owner_id": -group_id,
          "from_group": 1,
          "message": wall_text,
          "attachments": f"photo{owner_id}_{photo_id}",
          "access_token": VK_ACCESS_TOKEN,
          "v": "5.131",
      },
  )

  if "error" in post_res:
    raise Exception(f"Ошибка VK wall.post: {post_res['error']}")

  post_id = post_res["response"]["post_id"]

  # 5. Публикация комментария от имени группы
  try:
    vk_call(
        "wall.createComment",
        data={
            "owner_id": -group_id,
            "post_id": post_id,
            "from_group": 1,
            "message": comment_text,
            "access_token": VK_ACCESS_TOKEN,
            "v": "5.131",
        },
    )
  except Exception as c_err:
    print(f"Предупреждение по комментарию: {c_err}")

  return f"[https://vk.com/wall-](https://vk.com/wall-){group_id}_{post_id}"


@bot.message_handler(commands=["start"])
def send_welcome(message):
  bot.reply_to(
      message,
      "👋 Привет! Отправь мне фотографию, и я оформлю пост с названием,"
      " хэштегами и оставлю готовый промпт в комментариях!",
  )


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
  status_msg = bot.reply_to(
      message, "⏳ Анализирую сюжет фото и составляю хэштеги..."
  )

  try:
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    json_data = analyze_photo_with_gemini(downloaded_file)
    wall_text = build_wall_post_text(json_data)
    comment_text = build_comment_prompt_text(json_data)

    bot.edit_message_text(
        "🚀 Публикую пост и оставляю промпт в комментариях...",
        chat_id=message.chat.id,
        message_id=status_msg.message_id,
    )
    post_link = post_to_vk(downloaded_file, wall_text, comment_text)

    bot.reply_to(
        message,
        f"✅ Пост опубликован!\n\n🔗 Ссылка на запись: {post_link}\n\n📝 Текст"
        f" поста:\n{wall_text}\n\n💬 Текст в"
        f" комментариях:\n\n{comment_text}",
    )

  except Exception as e:
    bot.reply_to(message, f"❌ Произошла ошибка: {str(e)}")


if __name__ == "__main__":
  threading.Thread(target=run_web_server, daemon=True).start()
  print("Бот успешно запущен 24/7 и ожидает отправки фото...")
  bot.infinity_polling()
