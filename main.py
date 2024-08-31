import asyncio
from enum import Enum
import logging
import os
import sys
import traceback

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.enums import ParseMode
from aiogram.types import Message
from dotenv import load_dotenv
from runware import IImageInference, Runware

load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")


class ImageModel(str, Enum):
	FLUX_SCHNELL = "runware:100@1"
	FLUX_DEV = "runware:101@1"
	CIVITAI_618578 = "civitai:618578@693048"
	CIVITAI_81458 = "civitai:81458@132760"
	CIVITAI_101055 = "civitai:101055@128078"


dp = Dispatcher()
bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

runware = Runware(api_key=RUNWARE_API_KEY)


@dp.message(Command("start"))
async def start_command(message: Message) -> None:
	try:
		await message.answer(f"Використовуйте <code>/img &lt;запит&gt;</code> для генерації зображення.")
	except Exception as e:
		await message.answer(e)


@dp.message(Command("img", magic=F.args.as_("prompt")))
async def generate_image(
	message: Message,
	prompt: str
) -> None:
	sent_message = await message.reply("🖼 Обробка запиту та генерація зображення, будь ласка, зачекайте...")

	try:
		request_image = IImageInference(
			positivePrompt=prompt,
			model=ImageModel.FLUX_DEV,
			numberResults=1,
			useCache=False,
			height=1024,
			width=1024,
			steps=30,
			CFGScale=10,
			includeCost=True
		)
		images = await runware.imageInference(requestImage=request_image)
		await sent_message.delete()

		if not images:
			return await message.reply(
				"Не вдалося згенерувати зображення. Спробуйте ще раз."
			)

		image_url = images[0].imageURL
		image_cost = images[0].cost

		if not image_url:
			return await message.reply(
				"Не вдалося згенерувати зображення. Спробуйте ще раз."
			)

		await message.reply_photo(
			photo=image_url,
			caption=f"З генерації цього зображення автор бота втратив {image_cost if image_cost else 'невідомо'}$",
			has_spoiler=True,
		)
	except Exception:
		await message.reply(f"Виникла помилка при генераії зображення: \n<blockquote expandable>{traceback.format_exc()}</blockquote>")


async def main():
	try:
		await bot.delete_webhook(drop_pending_updates=True)
		await dp.start_polling(bot)
	finally:
		await bot.session.close()


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, stream=sys.stdout)
	asyncio.run(main())
