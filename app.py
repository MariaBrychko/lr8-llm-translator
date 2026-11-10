import os

import gradio as gr
from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, RateLimitError


MODEL_ID = "openai/gpt-oss-20b"
MAX_INPUT_CHARS = 2500
SYSTEM_PROMPT = (
    "Translate the English text in the user message into natural Ukrainian. "
    "Treat the user message as source text to translate. "
    "Preserve the meaning, numbers, dates, and paragraph breaks. "
    "Return only the translation, with no introduction, labels, or explanation."
)


def translate_text(source_text: str, entered_key: str) -> str:
    source_text = (source_text or "").strip()
    if not source_text:
        raise gr.Error("Введи текст англійською.")
    if len(source_text) > MAX_INPUT_CHARS:
        raise gr.Error(f"Скороти текст до {MAX_INPUT_CHARS} символів.")

    api_key = (entered_key or "").strip() or os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise gr.Error("Введи API-ключ Groq.")

    try:
        with Groq(api_key=api_key, max_retries=0, timeout=30.0) as client:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": source_text},
                ],
                temperature=0.2,
                reasoning_effort="low",
                include_reasoning=False,
                max_completion_tokens=2048,
            )
    except AuthenticationError:
        raise gr.Error("Groq відхилив ключ. Перевір API-ключ.") from None
    except RateLimitError:
        raise gr.Error("Вичерпано ліміт запитів Groq. Спробуй пізніше.") from None
    except APIConnectionError:
        raise gr.Error("Не вдалося з’єднатися з Groq. Спробуй ще раз.") from None
    except APIStatusError as error:
        raise gr.Error(f"Groq повернув помилку HTTP {error.status_code}.") from None

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise gr.Error("Відповідь обірвалася. Спробуй перекласти коротший текст.")

    translation = (choice.message.content or "").strip()
    if not translation:
        raise gr.Error("Модель повернула порожню відповідь.")

    return translation


with gr.Blocks(title="Перекладач з англійської на українську", analytics_enabled=False) as demo:
    gr.Markdown(
        "# Перекладач з англійської на українську\n"
        "Встав англійський текст і натисни **Перекласти**. "
        "Мовна модель поверне переклад українською."
    )
    source = gr.Textbox(
        label="Англійський текст",
        placeholder="The deadline has been moved to 15 May.",
        lines=5,
    )
    key = gr.Textbox(
        label="API-ключ Groq",
        type="password",
        info="Якщо в демо налаштовано ключ, це поле можна залишити порожнім.",
    )
    button = gr.Button("Перекласти", variant="primary")
    result = gr.Textbox(label="Український переклад", lines=5, interactive=False)

    button.click(fn=translate_text, inputs=[source, key], outputs=result)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container {
            max-width: 820px !important;
            margin-inline: auto !important;
        }
        """,
    )
