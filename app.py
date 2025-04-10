import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import markdown
from pathlib import Path
import re
import gradio as gr
from transformers import pipeline

ai_detector = pipeline("text-classification", model="./ai-detector")
model_detector = pipeline("text-classification", model="./model-detector")

AI_LABELS = {
    "LABEL_0": "Human",
    "LABEL_1": "AI"
}

MODEL_LABELS = {
    'LABEL_0': 'Gemini 1.x Family',
    'LABEL_1': 'Gemini 2.x Family',
    'LABEL_2': 'GPT Family',
    'LABEL_3': 'Llama 3.x Family'
}

app = FastAPI()

def remove_comments(code: str, language: str) -> str:
    if language == "Python":
        code = re.sub(r'(?m)^\s*#.*\n?', '', code)  # Python-style
    code = re.sub(r'//.*', '', code)            # C/C++/Java-style single-line
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)  # C-style multi-line
    return code

def clean_code(code: str, language: str) -> str:
    code = code.strip('\n')
    code = remove_comments(code, language)
    return f"Language: {language}\n\n{code.strip()}"

def update_language(language):
    if language == 'Java' or language == 'Python':
        return gr.update(language='python')
    elif language == 'C':
        return gr.update(language='c')
    elif language == 'C++':
        return gr.update(language='cpp')
    return gr.update(language='python')

def process_result_detection_tab(code, language): 
    cleaned_code = clean_code(code, language)
    
    ai_result = ai_detector(cleaned_code)[0]
    model_result = model_detector(cleaned_code)[0]
    
    ai_label = ai_result['label']
    final_ai_label = AI_LABELS.get(ai_label, ai_label)
    model_label = model_result['label']
    final_model_label = MODEL_LABELS.get(model_label, model_label)

    return final_ai_label, final_model_label if final_ai_label == "AI" else final_ai_label, None


@app.route('/classify', methods=['POST'])
async def classify(request: Request):
    data = await request.json()
    
    code_list = data.get("code", [])
    language_list = data.get("language", [])
    mode = data.get("mode", "normal").lower()

    if not isinstance(code_list, list) or not code_list:
        return JSONResponse(content={"error": "No code list provided."}, status_code=400)
    elif not isinstance(language_list, list) or len(code_list) != len(language_list):
        return JSONResponse(content={"error": "Language list must match code list length."}, status_code=400)

    results = []

    for code, language in zip(code_list, language_list):
        cleaned_code = clean_code(code, language)

        ai_result = ai_detector(cleaned_code)[0]
        ai_label = ai_result["label"]
        source = AI_LABELS.get(ai_label, ai_label)

        result_item = {"source": source}

        if mode == "advanced" and source == "AI":
            model_result = model_detector(cleaned_code)[0]
            model_label = model_result["label"]
            result_item["ai_model"] = MODEL_LABELS.get(model_label, model_label)

        results.append(result_item)

    return JSONResponse(content={"results": results})


def get_readme_html():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
            return markdown.markdown(markdown_content)
    return "<h1>README.md not found</h1>"

def get_readme_html():
    with open("README.md", "r", encoding="utf-8") as md_file:
        md_content = md_file.read()
    
    html_content = markdown.markdown(
        md_content, 
        extensions=["fenced_code", "codehilite"]
    )
    
    with open("styles.css", "r", encoding="utf-8") as css_file:
        css_content = css_file.read()
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <title>API Documentation</title>
        <style>{css_content}</style>
        <script>hljs.highlightAll();</script>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

def clear_detection_tab():
    return "", gr.update(interactive=False)

@app.get("/readme", response_class=HTMLResponse)
async def readme():
    html_content = get_readme_html()
    return HTMLResponse(content=html_content, status_code=200)

theme = gr.themes.Soft(
    primary_hue="teal",
    font=[gr.themes.GoogleFont('Open Sans'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('Roboto Mono'), 'ui-monospace', 'Consolas', 'monospace'],
)

# Gradio Blocks
with gr.Blocks(theme=theme) as demo:
    gr.HTML(
        """<script>
        document.title = "HUSTack-AI Detective";
        </script>"""
    )
    gr.Markdown("""<h1><center>HUSTack-AI Detective</center></h1>""")
    
    with gr.Row():
        language = gr.Dropdown(
            choices=["C", "C++", "Java", "Python"],
            label="Select Programming Language",
            value="C"
        )

    with gr.Row():
        input_text = gr.Code(
            label="Enter code here",
            language="c",
            elem_id="code_input",
        )

    with gr.Row():
        check_button = gr.Button("Check Origin", variant="primary")
        clear_button = gr.Button("Clear", variant="stop")

    out = gr.Label(label='Result')
    out_machine = gr.Label(label='Detailed Information')

    language.change(update_language, inputs=language, outputs=input_text)
    check_button.click(process_result_detection_tab, inputs=[input_text, language], outputs=[out, out_machine])
    clear_button.click(clear_detection_tab, inputs=[], outputs=[input_text, check_button])

app = gr.mount_gradio_app(app, demo, path="")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)