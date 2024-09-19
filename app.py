import gradio as gr
from dotenv import load_dotenv

from models import get_all_models, get_random_models

load_dotenv()


share_js = """
function () {
    const captureElement = document.querySelector('#share-region-annoy');
    // console.log(captureElement);
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'guardrails-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [];
}
"""


def activate_chat_buttons():
    regenerate_btn = gr.Button(
        value="🔄  Regenerate", interactive=True, elem_id="regenerate_btn"
    )
    clear_btn = gr.ClearButton(
        elem_id="clear_btn",
        interactive=True,
    )
    return regenerate_btn, clear_btn


def deactivate_chat_buttons():
    regenerate_btn = gr.Button(
        value="🔄  Regenerate", interactive=False, elem_id="regenerate_btn"
    )
    clear_btn = gr.ClearButton(
        elem_id="clear_btn",
        interactive=False,
    )
    return regenerate_btn, clear_btn


def handle_message(
    llms, user_input, temperature, top_p, max_output_tokens, states1, states2, states3, states4
):
    history1 = states1.value if states1 else []
    history2 = states2.value if states2 else []
    history3 = states3.value if states3 else []
    history4 = states4.value if states4 else []
    states = [states1, states2,states3, states4]
    history = [history1, history2,history3, history4]
    for hist in history:
        hist.append((user_input, None))
    for (
        updated_history1,
        updated_history2,
        updated_history3,
        updated_history4,
        updated_states1,
        updated_states2,
        updated_states3,
        updated_states4,
    ) in process_responses(
        llms, temperature, top_p, max_output_tokens, history, states
    ):
        yield updated_history1, updated_history2,updated_history3, updated_history4, updated_states1, updated_states2,updated_states3, updated_states4


def regenerate_message(llms, temperature, top_p, max_output_tokens, states1, states2, states3, states4):
    history1 = states1.value if states1 else []
    history2 = states2.value if states2 else []
    history3 = states3.value if states3 else []
    history4 = states4.value if states4 else []
    user_input = (
        history1.pop()[0] if history1 else None
    )  # Assumes regeneration is needed so there is at least one input
    if history2:
        history2.pop()
    if history3:
        history3.pop()
    if history4:
        history4.pop()
    states = [states1, states2,states3, states4]
    history = [history1, history2,history3, history4]
    for hist in history:
        hist.append((user_input, None))
    for (
        updated_history1,
        updated_history2,
        updated_history3,
        updated_history4,
        updated_states1,
        updated_states2,
        updated_states3,
        updated_states4,
    ) in process_responses(
        llms, temperature, top_p, max_output_tokens, history, states
    ):
        yield updated_history1, updated_history2,updated_history3, updated_history4, updated_states1, updated_states2,updated_states3, updated_states4


def process_responses(llms, temperature, top_p, max_output_tokens, history, states):
    generators = [
        llms[i]["model"](history[i], temperature, top_p, max_output_tokens)
        for i in range(4)
    ]
    # need to add num of here with models
    responses = [[], [],[], []]
    done = [False, False,False, False]

    while not all(done):
        for i in range(4):
            print(generators[i])
            print(done[i])
            if not done[i]:
                try:
                    response = next(generators[i])
                    if response:
                        responses[i].append(response)
                        history[i][-1] = (history[i][-1][0], "".join(responses[i]))
                        states[i] = gr.State(history[i])
                    yield history[0], history[1],history[2], history[3], states[0], states[1], states[2], states[3]
                except StopIteration:
                    done[i] = True
    print(history[0], history[1],history[2], history[3], states[0], states[1], states[2], states[3])
    yield history[0], history[1],history[2], history[3], states[0], states[1], states[2], states[3]


with gr.Blocks(
    title="Cherokee Language Function Test",
    theme=gr.themes.Soft(secondary_hue=gr.themes.colors.sky),
) as demo:
    num_sides = 4
    states = [gr.State() for _ in range(num_sides)]
    print(states)
    chatbots = [None] * num_sides
    models = gr.State(get_random_models)
    all_models = get_all_models()
    gr.Markdown(
        "# Cherokee Language Preserve Model\n\nChat with multiple models at the same time and compare their responses. "
    )
    with gr.Group(elem_id="share-region-annoy"):
        # with gr.Accordion(f"🔍 Expand to see the {len(all_models)} models", open=False):
        #     model_description_md = """| | | |\n| ---- | ---- | ---- |\n"""
        #     count = 0
        #     for model in all_models:
        #         if count % 3 == 0:
        #             model_description_md += "|"
        #         model_description_md += f" {model['name']} |"
        #         if count % 3 == 2:
        #             model_description_md += "\n"
        #         count += 1
        #     gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            for i in range(num_sides):
                label = models.value[i]["name"]
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label,
                        elem_id=f"chatbot",
                        height=550,
                        show_copy_button=True,
                    )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="Enter your query and press ENTER",
            elem_id="input_box",
            scale=4,
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.ClearButton(
            value="🎲 New Round",
            elem_id="clear_btn",
            interactive=False,
            components=chatbots + states,
        )
        regenerate_btn = gr.Button(
            value="🔄 Regenerate", interactive=False, elem_id="regenerate_btn"
        )
        share_btn = gr.Button(value="📷 Share Image")

    with gr.Row():
        examples = gr.Examples(
            [
                "Tell me a story",
                "What is the capital of France?",
                "Do you like me?",
            ],
            inputs=[textbox],
            label="Example task: General skill",
        )
    with gr.Row():
        examples = gr.Examples(
            [
                "translate: ᎧᏃᎮᏍᎩ",
                "Could you assist in rendering this Cherokee word into English?\nᎤᏲᎢ",
                "translate the following Cherokee word into English. ᏧᎩᏨᏅᏓ",
            ],
            inputs=[textbox],
            label="Example task: Translate words",
        )
    with gr.Row():
        examples = gr.Examples(
            [
                "translate: ᏚᏁᏤᎴᏃ ᎬᏩᏍᏓᏩᏗᏙᎯ, ᎾᏍᎩ ᏥᏳ ᎤᎦᏘᏗᏍᏗᏱ, ᎤᏂᏣᏘ ᎨᏒ ᎢᏳᏍᏗ, ᎾᏍᎩ ᎬᏩᏁᏄᎳᏍᏙᏗᏱ ᏂᎨᏒᎾ",
                "translate following Cherokee sentences into English.\nᏥᏌᏃ ᎤᏓᏅᏎ ᏚᏘᏅᏎ ᎬᏩᏍᏓᏩᏗᏙᎯ ᎥᏓᎵ ᏭᏂᎶᏎᎢ; ᎤᏂᏣᏘᏃ ᎬᏩᏍᏓᏩᏛᏎᎢ, ᏅᏓᏳᏂᎶᏒᎯ ᎨᎵᎵ, ᎠᎴ ᏧᏗᏱ,",
                "translate following sentences.\nᎯᎠᏃ ᏄᏪᏎᎴ ᎠᏍᎦᏯ ᎤᏬᏰᏂ ᎤᏩᎢᏎᎸᎯ; ᎠᏰᎵ ᎭᎴᎲᎦ.",
            ],
            inputs=[textbox],
            label="Example task: Translate sentences",
        )

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.01,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.01,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=4096,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    with gr.Row():
        examples = gr.Markdown(
        """
# Introducing The Cherokee Language Preserve Model

## Overview
I am excited to present the latest language model, which has been  fine-tuned using the state-of-the-art LoRA (Low-Rank Adaptation) technique on the robust foundation of the LLaMA3-8B model. This fine-tuning process has been specifically tailored to enhance the model's performance on Cherokee language translation tasks, setting a new standard in the field.

## Data Sets Utilized
This model has been trained on two specialized datasets build by myself to ensure its proficiency in Cherokee-English translation:

1. **Cherokee-English Bible Sentence (7.96k)**  [Dataset Link](https://huggingface.co/datasets/wang4067/cherokee-english-bible-7.96k)  
   This dataset provides a rich source of bilingual text, enabling our model to understand and reproduce the nuances of the Cherokee language within a religious context.

2. **Cherokee-English Word (10.2k)**  [Dataset Link](https://huggingface.co/datasets/wang4067/cherokee-english-word-10.2k)  
   This dataset focuses on vocabulary, ensuring that our model has a comprehensive grasp of Cherokee words and their English counterparts.

## Performance Achievements
This model has demonstrated exceptional performance in Cherokee language translation tasks, surpassing mainstream models such as LLaMA3-8B, LLaMA3.1-8B, and PHI3. It has achieved state-of-the-art (SOTA) results without the common issue of catastrophic forgetting.

Here are some details about performance.
```shell
{
    "predict_bleu-4": 96.79794598214286,
    "predict_rouge-1": 98.21964419642859,
    "predict_rouge-2": 97.57667857142857,
    "predict_rouge-l": 98.36520848214286,
    "predict_runtime": 93.1528,
    "predict_samples_per_second": 2.147,
    "predict_steps_per_second": 0.075
}
```
Here are some details about this training process.
```shell
bf16: true
cutoff_len: 1024
dataset: dict_word_v4,dict_sentence_v4
dataset_dir: data
ddp_timeout: 180000000
do_train: true
finetuning_type: lora
flash_attn: auto
gradient_accumulation_steps: 8
include_num_input_tokens_seen: true
learning_rate: 0.0001
logging_steps: 5
lora_alpha: 16
lora_dropout: 0.1
lora_rank: 8
lora_target: all
lr_scheduler_type: cosine
max_grad_norm: 1.0
max_samples: 100000
model_name_or_path: /wsh/models/Meta-Llama-3-8B-Instruct
num_train_epochs: 40.0
optim: adamw_torch
output_dir: saves/Custom/lora/train_2024-09-15-17-54-11-v4-learn_rate_0001
packing: false
per_device_train_batch_size: 2
plot_loss: true
preprocessing_num_workers: 16
report_to: none
save_steps: 100
stage: sft
warmup_steps: 0
```
        """
        )
    print(states[0]),
    print(states[1]),
    print(states[2]),
    print(states[3]),
    textbox.submit(
        handle_message,
        inputs=[
            models,
            textbox,
            temperature,
            top_p,
            max_output_tokens,
            states[0],
            states[1],
            states[2],
            states[3],
        ],
        # outputs=[chatbots[0], chatbots[1], states[0], states[1]],
        outputs=[chatbots[0], chatbots[1],chatbots[2], chatbots[3], states[0], states[1], states[2], states[3]],
    ).then(
        activate_chat_buttons,
        inputs=[],
        outputs=[regenerate_btn, clear_btn],
    )

    send_btn.click(
        handle_message,
        inputs=[
            models,
            textbox,
            temperature,
            top_p,
            max_output_tokens,
            states[0],
            states[1],
            states[2],
            states[3],
        ],
        outputs=[chatbots[0], chatbots[1],chatbots[2], chatbots[3], states[0], states[1], states[2], states[3]],
    ).then(
        activate_chat_buttons,
        inputs=[],
        outputs=[regenerate_btn, clear_btn],
    )

    regenerate_btn.click(
        regenerate_message,
        inputs=[
            models,
            temperature,
            top_p,
            max_output_tokens,
            states[0],
            states[1],
            states[2],
            states[3],
            
        ],
        outputs=[chatbots[0], chatbots[1],chatbots[2], chatbots[3], states[0], states[1], states[2], states[3]],
    )

    clear_btn.click(
        deactivate_chat_buttons,
        inputs=[],
        outputs=[regenerate_btn, clear_btn],
    ).then(lambda: get_random_models(), inputs=None, outputs=[models])

    share_btn.click(None, inputs=[], outputs=[], js=share_js)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10)
    demo.launch(server_port=50008)
