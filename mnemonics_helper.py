"""
Medical mnemonics generation tool using LLMs.

This module provides functionality to generate and manage mnemonic devices for medical concepts
using large language models. It includes semantic search to find similar existing mnemonics
and an interactive CLI interface for selecting from multiple generated options.
"""
from pathlib import Path
import litellm
import json
from typing import List
import numpy as np
import fire
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import joblib
from tqdm.asyncio import tqdm_asyncio

from utils.llm import model_name_matcher

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl


from iterator_cacher import IteratorCacher

# anchor_file = Path("./author_dir/anchors.json")
anchor_file = Path("/home/g/Documents/Perso/Bordel/git_repository/AnkiAiUtils/author_dir/anchors.json")
new_anchor_file = Path("/home/g/Documents/Perso/Bordel/git_repository/AnkiAiUtils/author_dir/new_anchors.json")

system_prompt = """
You are my best assistant. Your task is to generate new mnemonics to help me in medical school.

You follow these rules:
- Answer in the same language as the question.
- Answer only 1 mnemonic.
- The new mnemonics have to be easy to remember, so funny or tragic.
- The new mnemonics can only contain common words and can't use technical words.
- The new mnemonics can't contain digits.
- Directly answer the mnemonics, don't try to be polite, don't acknowledge those rules. Don't wrap your answer. Don't wrap the mnemonic in quotation signs.
- After your mnemonic, mention a single sentence between parentheses where you explain your reasonning. This will help me memorize it later.
- Your full answer must remain in a single line.
""".strip()

def prompt(
    examples: dict,
    question: str,
    ) -> str:
    """
    Generate a prompt for the LLM to create a mnemonic device.

    Parameters
    ----------
    examples : dict
        Dictionary of existing mnemonics to use as examples, where keys are concepts
        and values are their corresponding mnemonic devices
    question : str
        The concept/topic for which to generate a new mnemonic device

    Returns
    -------
    str
        The formatted prompt containing the system instructions, examples, and question
    """
    message = f"La notion pour laquelle j'ai besoin d'un moyen mnémotechnique est: '{question}'"
    message += "\nVoici des exemples de moyen mnémotechniques que j'utilise déjà:\n'''\n"
    ex = []
    for k, v in examples.items():
        ex.append(f"- '{k}': '{v}'")

    # ex = ex[::-1]  # reverse sorting order to have the end be closer to the question

    message += "\n".join(ex)

    message += f"\n'''\n\nMaintenant, suit les règles que je t'ai donné et suggère moi un moyen mnemotechnique adéquats. Je rappelle que la notion est: '{question}'"
    return message

anchors = json.loads(anchor_file.read_text())
assert anchors
try:
    new_anchors = json.loads(new_anchor_file.read_text())
except Exception as err:
    print(f"Not loading new anchors: '{err}'")
    new_anchors = {}


mem_object = joblib.Memory(".cache/iterator_cached_embeddings")
def embed(
    text_list: List[str],
    model: str = "openai/text-embedding-3-small",
    ):
    """
    Generate embeddings for a list of text strings using a specified model.

    Parameters
    ----------
    text_list : List[str]
        List of text strings to embed
    model : str, optional
        Name of the embedding model to use, by default "openai/text-embedding-3-small"

    Returns
    -------
    numpy.ndarray
        Array of embeddings, one per input text
    """
    assert text_list
    assert all(t.strip() for t in text_list)
    vec = litellm.embedding(
            model=model,
            input=text_list,
            )
    embeds = [
        np.array(elem["embedding"]).reshape(1, -1)
        for elem in vec.to_dict()["data"]
    ]
    embeds = np.array(embeds)
    return embeds

embed = IteratorCacher(
    memory_object=mem_object,
    iter_list=["text_list"],
    verbose=True,
    res_to_list = lambda ar: ar.tolist(),
)(embed)


def cli(
    top_k: int,
    n_gen: int,
    query: str,
    model: str,
    embed_model: str,
    ):
    """
    Run the interactive CLI interface for mnemonic generation.

    Parameters
    ----------
    top_k : int
        Number of similar existing mnemonics to use as examples
    n_gen : int
        Number of new mnemonics to generate per query
    query : str
        Initial query to process, if any
    model : str
        Name of the LLM model to use for generation
    embed_model : str
        Name of the model to use for text embeddings
    """
    session = PromptSession()

    if model.startswith("openrouter"):
        model_params = litellm.get_supported_openai_params(
            model_name_matcher(
                model.split("/", 1)[1]
                )
            )
    else:
        model_params = litellm.get_supported_openai_params(model)

    while True:

        # refresh db
        for k, v in new_anchors.items():
            assert k not in anchors.keys()
            assert v not in anchors.values()
        anchors_list = sorted(list(anchors.keys()) + list(new_anchors.keys()))
        db = np.array(
                embed(
                text_list=anchors_list,
                model=embed_model,
            )
        )

        if query is not None:
            ans, query = query, None
            query = None
        else:
            ans = session.prompt(
                "Enter the string your want mnemonics for:\n>",
                vi_mode=True,
                multiline=False,
                completer=WordCompleter(anchors_list),
            ).strip()
        assert ans
        if ans in anchors_list:
            if ans in anchors:
                print(f"Already part of anchors: '{anchors[ans]}'")
            elif ans in new_anchors:
                print(f"Already part of anchors: '{new_anchors[ans]}'")
            else:
                raise ValueError(ans)

        vec = embed(text_list=[ans])
        assert len(vec) == 1
        vec = vec[0]
        if isinstance(vec, list):
            vec = np.array(vec)

        sims = cosine_similarity(
            vec.squeeze().reshape(1, -1),
            db.squeeze(),
        )

        best_index = np.argsort(sims).tolist()[0][-top_k:]
        best_k = [anchors_list[b] for b in best_index]
        best_a = [anchors[k] if k in anchors else new_anchors[k] for k in best_k]
        best = {k:v for k, v in zip(best_k, best_a)}


        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt(examples=best, question=ans),
            }
        ]

        if "n" in model_params:
            resp = litellm.completion(
                model=model,
                messages=messages,
                num_retries=3,
                verbose=True,
                n=n_gen,
                temperature=1,
            ).json()
            texts = [elem["message"]["content"] for elem in resp["choices"]]
        else:
            async def async_completion(model, messages):
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, lambda: litellm.completion(
                    model=model,
                    messages=messages,
                    num_retries=3,
                    verbose=True,
                    temperature=1,
                ).json())
                return resp["choices"][0]["message"]["content"]
            async def generate_texts_async(n_gen, model, messages):
                tasks = [async_completion(model, messages) for _ in range(n_gen)]
                return await tqdm_asyncio.gather(*tasks, desc="Generating")
            def generate_texts(n_gen, model, messages):
                """
                Generate multiple mnemonic texts using async calls to the LLM.

                Parameters
                ----------
                n_gen : int
                    Number of texts to generate
                model : str
                    Name of the LLM model to use
                messages : List[Dict]
                    Prompt messages to send to the model

                Returns
                -------
                List[str]
                    List of generated mnemonic texts
                """
                return asyncio.run(generate_texts_async(n_gen, model, messages))
# Usage (inside your synchronous function):
            texts = asyncio.run(generate_texts_async(n_gen, model, messages))


        texts.append("Cancel")

        selected_index = [0]
        def get_menu_text():
            """
            Format the menu text for the CLI interface.

            Returns
            -------
            List[Tuple[str, str]]
                List of formatted text entries with their styles
            """
            out = []
            for i, item in enumerate(texts):
                n = ' ' * len(str(n_gen)) if item == "Cancel" else str(i + 1)
                sign = ">" if i == selected_index[0] else " "
                line = f"{sign} {n} {item}\n\n"
                out.append(("class:menu-item", line))
            return out

        kb = KeyBindings()
        @kb.add("k")
        @kb.add("up")
        def move_up(event):
            selected_index[0] = max(selected_index[0] - 1, 0)

        @kb.add("j")
        @kb.add("down")
        def move_down(event):
            selected_index[0] = min(selected_index[0] + 1, len(texts) - 1)

        @kb.add("enter")
        def select(event):
            event.app.exit()

        @kb.add("c-c")
        @kb.add("q")
        def quit(event):
            print("Quit.")
            raise SystemExit()

        for i in range(10):
            @kb.add(str(i))
            def _(event, i=i):
                i = min(i, len(texts))
                i = max(0, i)
                selected_index[0] = i


        menu_control = FormattedTextControl(get_menu_text)
        menu_window = Window(
            content=menu_control,
            always_hide_cursor=True,
            wrap_lines=True,
        )
        root_container = HSplit([menu_window])
        layout = Layout(root_container)
        app = Application(layout=layout, key_bindings=kb, full_screen=False)
        print("\n")
        try:
            app.run()
        except EOFError:
            raise SystemExit()

        new_anc = texts[selected_index[0]]
        if new_anc == "Cancel":
            print("Cancelled")
            continue

        # store to new_anchors
        new_anchors[ans] = new_anc
        with open(new_anchor_file, "w") as f:
            json.dump(new_anchors, f, indent=4, ensure_ascii=False)

def start(
    top_k: int = 100,
    n_gen: int = 10,
    query: str = None,
    model: str = "openrouter/anthropic/claude-3.5-sonnet:beta",
    embed_model: str = "openai/text-embedding-3-small",
    gui: bool = False,
    ):
    """
    Entry point for the mnemonic generation tool.

    Parameters
    ----------
    top_k : int, optional
        Number of similar existing mnemonics to use, by default 100
    n_gen : int, optional
        Number of new mnemonics to generate per query, by default 10
    query : str, optional
        Initial query to process, by default None
    model : str, optional
        LLM model name, by default "openrouter/anthropic/claude-3.5-sonnet:beta"
    embed_model : str, optional
        Embedding model name, by default "openai/text-embedding-3-small"
    gui : bool, optional
        Whether to use GUI interface (not implemented), by default False
    """
    if not gui:
        cli(
            top_k=top_k,
            n_gen=n_gen,
            query=query,
            model=model,
            embed_model=embed_model,
        )
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    fire.Fire(start)
