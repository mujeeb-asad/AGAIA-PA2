#single turn conversion of the raw preference datasets

from alignment import data


class CharTokenizer:
    #one token per character, so the prompt length is just the length of the formatted string

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_hh_row_extracts_a_single_turn_pair():
    row = {
        "chosen": "\n\nHuman: How do I boil an egg?\n\nAssistant: Simmer it for ten minutes.",
        "rejected": "\n\nHuman: How do I boil an egg?\n\nAssistant: No idea.",
    }
    pair = data.hh_row(row)

    assert pair["instruction"] == "How do I boil an egg?"
    assert pair["chosen"] == "Simmer it for ten minutes."
    assert pair["rejected"] == "No idea."


def test_hh_row_rejects_multi_turn_transcripts():
    transcript = "\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: Bye\n\nAssistant: Goodbye"
    assert data.hh_row({"chosen": transcript, "rejected": transcript}) is None


def test_ultrafeedback_row_takes_the_assistant_turn():
    row = {
        "prompt": " name a colour ",
        "prompt_id": "abc",
        "chosen": [{"role": "user", "content": "name a colour"}, {"role": "assistant", "content": "Blue."}],
        "rejected": [{"role": "user", "content": "name a colour"}, {"role": "assistant", "content": "Seven."}],
    }
    assert data.ultrafeedback_row(row) == {
        "id": "abc",
        "instruction": "name a colour",
        "chosen": "Blue.",
        "rejected": "Seven.",
    }


def test_is_usable_filters_empty_identical_and_long_rows():
    tokenizer = CharTokenizer()
    pair = {"instruction": "hi", "chosen": "a", "rejected": "b"}

    assert data.is_usable(pair, tokenizer, max_prompt_tokens=64)
    assert not data.is_usable({**pair, "rejected": "a"}, tokenizer, max_prompt_tokens=64)
    assert not data.is_usable({**pair, "rejected": ""}, tokenizer, max_prompt_tokens=64)
    assert not data.is_usable(pair, tokenizer, max_prompt_tokens=8)
