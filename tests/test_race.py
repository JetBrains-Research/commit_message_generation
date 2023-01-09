import pytest
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.model.configurations.utils.race import RACE


def test_forward_without_retrieval():
    model_name = "t5-small"
    race = RACE.from_pretrained(model_name)
    t5 = T5ForConditionalGeneration.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    input = "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)"
    target = "Amanda baked cookies and will bring Jerry some tomorrow."
    input_encodings = tok(
        "summarize: " + input, truncation=False, padding=False, add_special_tokens=True, return_tensors="pt"
    )
    target_encodings = tok(target, truncation=False, padding=False, add_special_tokens=True, return_tensors="pt")

    with torch.no_grad():
        race_outputs = race(
            input_ids=input_encodings.input_ids,
            attention_mask=input_encodings.attention_mask,
            labels=target_encodings.input_ids,
        )
        t5_outputs = t5(
            input_ids=input_encodings.input_ids,
            attention_mask=input_encodings.attention_mask,
            labels=target_encodings.input_ids,
        )

    assert race_outputs.keys() == t5_outputs.keys()
    assert race_outputs.loss == pytest.approx(t5_outputs.loss)
    assert race_outputs.logits.numpy() == pytest.approx(t5_outputs.logits.numpy())


def test_generation_without_retrieval():
    model_name = "t5-small"
    race = RACE.from_pretrained(model_name)
    t5 = T5ForConditionalGeneration.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    input = "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)"
    input_encodings = tok(
        "summarize: " + input, truncation=False, padding=False, add_special_tokens=True, return_tensors="pt"
    )
    with torch.no_grad():
        race_preds = race.generate(
            input_ids=input_encodings.input_ids,
            attention_mask=input_encodings.attention_mask,
            max_new_tokens=20,
            num_beams=10,
        )
        t5_preds = t5.generate(
            input_ids=input_encodings.input_ids,
            attention_mask=input_encodings.attention_mask,
            max_new_tokens=20,
            num_beams=10,
        )

    assert (race_preds.numpy() == t5_preds.numpy()).all()

    race_preds_str = tok.batch_decode(race_preds, skip_special_tokens=True)[0]
    t5_preds_str = tok.batch_decode(t5_preds, skip_special_tokens=True)[0]

    assert race_preds_str == t5_preds_str


def test_forward_with_retrieval():
    model_name = "t5-small"
    race = RACE.from_pretrained(model_name)
    t5 = T5ForConditionalGeneration.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    input = "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)"
    similar_input = "Jerry: I baked cookies. Do you want some? Amanda: Sure! Jerry: I'll bring you tomorrow :-)"
    similar_target = "Jerry baked cookies and will bring Amanda some tomorrow."
    target = "Amanda baked cookies and will bring Jerry some tomorrow."

    input_encodings = tok(input, truncation=False, padding=False, add_special_tokens=True, return_tensors="pt")
    similar_input_encodings = tok(
        similar_input, truncation=False, padding=False, add_special_tokens=True, return_tensors="pt"
    )
    similar_target_encodings = tok(
        similar_target, truncation=False, padding=False, add_special_tokens=True, return_tensors="pt"
    )

    with torch.no_grad():
        race_preds = race.generate(
            input_ids=input_encodings.input_ids,
            attention_mask=input_encodings.attention_mask,
            retrieved_diff_input_ids=similar_input_encodings.input_ids,
            retrieved_diff_attention_mask=similar_input_encodings.attention_mask,
            retrieved_msg_input_ids=similar_target_encodings.input_ids,
            retrieved_msg_attention_mask=similar_target_encodings.attention_mask,
            max_new_tokens=20,
            num_beams=10,
        )
        t5_preds = t5.generate(
            input_ids=input_encodings.input_ids,
            attention_mask=input_encodings.attention_mask,
            max_new_tokens=20,
            num_beams=10,
        )

    race_preds_str = tok.batch_decode(race_preds, skip_special_tokens=True)[0]
    t5_preds_str = tok.batch_decode(t5_preds, skip_special_tokens=True)[0]


def test_params():
    model_name = "t5-small"
    race = RACE.from_pretrained(model_name)
    t5 = T5ForConditionalGeneration.from_pretrained(model_name)
    race_params = race.num_parameters()
    t5_params = t5.num_parameters()
    # we add a linear layer from 2 * hidden_size to 1
    # it has 2 * hidden_size parameters for weight and 1 parameter for bias
    assert race_params - t5_params == race.config.d_model * 2 + 1
