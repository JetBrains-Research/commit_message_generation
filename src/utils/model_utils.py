from copy import copy
from typing import Union, no_type_check

from transformers import GPT2LMHeadModel, RobertaForCausalLM, RobertaModel


@no_type_check  # a lot of attr-defined errors from transformers
def remove_layers_from_model(
    teacher: Union[RobertaModel, RobertaForCausalLM, GPT2LMHeadModel], num_layers: int
) -> Union[RobertaModel, RobertaForCausalLM, GPT2LMHeadModel]:
    if isinstance(teacher, RobertaForCausalLM):
        student_config = copy(teacher.config)
        student_config.num_hidden_layers = num_layers
        roberta_lm = RobertaForCausalLM(config=student_config)

        # copy all embeddings
        roberta_lm.roberta.embeddings.word_embeddings = teacher.roberta.embeddings.word_embeddings
        roberta_lm.roberta.embeddings.position_embeddings = teacher.roberta.embeddings.position_embeddings
        roberta_lm.roberta.embeddings.token_type_embeddings = teacher.roberta.embeddings.token_type_embeddings
        roberta_lm.roberta.embeddings.LayerNorm = teacher.roberta.embeddings.LayerNorm
        roberta_lm.roberta.embeddings.dropout = teacher.roberta.embeddings.dropout

        # uniformly pick from middle layers from teacher
        # it is basically np.linspace(0, teacher_config.num_hidden_layers,
        #                             num=student_config.num_hidden_layers, endpoint=True)
        step = (teacher.config.num_hidden_layers - 1) / (student_config.num_hidden_layers - 1)
        for student_layer, teacher_layer in enumerate(int(i * step) for i in range(student_config.num_hidden_layers)):
            roberta_lm.roberta.encoder.layer[student_layer] = teacher.roberta.encoder.layer[teacher_layer]
        return roberta_lm
    elif isinstance(teacher, RobertaModel):
        student_config = copy(teacher.config)
        student_config.num_hidden_layers = num_layers
        roberta = RobertaModel(config=student_config)  # type: ignore[assignment]

        # copy all embeddings
        roberta.embeddings.word_embeddings = teacher.embeddings.word_embeddings
        roberta.embeddings.position_embeddings = teacher.embeddings.position_embeddings
        roberta.embeddings.token_type_embeddings = teacher.embeddings.token_type_embeddings
        roberta.embeddings.LayerNorm = teacher.embeddings.LayerNorm
        roberta.embeddings.dropout = teacher.embeddings.dropout

        # uniformly pick from middle layers from teacher
        # it is basically np.linspace(0, teacher_config.num_hidden_layers,
        #                             num=student_config.num_hidden_layers, endpoint=True)
        step = (teacher.config.num_hidden_layers - 1) / (student_config.num_hidden_layers - 1)
        for student_layer, teacher_layer in enumerate(int(i * step) for i in range(student_config.num_hidden_layers)):
            roberta.encoder.layer[student_layer] = teacher.encoder.layer[teacher_layer]
        return roberta
    elif isinstance(teacher, GPT2LMHeadModel):
        student_config = copy(teacher.config)
        student_config.n_layer = num_layers
        gpt2_lm = GPT2LMHeadModel(config=student_config)

        # Copying all embeddings
        gpt2_lm.transformer.wte = teacher.transformer.wte
        gpt2_lm.transformer.wpe = teacher.transformer.wpe
        gpt2_lm.transformer.drop = teacher.transformer.drop

        # Specific thing for GPT2LMHead
        gpt2_lm.tie_weights()
        # Uniformly pick from middle layers from teacher
        # It is basically np.linspace(0, teacher_config.n_layer, num=student_config.n_layer, endpoint=True)
        step = (teacher.config.n_layer - 1) / (student_config.n_layer - 1)
        for student_layer, teacher_layer in enumerate(int(i * step) for i in range(student_config.n_layer)):
            gpt2_lm.transformer.h[student_layer] = teacher.transformer.h[teacher_layer]
        return gpt2_lm
