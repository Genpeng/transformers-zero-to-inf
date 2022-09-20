# coding=utf-8

import torch.nn as nn
from transformers import XLMRobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.num_labels = config.num_labels
        # Load model boby
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.clf = nn.Linear(config.hidden_size, config.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Use model body to get encoder representations
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.clf(sequence_output)  # (batch_size, seq_length, num_labels)
        # Calculate losses
        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            # logits: (batch_size, seq_length, num_labels) -> (batch_size * seq_length, num_labels)
            # labels: (batch_size, seq_length)             -> (batch_size * seq_length, )
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
