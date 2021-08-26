import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class BertCrf(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.num_labels = args.num_labels
        self.backbone_drop_rate = args.backbone_drop_rate
        self.backbone = backbone
        self.backbone_dropout_layer = nn.Dropout(self.backbone_drop_rate)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,):

        outputs = self.backbone(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,)
        loss = None
        logits = self.classifier(self.backbone_dropout_layer(outputs[0]))
        loss_mask = attention_mask.bool().bool()

        if labels is not None:
            loss = -self.crf(logits[:, 1:], labels[:, 1:], mask=loss_mask[:, 1:], reduction='mean')
        prediction = self.crf.decode(logits[:, 1:], mask=loss_mask[:, 1:])

        return loss, prediction
