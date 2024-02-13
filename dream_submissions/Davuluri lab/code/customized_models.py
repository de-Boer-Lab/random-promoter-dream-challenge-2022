import torch
import torch.nn as nn


from transformers import BertPreTrainedModel, BertModel


class BertSiameseForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.hidden = nn.Linear(config.hidden_size*3, 2048)
        self.classifier = nn.Linear(2048, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids_1 = torch.zeros([len(input_ids), 512])
        input_ids_2 = torch.zeros([len(input_ids), 512])
        for i, item in enumerate(input_ids):
            first_sep = torch.where(item==3)[0][0]
            first_sequence = item[:first_sep+1].clone()
            second_sequence = item[first_sep:].clone()
            second_sequence[0] = 2

            if len(first_sequence) > 512:
                first_sequence = first_sequence[:512]
                first_sequence[-1] = 3
            if len(second_sequence) > 512:
                second_sequence = second_sequence[:512]
                second_sequence[-1] = 3

            input_ids_1[i, :len(first_sequence)] = first_sequence
            input_ids_2[i, :len(second_sequence)] = second_sequence
        
        input_ids_1 = input_ids_1.type(input_ids.type()).to(input_ids.device)
        input_ids_2 = input_ids_2.type(input_ids.type()).to(input_ids.device)

        attention_mask_1 = (input_ids_1 != 0).type(input_ids.type()).to(input_ids.device)
        attention_mask_2 = (input_ids_2 != 0).type(input_ids.type()).to(input_ids.device)


        outputs_1 = self.bert(
            input_ids_1,
            attention_mask=attention_mask_1,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs_2 = self.bert(
            input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # cls
        # pooled_output_1 = outputs_1[1]
        # pooled_output_2 = outputs_2[1]

        # average
        last_hidden_1 = outputs_1[0]
        pooled_output_1 = ((last_hidden_1 * attention_mask_1.unsqueeze(-1)).sum(1) / attention_mask_1.sum(-1).unsqueeze(-1))
        last_hidden_2 = outputs_2[0]
        pooled_output_2 = ((last_hidden_2 * attention_mask_2.unsqueeze(-1)).sum(1) / attention_mask_2.sum(-1).unsqueeze(-1))


        # pooled_output_1, pooled_output_2 = self.dropout(pooled_output_1), self.dropout(pooled_output_2)

        pooled_output = torch.cat((pooled_output_1, pooled_output_2, pooled_output_1-pooled_output_2),dim=1)
        pooled_output = self.hidden(pooled_output)
        # pooled_output = torch.tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(logits.device))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                

        output = (logits,) 
        return ((loss,) + output) if loss is not None else output





class BertSiameseSimForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.cos = nn.CosineSimilarity(dim=-1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids_1 = torch.zeros([len(input_ids), 512])
        input_ids_2 = torch.zeros([len(input_ids), 512])
        for i, item in enumerate(input_ids):
            first_sep = torch.where(item==3)[0][0]
            first_sequence = item[:first_sep+1].clone()
            second_sequence = item[first_sep:].clone()
            second_sequence[0] = 2

            if len(first_sequence) > 512:
                first_sequence = first_sequence[:512]
                first_sequence[-1] = 3
            if len(second_sequence) > 512:
                second_sequence = second_sequence[:512]
                second_sequence[-1] = 3

            input_ids_1[i, :len(first_sequence)] = first_sequence
            input_ids_2[i, :len(second_sequence)] = second_sequence
        
        input_ids_1 = input_ids_1.type(input_ids.type()).to(input_ids.device)
        input_ids_2 = input_ids_2.type(input_ids.type()).to(input_ids.device)

        attention_mask_1 = (input_ids_1 != 0).type(input_ids.type()).to(input_ids.device)
        attention_mask_2 = (input_ids_2 != 0).type(input_ids.type()).to(input_ids.device)


        outputs_1 = self.bert(
            input_ids_1,
            attention_mask=attention_mask_1,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs_2 = self.bert(
            input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        pooled_output_1 = outputs_1[1]
        # pooled_output_1 = self.dropout(pooled_output_1)

        pooled_output_2 = outputs_2[1]
        # pooled_output_2 = self.dropout(pooled_output_2)

 
        pooled_output = self.cos(pooled_output_1, pooled_output_2)
        logits = torch.cat((1-pooled_output.unsqueeze(1), pooled_output.unsqueeze(1)), dim=1)


        loss = None
        if labels is not None:
            # labels.type(pooled_output.type()).to(pooled_output.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


                

        output = (logits,) 
        return ((loss,) + output) if loss is not None else output

