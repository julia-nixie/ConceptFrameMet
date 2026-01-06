import numpy as np
import torch
import torch.nn as nn

from utils import Config
from transformers import AutoTokenizer, AutoModel


class AutoModelForSequenceClassification(nn.Module):
    """Base model for sequence classification"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        target_mask=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`:   [batch_size, sequence_length] with the word token indices in the vocabulary
            `target_mask`:   [batch_size, sequence_length] with the mask for target wor. 1 for target word and 0 otherwise.
            `token_type_ids`:   [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`:   [batch_size, sequence_length] with indices selected in [0, 1].
                It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch.
                It's the mask that we typically use for attention when a batch has varying length sentences.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class SourceLogitsMelBert(nn.Module):
    """MelBERT with Source Logits INSTEAD of Frame Logits"""

    def __init__(self, args, Model, config, Source_Model, num_labels=2):
        """Initialize the model"""
        super(SourceLogitsMelBert, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.source_encoder = Source_Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.SPV_linear = nn.Linear((config.hidden_size * 2) + 2* 100, args.classifier_hidden)
        self.MIP_linear = nn.Linear((config.hidden_size * 2) + 2* 100, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
        input_with_mask_ids=None
    ):
        """
        Forward pass using SOURCE LOGITS (99 classes) INSTEAD of frame logits
        
        This model replaces frame semantic information with metaphor source domain information.
        Source domains include: MACHINE, JOURNEY, WAR, BUILDING, CONTAINER, etc. (99 total)
        """
        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        
        # 🎯 EXTRACT SOURCE LOGITS from source prediction model (99 metaphor source domains)
        source_outputs = self.source_encoder(
            input_ids,
            token_type_ids=target_mask.int(),
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]  # [batch, max_len, hidden]
        source_output = source_outputs.logits  # SOURCE logits [batch, seq_len, 100]
        pooled_output = outputs[1]  # [batch, hidden]

        # 🎯 Compute SOURCE distribution (99 source domains + 1 'O' class)
        source_cls = torch.sigmoid(source_output[:, 0])  # [batch, 100] - Sentence-level source prediction from CLS token
        source_logits = torch.softmax(source_output[:, 1:], dim=1)  # [batch, seq_len-1, 100]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        if self.args.small_mean:
            target_output = target_output.mean(1)  # [batch, hidden]
        else:
            target_output = target_output.sum(dim=1)/target_mask.sum(-1, keepdim=True)

        _, target_index_y = target_mask.max(dim=-1)
        target_index_x = torch.arange(target_index_y.size(0), device=target_mask.device)
        target_source_distribution = source_logits[target_index_x, target_index_y-1]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        source_outputs_2 = self.source_encoder(input_ids_2, token_type_ids=target_mask_2.int(), attention_mask=attention_mask_2, head_mask=head_mask)

        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]
        source_output_2 = source_outputs_2.logits

        source_logits_2 = torch.softmax(source_output_2[:, 1:], dim=1)

        _, target_index_y_2 = target_mask_2.max(dim=-1)
        target_index_x_2 = torch.arange(target_index_y_2.size(0), device=target_mask_2.device)
        isolated_target_source_distribution = source_logits_2[target_index_x_2, target_index_y_2-1]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        if self.args.small_mean:
            target_output_2 = target_output_2.mean(1)  # [batch, hidden]
        else:
            target_output_2 = target_output_2.sum(dim=1)/target_mask_2.sum(-1, keepdim=True)

        # Get hidden vectors each from SPV and MIP linear layers
        # SPV uses: [pooled/target, target_isolated, source_cls, isolated_source] - matches FrameLogits pattern
        # MIP uses: [target_isolated, target_ctx, ctx_source, isolated_source] - matches FrameLogits pattern
        if self.args.spv_isolate:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output_2, source_cls, isolated_target_source_distribution], dim=1))
        else:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output, source_cls, isolated_target_source_distribution], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output, target_source_distribution, isolated_target_source_distribution], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class FrameSourceLogitsMelBert(nn.Module):
    """MelBERT with BOTH Frame Logits AND Source Logits"""

    def __init__(self, args, Model, config, Frame_Model, Source_Model, num_labels=2):
        """Initialize the model"""
        super(FrameSourceLogitsMelBert, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.frame_encoder = Frame_Model
        self.source_encoder = Source_Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        # Frame: 797 classes, Source: 100 classes (including O)
        # SPV: pooled + target_isolated + frame_cls + isolated_frame + isolated_source = 768*2 + 797*2 + 100*1
        # MIP: target_isolated + target_ctx + frame_ctx + frame_isolated + source_ctx + source_isolated = 768*2 + 797*2 + 100*2
        self.SPV_linear = nn.Linear((config.hidden_size * 2) + 2* 797 + 100, args.classifier_hidden)
        self.MIP_linear = nn.Linear((config.hidden_size * 2) + 2* 797 + 2* 100, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
        input_with_mask_ids=None
    ):
        """
        Forward pass using BOTH Frame Logits (797 classes) AND Source Logits (99 classes)
        
        This model combines:
        - Frame semantic information (FrameNet frames)
        - Metaphor source domain information (MACHINE, JOURNEY, WAR, etc.)
        """
        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        
        # 🎯 EXTRACT FRAME LOGITS from frame prediction model (797 FrameNet frames)
        frame_outputs = self.frame_encoder(
            input_ids,
            token_type_ids=target_mask.int(),
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        
        # 🎯 EXTRACT SOURCE LOGITS from source prediction model (99 metaphor source domains)
        source_outputs = self.source_encoder(
            input_ids,
            token_type_ids=target_mask.int(),
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]  # [batch, max_len, hidden]
        frame_output = frame_outputs.logits  # FRAME logits [batch, seq_len, 797]
        source_output = source_outputs.logits  # SOURCE logits [batch, seq_len, 100]
        pooled_output = outputs[1]  # [batch, hidden]

        # 🎯 Compute FRAME distribution (797 FrameNet frames)
        frame_cls = torch.sigmoid(frame_output[:, 0])
        frame_logits = torch.softmax(frame_output[:, 1:], dim=1)  # [batch, seq_len-1, 797]
        
        # 🎯 Compute SOURCE distribution (99 source domains + 1 'O' class)
        source_logits = torch.softmax(source_output[:, 1:], dim=1)  # [batch, seq_len-1, 100]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        if self.args.small_mean:
            target_output = target_output.mean(1)  # [batch, hidden]
        else:
            target_output = target_output.sum(dim=1)/target_mask.sum(-1, keepdim=True)

        _, target_index_y = target_mask.max(dim=-1)
        target_index_x = torch.arange(target_index_y.size(0), device=target_mask.device)
        target_frame_distribution = frame_logits[target_index_x, target_index_y-1]
        target_source_distribution = source_logits[target_index_x, target_index_y-1]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        frame_outputs_2 = self.frame_encoder(input_ids_2, token_type_ids=target_mask_2.int(), attention_mask=attention_mask_2, head_mask=head_mask)
        source_outputs_2 = self.source_encoder(input_ids_2, token_type_ids=target_mask_2.int(), attention_mask=attention_mask_2, head_mask=head_mask)

        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]
        frame_output_2 = frame_outputs_2.logits
        source_output_2 = source_outputs_2.logits

        frame_logits_2 = torch.softmax(frame_output_2[:, 1:], dim=1)
        source_logits_2 = torch.softmax(source_output_2[:, 1:], dim=1)

        _, target_index_y_2 = target_mask_2.max(dim=-1)
        target_index_x_2 = torch.arange(target_index_y_2.size(0), device=target_mask_2.device)
        isolated_target_frame_distribution = frame_logits_2[target_index_x_2, target_index_y_2-1]
        isolated_target_source_distribution = source_logits_2[target_index_x_2, target_index_y_2-1]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        if self.args.small_mean:
            target_output_2 = target_output_2.mean(1)  # [batch, hidden]
        else:
            target_output_2 = target_output_2.sum(dim=1)/target_mask_2.sum(-1, keepdim=True)

        # Get hidden vectors each from SPV and MIP linear layers
        if self.args.spv_isolate:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output_2, frame_cls, isolated_target_frame_distribution, isolated_target_source_distribution], dim=1))
        else:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output, frame_cls, target_frame_distribution, target_source_distribution], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output, target_frame_distribution, isolated_target_frame_distribution, target_source_distribution, isolated_target_source_distribution], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForTokenClassification(nn.Module):
    """Base model for token classification"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        target_mask,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`:   [batch_size, sequence_length] with the word token indices in the vocabulary
            `target_mask`:   [batch_size, sequence_length] with the mask for target wor. 1 for target word and 0 otherwise.
            `token_type_ids`:   [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`:   [batch_size, sequence_length] with indices selected in [0, 1].
                It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch.
                It's the mask that we typically use for attention when a batch has varying length sentences.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = self.dropout(target_output)
        target_output = target_output.sum(1) / target_mask.sum()  # [batch, hideen]

        logits = self.classifier(target_output)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForSequenceClassification_SPV(nn.Module):
    """MelBERT with only SPV"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_SPV, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        target_mask,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`:   [batch_size, sequence_length] with the word token indices in the vocabulary
            `target_mask`:   [batch_size, sequence_length] with the mask for target wor. 1 for target word and 0 otherwise.
            `token_type_ids`:   [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`:   [batch_size, sequence_length] with indices selected in [0, 1].
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)  # [batch, hidden]

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        # Get mean value of target output if the target output consistst of more than one token
        target_output = target_output.mean(1)

        logits = self.classifier(torch.cat([target_output, pooled_output], dim=1))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForSequenceClassification_MIP(nn.Module):
    """MelBERT with only MIP"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_MIP, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`:   [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`:   [batch_size, sequence_length] with the second input token indicies
            `target_mask`:   [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`:   [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`:   [batch_size, sequence_length] with indices selected in [0, 1] for the second input.
            `token_type_ids`:   [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`:   [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = self.dropout(target_output)
        target_output = target_output.sum(1) / target_mask.sum()  # [batch, hidden]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        target_output_2 = target_output_2.sum(1) / target_mask_2.sum()

        logits = self.classifier(torch.cat([target_output_2, target_output], dim=1))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForSequenceClassification_SPV_MIP(nn.Module):
    """MelBERT"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_SPV_MIP, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.SPV_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.MIP_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        frame_input_ids=None,
        frame_attention_mask=None,
        frame_token_type=None,
        frame_labels=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
        input_with_mask_ids=None,
    ):
        """
        Inputs:
            `input_ids`: [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`:  [batch_size, sequence_length] with the second input token indicies

            --> mark target index
            `target_mask`:   [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`:   [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`:   [batch_size, sequence_length] with indices selected in [0, 1] for the second input.

            --> frame parameters (optional, for compatibility with multitask training)
            `frame_input_ids`: optional, for multitask learning (not used in FrameMelBert)
            `frame_attention_mask`: optional, for multitask learning (not used in FrameMelBert)
            `frame_token_type`: optional, for multitask learning (not used in FrameMelBert)
            `frame_labels`: optional, for multitask learning (not used in FrameMelBert)

            --> 0,1,2,3 四种输入; 1: target, 2: clause (local context), 3: pos info, 0: others
            `token_type_ids`:   [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`:   [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = self.dropout(target_output)

        if self.args.spvmask:
            outputs_with_mask = self.encoder(
                input_with_mask_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )
            sequence_output_with_mask = outputs_with_mask[0]
            mask_output = sequence_output * target_mask.unsqueeze(2)
            mask_output = self.dropout(target_output)
            if self.args.small_mean:
                mask_output = mask_output.mean(1)  # [batch, hidden]
            else:
                mask_output = mask_output.sum(dim=1)/target_mask.sum(-1, keepdim=True)
            pooled_output = mask_output
        elif self.args.spvmaskcls:
            outputs_with_mask = self.encoder(
                input_with_mask_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )
            pooled_output = outputs_with_mask[1]
            pooled_output = self.dropout(pooled_output)
        else:
            pooled_output = outputs[1]  # [batch, hidden]
            pooled_output = self.dropout(pooled_output)

        if self.args.small_mean:
            target_output = target_output.mean(1)  # [batch, hidden]
        else:
            target_output = target_output.sum(dim=1)/target_mask.sum(-1, keepdim=True)

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        if self.args.small_mean:
            target_output_2 = target_output_2.mean(1)  # [batch, hidden]
        else:
            target_output_2 = target_output_2.sum(dim=1)/target_mask_2.sum(-1, keepdim=True)
        # target_output_2 = target_output_2.mean(1)

        # Get hidden vectors each from SPV and MIP linear layers
        if self.args.spv_isolate:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output_2], dim=1))
        else:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class FrameMelBert(nn.Module):
    """MelBERT"""

    def __init__(self, args, Model, config, Frame_Model, num_labels=2):
        """Initialize the model"""
        super(FrameMelBert, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.frame_encoder = Frame_Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.SPV_linear = nn.Linear(config.hidden_size * 4, args.classifier_hidden)
        self.MIP_linear = nn.Linear(config.hidden_size * 4, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
        input_with_mask_ids=None,
    ):
        """
        Inputs:
            `input_ids`: [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`:  [batch_size, sequence_length] with the second input token indicies

            --> mark target index
            `target_mask`:   [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`:   [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`:   [batch_size, sequence_length] with indices selected in [0, 1] for the second input.

            --> 0,1,2,3 四种输入; 1: target, 2: clause (local context), 3: pos info, 0: others
            `token_type_ids`:   [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`:   [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        frame_outputs = self.frame_encoder(
            input_ids,
            token_type_ids=target_mask.int(),
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]  # [batch, max_len, hidden]
        frame_sequence_output = frame_outputs[0]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)
        frame_sequence_output = self.dropout(frame_sequence_output)

        if self.args.frame_mean:
            # frame_cls = (frame_sequence_output * attention_mask.unsqueeze(2)).mean(1)
            frame_cls = (frame_sequence_output * attention_mask.unsqueeze(2)).sum(dim=1)/attention_mask.sum(-1, keepdim=True)
        else:
            frame_cls = frame_sequence_output[:, 0]
        if self.args.small_mean:
            target_output = target_output.mean(1)  # [batch, hidden]
        else:
            target_output = target_output.sum(dim=1)/target_mask.sum(-1, keepdim=True)
        # target_output = target_output.mean(1)  # [batch, hidden]
        _, target_index_y = target_mask.max(dim=-1)
        target_index_x = torch.arange(target_index_y.size(0), device=target_mask.device)
        target_frame_embedding = frame_sequence_output[target_index_x, target_index_y]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        frame_outputs_2 = self.frame_encoder(input_ids_2, token_type_ids=target_mask_2.int(), attention_mask=attention_mask_2, head_mask=head_mask, output_hidden_states=True)

        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]
        # FrameFinder returns TokenClassifierOutput with hidden_states
        frame_sequence_output_2 = frame_outputs_2.hidden_states[-1]

        _, target_index_y_2 = target_mask_2.max(dim=-1)
        target_index_x_2 = torch.arange(target_index_y_2.size(0), device=target_mask_2.device)
        isolated_target_frame_embedding = frame_sequence_output_2[target_index_x_2, target_index_y_2]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        if self.args.small_mean:
            target_output_2 = target_output_2.mean(1)  # [batch, hidden]
        else:
            target_output_2 = target_output_2.sum(dim=1)/target_mask_2.sum(-1, keepdim=True)
        # target_output_2 = target_output_2.mean(1)
        isolated_target_frame_embedding = self.dropout(isolated_target_frame_embedding)

        # Get hidden vectors each from SPV and MIP linear layers
        if self.args.shuffle_concepts_in_batch:
            batch_size = input_ids.shape[0]
            frame_cls = frame_cls[torch.randperm(batch_size)]
            isolated_target_frame_embedding = isolated_target_frame_embedding[torch.randperm(batch_size)]
            target_frame_embedding = target_frame_embedding[torch.randperm(batch_size)]
        if self.args.spv_isolate:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output_2, frame_cls, isolated_target_frame_embedding], dim=1))
        else:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output, frame_cls, target_frame_embedding], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output, target_frame_embedding, isolated_target_frame_embedding], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits

class MultiTaskMelbert(FrameMelBert):

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,

        frame_input_ids=None,
        frame_attention_mask=None,
        frame_token_type = None,
        frame_labels = None,

        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
        input_with_mask_ids=None,
    ):
        """
            Inputs:
                `input_ids`: [batch_size, sequence_length] with the first input token indices in the vocabulary
                `input_ids_2`:  [batch_size, sequence_length] with the second input token indicies

                --> mark target index
                `target_mask`:   [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
                `target_mask_2`:   [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
                `attention_mask_2`:   [batch_size, sequence_length] with indices selected in [0, 1] for the second input.

                --> 0,1,2,3 四种输入; 1: target, 2: clause (local context), 3: pos info, 0: others
                `token_type_ids`:   [batch_size, sequence_length] with the token types indices
                    selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
                `attention_mask`:   [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
                `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                    with indices selected in [0, ..., num_labels].
                `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                    It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
            """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        frame_outputs = self.frame_encoder(
            input_ids,
            token_type_ids=target_mask.int(),
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_hidden_states=True,  # Enable hidden states output for TokenClassifierOutput
        )

        frame_logits = None
        if frame_input_ids is not None:
            frame_task_output = self.frame_encoder(frame_input_ids, 
                token_type_ids = frame_token_type, attention_mask = frame_attention_mask, labels = frame_labels)
            frame_loss = frame_task_output.loss

        sequence_output = outputs[0]  # [batch, max_len, hidden]
        # FrameFinder returns TokenClassifierOutput with hidden_states
        frame_sequence_output = frame_outputs.hidden_states[-1]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)
        frame_sequence_output = self.dropout(frame_sequence_output)

        if self.args.frame_mean:
            # frame_cls = (frame_sequence_output * attention_mask.unsqueeze(2)).mean(1)
            frame_cls = (frame_sequence_output * attention_mask.unsqueeze(2)).sum(dim=1)/attention_mask.sum(-1, keepdim=True)
        else:
            frame_cls = frame_sequence_output[:, 0]
        if self.args.small_mean:
            target_output = target_output.mean(1)  # [batch, hidden]
        else:
            target_output = target_output.sum(dim=1)/target_mask.sum(-1, keepdim=True)
        # target_output = target_output.mean(1)  # [batch, hidden]
        _, target_index_y = target_mask.max(dim=-1)
        target_index_x = torch.arange(target_index_y.size(0), device=target_mask.device)
        target_frame_embedding = frame_sequence_output[target_index_x, target_index_y]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        frame_outputs_2 = self.frame_encoder(input_ids_2, token_type_ids=target_mask_2.int(), attention_mask=attention_mask_2, head_mask=head_mask, output_hidden_states=True)

        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]
        # FrameFinder returns TokenClassifierOutput with hidden_states
        frame_sequence_output_2 = frame_outputs_2.hidden_states[-1]

        _, target_index_y_2 = target_mask_2.max(dim=-1)
        target_index_x_2 = torch.arange(target_index_y_2.size(0), device=target_mask_2.device)
        isolated_target_frame_embedding = frame_sequence_output_2[target_index_x_2, target_index_y_2]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        if self.args.small_mean:
            target_output_2 = target_output_2.mean(1)  # [batch, hidden]
        else:
            target_output_2 = target_output_2.sum(dim=1)/target_mask_2.sum(-1, keepdim=True)
        # target_output_2 = target_output_2.mean(1)
        isolated_target_frame_embedding = self.dropout(isolated_target_frame_embedding)

        # Get hidden vectors each from SPV and MIP linear layers
        if self.args.shuffle_concepts_in_batch:
            batch_size = input_ids.shape[0]
            frame_cls = frame_cls[torch.randperm(batch_size)]
            isolated_target_frame_embedding = isolated_target_frame_embedding[torch.randperm(batch_size)]
            target_frame_embedding = target_frame_embedding[torch.randperm(batch_size)]
        if self.args.spv_isolate:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output_2, frame_cls, isolated_target_frame_embedding], dim=1))
        else:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output, frame_cls, target_frame_embedding], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output, target_frame_embedding, isolated_target_frame_embedding], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits, frame_loss


class FrameLogitsMelBert(nn.Module):
    """MelBERT"""

    def __init__(self, args, Model, config, Frame_Model, num_labels=2):
        """Initialize the model"""
        super(FrameLogitsMelBert, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.frame_encoder = Frame_Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.SPV_linear = nn.Linear( (config.hidden_size * 2) + 2* 797, args.classifier_hidden)
        self.MIP_linear = nn.Linear((config.hidden_size * 2) + 2* 797, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
        input_with_mask_ids=None
    ):
        """
        Inputs:
            `input_ids`: [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`:  [batch_size, sequence_length] with the second input token indicies

            --> mark target index
            `target_mask`:   [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`:   [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`:   [batch_size, sequence_length] with indices selected in [0, 1] for the second input.

            --> 0,1,2,3 四种输入; 1: target, 2: clause (local context), 3: pos info, 0: others
            `token_type_ids`:   [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`:   [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        frame_outputs = self.frame_encoder(
            input_ids,
            token_type_ids=target_mask.int(),
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]  # [batch, max_len, hidden]
        frame_output = frame_outputs.logits
        pooled_output = outputs[1]  # [batch, hidden]

        frame_cls = torch.sigmoid(frame_output[:, 0])
        frame_logits = torch.softmax(frame_output[:, 1:], dim=1)

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)
        # frame_logits = self.dropout(frame_logits)

        if self.args.frame_mean:
            # frame_cls = (frame_sequence_output * attention_mask.unsqueeze(2)).mean(1)
            frame_cls = (frame_logits * attention_mask[:, 1:].unsqueeze(2)).sum(dim=1)/attention_mask[:, 1:].sum(-1, keepdim=True)

        if self.args.small_mean:
            target_output = target_output.mean(1)  # [batch, hidden]
        else:
            target_output = target_output.sum(dim=1)/target_mask.sum(-1, keepdim=True)

        _, target_index_y = target_mask.max(dim=-1)
        target_index_x = torch.arange(target_index_y.size(0), device=target_mask.device)
        target_frame_distribution = frame_logits[target_index_x, target_index_y-1]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        frame_outputs_2 = self.frame_encoder(input_ids_2, token_type_ids=target_mask_2.int(), attention_mask=attention_mask_2, head_mask=head_mask)

        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]
        frame_output_2 = frame_outputs_2.logits

        # frame_cls_2[:, 0] = torch.sigmoid(frame_output_2[:, 0])
        frame_logits_2 = torch.softmax(frame_output_2[:, 1:], dim=1)

        _, target_index_y_2 = target_mask_2.max(dim=-1)
        target_index_x_2 = torch.arange(target_index_y_2.size(0), device=target_mask_2.device)
        isolated_target_frame_distribution = frame_logits_2[target_index_x_2, target_index_y_2-1]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        if self.args.small_mean:
            target_output_2 = target_output_2.mean(1)  # [batch, hidden]
        else:
            target_output_2 = target_output_2.sum(dim=1)/target_mask_2.sum(-1, keepdim=True)

        # isolated_target_frame_distribution = self.dropout(isolated_target_frame_distribution)

        # Get hidden vectors each from SPV and MIP linear layers
        if self.args.shuffle_concepts_in_batch:
            batch_size = input_ids.shape[0]
            frame_cls = frame_cls[torch.randperm(batch_size)]
            isolated_target_frame_distribution = isolated_target_frame_distribution[torch.randperm(batch_size)]
            target_frame_distribution = target_frame_distribution[torch.randperm(batch_size)]
        if self.args.spv_isolate:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output_2, frame_cls, isolated_target_frame_distribution], dim=1))
        else:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output, frame_cls, target_frame_distribution], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output, target_frame_distribution, isolated_target_frame_distribution], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits
