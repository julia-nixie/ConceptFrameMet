"""
Adaptive Source QA MelBERT with Configurable Blending Strategies

This model provides configurable approaches to incorporating source domain information:

FLAGS:
1. --source_blend_mode: 'additive' or 'replacement' (default: 'replacement')
   - additive: enhanced = target + alpha * source (keeps target strength)
   - replacement: blended = conf * source + (1-conf) * target (original approach)

2. --source_use_mode: 'metaphor_only' or 'all' (default: 'all')
   - metaphor_only: Only use source for samples with high metaphor probability
   - all: Use source for all samples

3. --source_alpha: float (default: 0.3) - scaling factor for additive mode

4. --metaphor_threshold: float (default: 0.5) - threshold for metaphor-only mode

Architecture:
- CONTEXT: target_word in full sentence → encoder 1 → target_context_embedding  
- SOURCE: [SEP] sentence [SEP] target [SEP] → QA model → predict source + confidence
- ISOLATED: isolated target → encoder 2 → target_embedding
- BLEND: Configurable (additive or replacement)
- FILTER: Configurable (metaphor-only or all)
- MIP: [enhanced_embedding, target_context_embedding]
- SPV: [pooled, enhanced_embedding] or [pooled, target_context_embedding]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSourceQAMelBert(nn.Module):
    """MelBERT with configurable source domain blending strategies"""

    def __init__(self, args, Model, config, Source_QA_Model, 
                 source_qa_tokenizer, melbert_tokenizer, num_labels=2):
        """
        Initialize the model with configurable flags
        
        Args:
            args: Configuration arguments with:
                - source_blend_mode: 'additive' or 'replacement'
                - source_use_mode: 'metaphor_only' or 'all'
                - source_alpha: scaling factor for additive mode
                - metaphor_threshold: threshold for metaphor-only mode
            Model: MelBert encoder (RoBERTa/BERT)
            config: Model configuration
            Source_QA_Model: QA-style model to predict source domain
            source_qa_tokenizer: Tokenizer for QA model
            melbert_tokenizer: Tokenizer for MelBert
            num_labels: Number of metaphor classes (2: literal/metaphorical)
        """
        super(AdaptiveSourceQAMelBert, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.source_qa_model = Source_QA_Model
        self.source_qa_tokenizer = source_qa_tokenizer
        self.melbert_tokenizer = melbert_tokenizer
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        # Configuration flags with defaults
        self.source_blend_mode = getattr(args, 'source_blend_mode', 'replacement')
        self.source_use_mode = getattr(args, 'source_use_mode', 'all')
        self.source_alpha = getattr(args, 'source_alpha', 0.3)
        self.metaphor_threshold = getattr(args, 'metaphor_threshold', 0.5)

        # Freeze or unfreeze source QA model
        if not getattr(args, 'unfreeze_source_qa', False):
            for param in self.source_qa_model.parameters():
                param.requires_grad = False
        else:
            for param in self.source_qa_model.parameters():
                param.requires_grad = True

        # Load source labels
        self.source_id2label = {}
        try:
            import json
            with open('source_finder/source_labels.json', 'r') as f:
                source_label2id = json.load(f)
                self.source_id2label = {v: k for k, v in source_label2id.items()}
                print(f"✓ Loaded {len(self.source_id2label)} source domain labels")
        except Exception as e:
            print(f"❌ Warning: Could not load source labels: {e}")

        # SPV and MIP linear layers
        self.SPV_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.MIP_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)
        self._init_weights(self.classifier)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Print configuration
        print(f"\n{'='*80}")
        print(f"✓ AdaptiveSourceQAMelBert initialized")
        print(f"  - Blend Mode: {self.source_blend_mode.upper()}")
        if self.source_blend_mode == 'additive':
            print(f"    → enhanced = target + {self.source_alpha} * source")
        else:
            print(f"    → blended = conf * source + (1-conf) * target")
        print(f"  - Use Mode: {self.source_use_mode.upper()}")
        if self.source_use_mode == 'metaphor_only':
            print(f"    → Only use source when metaphor_score > {self.metaphor_threshold}")
        else:
            print(f"    → Use source for all samples")
        print(f"{'='*80}\n")

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def predict_source_and_embeddings(self, input_ids, target_mask, attention_mask, 
                                       input_ids_2, target_mask_2, attention_mask_2):
        """
        Predict source domain and get source/target embeddings
        
        Returns:
            source_embeddings: [batch_size, hidden_size]
            target_embeddings: [batch_size, hidden_size]
            confidences: [batch_size] - confidence scores
        """
        batch_size = input_ids.size(0)
        
        # 1. Decode sentences and extract target words
        sentences = []
        target_words = []
        
        for i in range(batch_size):
            sentence = self.melbert_tokenizer.decode(input_ids[i], skip_special_tokens=True)
            target_positions = target_mask[i].nonzero(as_tuple=True)[0]
            
            if len(target_positions) > 0:
                target_tokens = input_ids[i][target_positions]
                target_word = self.melbert_tokenizer.decode(target_tokens, skip_special_tokens=True)
            else:
                target_word = "unknown"
            
            sentences.append(sentence)
            target_words.append(target_word)
        
        # 2. Format QA input and predict source
        with torch.no_grad():
            qa_inputs = self.source_qa_tokenizer(
                sentences,
                target_words,
                max_length=self.args.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            qa_inputs = {k: v.to(input_ids.device) for k, v in qa_inputs.items()}
            
            # If source model is FrameAwareSourcePredictor, also pass frame inputs
            # (frame inputs are the same as source inputs for this use case)
            if hasattr(self.source_qa_model, 'frame_finder'):
                qa_inputs['frame_input_ids'] = qa_inputs['input_ids']
                qa_inputs['frame_attention_mask'] = qa_inputs['attention_mask']
        
        # 3. Get source predictions with confidence
        qa_outputs = self.source_qa_model(**qa_inputs)
        source_logits = qa_outputs.logits
        source_probs = torch.softmax(source_logits, dim=-1)
        predicted_source_ids = torch.argmax(source_logits, dim=-1)
        
        # Get confidence scores
        confidences = source_probs.gather(1, predicted_source_ids.unsqueeze(1)).squeeze(1)
        
        # Map to source words
        with torch.no_grad():
            predicted_sources = [self.source_id2label.get(sid.item(), "UNKNOWN") 
                                for sid in predicted_source_ids]
        
        # 4. Encode predicted source words
        source_inputs = self.melbert_tokenizer(
            predicted_sources,
            max_length=self.args.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        source_inputs = {k: v.to(input_ids.device) for k, v in source_inputs.items()}
        source_target_mask = (source_inputs['input_ids'] != self.melbert_tokenizer.pad_token_id).float()
        
        source_outputs = self.encoder(
            source_inputs['input_ids'],
            attention_mask=source_inputs['attention_mask']
        )
        
        source_sequence_output = source_outputs[0]
        source_target_output = source_sequence_output * source_target_mask.unsqueeze(2)
        
        if self.args.small_mean:
            source_embeddings = source_target_output.mean(1)
        else:
            source_embeddings = source_target_output.sum(dim=1) / source_target_mask.sum(-1, keepdim=True)
        
        # 5. Encode original isolated target words
        target_outputs_2 = self.encoder(
            input_ids_2,
            attention_mask=attention_mask_2
        )
        
        target_sequence_output_2 = target_outputs_2[0]
        target_output_2 = target_sequence_output_2 * target_mask_2.unsqueeze(2)
        
        if self.args.small_mean:
            target_embeddings_2 = target_output_2.mean(1)
        else:
            target_embeddings_2 = target_output_2.sum(dim=1) / target_mask_2.sum(-1, keepdim=True)
        
        return source_embeddings, target_embeddings_2, confidences

    def blend_embeddings(self, source_embeddings, target_embeddings, confidences):
        """
        Blend source and target embeddings based on configuration
        
        Args:
            source_embeddings: [batch_size, hidden_size]
            target_embeddings: [batch_size, hidden_size]
            confidences: [batch_size]
            
        Returns:
            blended_embeddings: [batch_size, hidden_size]
        """
        confidence_weights = confidences.unsqueeze(1)
        
        if self.source_blend_mode == 'additive':
            # ADDITIVE: enhanced = target + alpha * source
            # Keeps target strength, adds source as enhancement
            enhanced = target_embeddings + self.source_alpha * confidence_weights * source_embeddings
            return enhanced
        else:
            # REPLACEMENT: blended = conf * source + (1-conf) * target
            # Original soft confidence approach
            blended = confidence_weights * source_embeddings + (1 - confidence_weights) * target_embeddings
            return blended

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
        Forward pass with configurable source blending
        """
        # ===== ENCODER 1: Target in context =====
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Get target output with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        if self.args.small_mean:
            target_output = target_output.mean(1)
        else:
            target_output = target_output.sum(dim=1) / target_mask.sum(-1, keepdim=True)

        # ===== ENCODER 2: Get source and target embeddings =====
        source_embeddings, target_embeddings_2, confidences = self.predict_source_and_embeddings(
            input_ids, target_mask, attention_mask,
            input_ids_2, target_mask_2, attention_mask_2
        )

        # ===== METAPHOR-ONLY FILTERING (if enabled) =====
        if self.source_use_mode == 'metaphor_only':
            # Get preliminary metaphor score
            # Use simple heuristic based on target context
            prelim_features = torch.cat([pooled_output, target_output], dim=1)
            prelim_hidden = self.SPV_linear(prelim_features)
            prelim_logits = self.classifier(torch.cat([prelim_hidden, prelim_hidden], dim=1))
            prelim_probs = torch.exp(self.logsoftmax(prelim_logits))
            metaphor_scores = prelim_probs[:, 1]  # Probability of metaphor class
            
            # Only use source for samples with high metaphor probability
            use_source_mask = (metaphor_scores > self.metaphor_threshold).float().unsqueeze(1)
        else:
            # Use source for all samples
            use_source_mask = torch.ones(source_embeddings.size(0), 1).to(source_embeddings.device)

        # ===== BLEND: Apply configured blending strategy =====
        blended_embedding = self.blend_embeddings(source_embeddings, target_embeddings_2, confidences)
        
        # Apply metaphor-only mask
        final_embedding = use_source_mask * blended_embedding + (1 - use_source_mask) * target_embeddings_2
        final_embedding = self.dropout(final_embedding)

        # ===== SPV and MIP =====
        if self.args.spv_isolate:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, final_embedding], dim=1))
        else:
            SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output], dim=1))
        
        MIP_hidden = self.MIP_linear(torch.cat([final_embedding, target_output], dim=1))

        # Final classification
        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits
