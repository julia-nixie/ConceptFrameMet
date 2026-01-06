# ConceptFrameMet


This repository contains the implementation of `ConceptFrameMet`, a framework for incorporating source domain information into metaphor detection models. The model extends MelBERT with flexible strategies for blending source domain predictions with target word embeddings.

## Overview

Metaphors often involve mapping concepts from a **source domain** (e.g., "war", "journey") to a **target domain**. This model uses a QA-based approach to predict source domains and integrates them adaptively into the metaphor detection pipeline.


## Architecture

```
INPUT SENTENCE → [Context Encoder] → target_context_embedding
                                             ↓
TARGET WORD → [QA Model] → Source Prediction + Confidence
                                             ↓
SOURCE WORD → [Isolated Encoder] → source_embedding
TARGET WORD → [Isolated Encoder] → target_embedding
                                             ↓
                        [ADAPTIVE BLENDING]
                   (replacement or additive mode)
                                             ↓
                        enhanced_embedding
                                             ↓
              [SPV + MIP] → [Classifier] → Metaphor Label
```



For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

This work builds upon the MetaphorFrame project and MelBERT architecture.
