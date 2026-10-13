# Feature explanations

These explanations use a simplified technical English style. Each feature has three parts: what, why, and how. The source does not define every implementation detail. See [Architecture decisions](architecture.md) for the selected defaults.

## 1. Document input

**What:** The system reads legal text from UTF-8 files. Training files also contain a class label.

**Why:** The model needs document text and a known target for each training example.

**How:** The data reader checks each JSON record. It rejects empty text and unknown labels. Prediction reads one text file.

## 2. Span segmentation

**What:** The system divides a document into at most eight ordered spans.

**Why:** Span groups let the model examine larger parts of a document.

**How:** The processor divides words into balanced groups. A word belongs to one span only. The groups keep the original order. This default does not detect legal sections.

## 3. Token packing

**What:** The processor creates one sequence of at most 384 tokens.

**Why:** LegalBERT needs a sequence within the model limit.

**How:** The tokenizer converts spans to token IDs. The processor adds CLS and SEP markers. It keeps leading tokens when the document is too long. It adds padding when the sequence is too short. Text after the token limit is not used.

## 4. Segment mapping

**What:** Each content token has a span index.

**Why:** The span path must know which token states belong to each span.

**How:** The processor assigns zero-based span indices. It assigns `-1` to special tokens and padding. It removes spans with no retained tokens. The model checks the mapping before computation.

## 5. LegalBERT encoding

**What:** LegalBERT produces token states and a document state.

**Why:** These states carry contextual information from the legal text.

**How:** The model loads `nlpaueb/legal-bert-base-uncased`. Its last hidden layer supplies token states. The first token supplies the CLS document state. Training updates all layers by default. A setting can freeze the lower layers.

## 6. Dual token attention

**What:** Two attention branches produce one token evidence vector per class.

**Why:** Each class can need different token evidence. Two branches can learn different patterns.

**How:** Each branch has learned class queries. The queries attend to visible token states. A learned gate combines the two branch outputs. Padding receives zero attention.

## 7. Span pooling

**What:** The system creates one representation per retained span.

**Why:** The graph path needs one node state for each span.

**How:** The pooling function calculates the mean of the token states in each span. It excludes CLS, SEP, and padding. It returns zero for padded span slots.

## 8. Legal cue counts

**What:** The system counts 12 types of legal text patterns in each retained span.

**Why:** Explicit cue counts can help the model identify useful legal content.

**How:** Case-insensitive regular expressions count each cue type. The defaults cover statutes, precedents, evidence, proof burdens, negation, grants, denials, appeals, liability, remedies, procedure, and obligations. These patterns are features. They are not legal rules. The checkpoint stores the selected patterns.

## 9. Cue-guided span refinement

**What:** A learned gate adds cue information to each span representation.

**Why:** The model must control the effect of cue counts on span evidence.

**How:** A linear layer and a sigmoid calculate a gate from the pooled state and cue counts. A second layer projects counts to the hidden size. The system adds the gated projection to the pooled state. A scale setting controls this addition.

## 10. Semantic graph

**What:** MiniLM supplies span similarity scores.

**Why:** Related spans can share information even when they are far apart.

**How:** Frozen MiniLM embeds the retained span text. The graph uses cosine similarity. Negative scores become zero. MiniLM vectors are not concatenated with classifier inputs. Gradients do not enter MiniLM.

## 11. Order graph and self graph

**What:** The graph also includes span order and self connections.

**Why:** Nearby spans can depend on each other. A node also needs its own state.

**How:** Order scores decrease exponentially with span distance. The identity matrix adds self connections. A learned softmax combines semantic, order, and self scores. Degree normalization scales the final graph. Padded nodes have no edges.

## 12. Graph propagation

**What:** Graph layers update span states.

**Why:** A span can need context from related spans.

**How:** Each layer combines adjacent states through the graph matrix. A linear layer and GELU transform the result. A residual connection and layer normalization keep the span state. The default model uses two layers.

## 13. Span attention

**What:** The graph path produces one span evidence vector per class.

**Why:** Each class can depend on different spans.

**How:** Learned class queries attend to the graph states. A weighted sum produces each vector. Padded spans receive zero attention.

## 14. Gated fusion

**What:** The system combines token evidence, span evidence, and document context.

**Why:** The prediction needs both evidence paths and the overall document state.

**How:** The system concatenates the three vectors to calculate a gate. It applies the gate only to span evidence. It then concatenates token evidence, gated span evidence, and the repeated CLS state.

## 15. Class prediction

**What:** The system returns class scores, probabilities, and one predicted class.

**Why:** A judgment classification task needs one result from the configured class list.

**How:** Each class has its own linear weights and bias. Softmax normalizes the scores across classes. Argmax selects the largest score. Cross-entropy supplies the training loss.

## 16. Training and checkpoint storage

**What:** The command trains the model and saves a reusable checkpoint.

**Why:** New attention, graph, and prediction weights need labeled examples.

**How:** AdamW updates trainable weights. Gradient clipping limits large updates. The command records epoch results. It saves model weights, encoder settings, class names, cue patterns, and the tokenizer. Loading restores the LegalBERT structure locally. MiniLM loads from its configured model location.

## 17. Evaluation and evidence output

**What:** Commands report model scores and optional span attention.

**Why:** Users need to measure performance and inspect model behavior.

**How:** Evaluation counts correct predictions and class errors. It calculates macro F1. Prediction can list retained spans and their attention weights. Real performance must be measured on separate labeled cases.

## 18. Automated checks

**What:** Tests check the architecture and its data contracts. GitHub Actions runs the tests.

**Why:** Model changes can break masking, graph flow, or checkpoint loading.

**How:** Small local encoders exercise forward computation, gradients, optimizer updates, padding, and checkpoint recovery. These checks do not require public model downloads. A separate integration check uses the named public encoders when access is available.
