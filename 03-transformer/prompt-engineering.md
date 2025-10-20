# PROMPT ENGINEERING

### **1. Definition & Purpose**

* Designing **prompts** to guide LLM responses.
* Prompts = instructions, statements, or examples.
* Purpose: Increase **accuracy**, **usefulness**, and **safety**.
* Involves **experimentation**; no perfect prompt.

### **2. Basic Prompt Example (Page 59)**

**Input:** “The sky is”
**Output:** “blue.”
→ Illustrates simplest completion pattern.

### **3. Components of a Prompt (Page 60)**

* **Instruction:** What the model should do.
* **Context:** Background info or setting.
* **Input Data:** The main text or task data.
* **Output Indicator:** Expected format or answer label.
  Example:
  “Classify the text into *neutral*, *negative*, or *positive*.”

### **4. Key Settings (Page 61)**

* **Temperature:** Controls creativity/randomness.
* **top_p (nucleus sampling):** Controls probability distribution cutoff.

  * Low values → deterministic outputs.
  * High values → diverse responses.

### **5. Structured Prompting (Pages 62–63)**

* Improves reliability by clearly defining structure:

  * Explicit **instruction**, **input**, and **output indicator**.
  * Example:

    * ❌ “Is this sentence positive?”
    * ✅ “Classify this sentence as positive or negative: ‘The movie was amazing!’”

### **6. Instruction-Based Prompting (Pages 64–65)**

* Tailors prompts for **task-specific goals**: summarization, classification, NER, etc.
* **Best Practices:**

  * Be **specific** (“write a two-sentence formal summary”).
  * Reduce **hallucinations** (“respond with ‘I don’t know’ if unsure”).
  * Mind **order** (primacy & recency effects → key instructions at start/end).

### **7. Advanced Prompt Components (Pages 66–67)**

* **Persona:** Defines LLM’s role.
* **Instruction:** Defines the task.
* **Context:** Adds background info.
* **Format:** Specifies output layout.
* **Audience:** Defines who it’s for.
* **Tone:** Specifies style.
* **Data:** Main content to process.
  → Combining all improves precision and control.

### **8. In-Context Learning (Page 68)**

* **Zero-shot:** No examples, only instruction.
* **One-shot:** One example included.
* **Few-shot:** Several examples provided.
  → Teaches model “by example” without weight updates.
  (*Also shown with GPT-2 performance on benchmarks, page 69.*)

### **9. Chain-of-Thought (CoT) Prompting (Pages 70–74)**

* Encourages **step-by-step reasoning** like humans.
* “Explain your answer step-by-step.”
* Improves logical accuracy and multi-step tasks.
* **Zero-shot CoT (Kojima et al., 2022):**

  * Model triggers reasoning internally (“Let’s think step-by-step”).
  * Outperforms regular zero-shot and manual CoT.
* **Human vs LM-designed CoT prompts:** LM-designed (e.g., “Let’s work this out step-by-step”) achieved **~82% accuracy**.

### **10. Few-Shot Learning (Page 75)**

* Multiple examples guide task execution.
* Known as **in-context learning**; no gradient updates required.
* Demonstrated in **GPT-3 (Brown et al., 2020)**.

### **11. Model Reasoning Techniques (Page 76)**

* **Chain-of-Thought (CoT):** Sequential reasoning steps.
* **Self-Consistency:** Multiple reasoning paths → majority voting.
* **Tree-of-Thought:** Exploratory reasoning trees (not detailed in slides).
* Benefits:

  * Multi-step reasoning
  * Stable outputs
  * Improved accuracy.

### **12. Self-Consistency (Pages 78–79)**

* Run model **n times**, vary randomness, and **vote** on outputs.
* Reduces variance in reasoning.
* Improves accuracy by balancing creativity and consistency.

### **13. Grammar-Constrained Sampling (Page 80)**

* **Goal:** Prevent undesired outputs during generation.
* **Tools:** Guidance, Guardrails, LMQL.
* **Method:** Apply **constraints during token selection** (not post-checks).
* Influenced by **temperature** and **top_p**.
* Enforces format, safety, and correctness inline.
