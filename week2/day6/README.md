# Prompt Engineering — Complete Reference Guide

## 1. What Is Prompt Engineering?

Prompt engineering is the process of designing clear, specific, and structured instructions to guide a Large Language Model (LLM) toward the desired output.

A well-designed prompt helps the model understand:
- Who it should act as.
- What task it must perform.
- What rules it must follow.
- What the output should look like.
- What examples to follow.
- What to do when the input is unclear or outside the scope.

## 2. The 6 Essential Components of a Prompt

Use the following structure as your default starting point:

**ROLE → TASK → CONSTRAINTS → OUTPUT FORMAT → EXAMPLES → FALLBACK**

These components are not mandatory for every prompt. Use the ones that help define the task clearly.

### 1. ROLE — Who should the AI act as?

**Purpose:** Define the model's role, expertise, or perspective.

**Question to ask yourself:** Who should the AI be for this task?

**Example:**
```text
ROLE:
You are an experienced customer support assistant
for a mobile and laptop company.
```

**When to use:** When the task requires a specific perspective, domain expertise, or professional behavior.

**Remember:** Define a role that is relevant to the task. A role does not automatically make the answer correct.

### 2. TASK — What should the AI do?

**Purpose:** Clearly state the action the model must perform.

**Question to ask yourself:** What exactly do I want the AI to do?

**Example:**
```text
TASK:
Classify the customer's complaint into one of
the predefined categories.
```

**When to use:** Always. Every useful task-oriented prompt needs a clear objective.

**Remember:** Use specific action words such as classify, summarize, extract, compare, explain, generate, translate, or evaluate.

Avoid vague instructions such as "Help me with this."

### 3. CONSTRAINTS — What rules must the AI follow?

**Purpose:** Define boundaries, restrictions, and conditions.

**Question to ask yourself:** What must the AI do or avoid doing?

**Example:**
```text
CONSTRAINTS:
- Choose only from Billing, Technical, Return, OTHER.
- Do not invent additional categories.
- Classify based only on the supplied complaint.
- If information is insufficient, use OTHER.
```

**When to use:** When you need predictable behavior, restricted choices, a particular scope, or compliance with specific rules.

**Remember:** State both what is allowed and what is prohibited when necessary.

### 4. OUTPUT FORMAT — How should the answer look?

**Purpose:** Specify the exact structure of the response.

**Question to ask yourself:** What should the final answer look like?

**Example:**
```text
OUTPUT FORMAT:
Return exactly one category name.
Do not include explanations, punctuation,
or additional text.
```

Other possible output formats:

- One word or one category.
- Bullet points.
- A numbered list.
- JSON.
- A Markdown table.
- A fixed number of sentences.
- A specific structure with named fields.

**Example — JSON output:**
```json
{
  "category": "Technical",
  "reason": "The laptop is not working."
}
```

**When to use:** Whenever the response must follow a specific structure or be consumed by another program.

**Remember:** Describe the format explicitly. If you need valid JSON, say so and define the required keys and value types.

### 5. EXAMPLES — What does a correct answer look like?

**Purpose:** Demonstrate the desired behavior through sample inputs and outputs. This technique is called *few-shot prompting* when multiple examples are provided.

**Question to ask yourself:** Can I show the AI what a correct result looks like?

**Example:**
```text
EXAMPLES:

Complaint: I was charged twice for my purchase.
Category: Billing

Complaint: My laptop screen is not turning on.
Category: Technical

Complaint: I want to return my laptop and get a refund.
Category: Return

Complaint: What is the weather today?
Category: OTHER
```

**When to use:** When the task involves classification, extraction, formatting, tone matching, or a pattern that is easier to demonstrate than explain.

**Remember:** Examples should be accurate, representative, and consistent with your constraints.

### 6. FALLBACK — What should the AI do when the normal rules do not apply?

**Purpose:** Define a safe and predictable response for unsupported, ambiguous, incomplete, or out-of-scope inputs.

**Question to ask yourself:** What should happen if the AI cannot confidently complete the task within the rules?

**Example:**
```text
FALLBACK:
If the complaint does not belong to Billing,
Technical, or Return, output OTHER.

If the complaint is too vague to classify reliably,
output OTHER.
```

**When to use:** When unexpected inputs are possible or the task requires a defined default response.

**Remember:** A fallback should tell the model exactly what to return or do. Do not leave the decision ambiguous.

---

## 3. Complete Reusable Prompt Template

Copy this template whenever you start a new prompt engineering task. Replace the placeholders with the details of your task.

```text
# ROLE
You are [define the role, expertise, or perspective].

# TASK
Your task is to [describe the exact action to perform].

# CONTEXT
[Provide the background information needed to understand the task.]

# CONSTRAINTS
- [Rule 1]
- [Rule 2]
- [What the model must avoid]
- [Allowed options, if applicable]

# INPUT
[Insert the actual input or data to process.]

# OUTPUT FORMAT
- Format: [one word / JSON / bullets / table / other]
- Required fields or structure: [specify them]
- Length: [specify if necessary]
- Additional instructions: [such as no explanation]

# EXAMPLES
Input: [representative example]
Output: [correct expected answer]

Input: [another representative example]
Output: [correct expected answer]

# FALLBACK
If [unsupported, ambiguous, or invalid condition],
then [specific action or default response].
```

**Important:** `CONTEXT` and `INPUT` are useful additions, not compulsory components in every prompt. Include examples when they help, and omit irrelevant sections.

## 4. Practical Example — Customer Complaint Classification

Here is the complete template applied to your Groq API project.

```text
# ROLE
You are a customer support assistant for a
mobile and laptop company.

# TASK
Classify the customer's complaint into exactly
one category.

# CONTEXT
The company uses four categories to organize
customer complaints.

# CONSTRAINTS
- Billing: Payment, invoice, or charging issues.
- Technical: Device malfunctions or technical issues.
- Return: Product returns or refund requests.
- OTHER: Complaints that do not fit the other
  categories or cannot be classified reliably.
- Select exactly one category.

# INPUT
Customer complaint: My laptop is not working.

# OUTPUT FORMAT
Return only one category name.
Do not provide explanations or additional text.

# EXAMPLES
Complaint: I was charged twice.
Category: Billing

Complaint: My laptop will not start.
Category: Technical

Complaint: I want to return my product.
Category: Return

Complaint: What is the capital of India?
Category: OTHER

# FALLBACK
If the complaint is unrelated, ambiguous, or
insufficiently detailed for reliable classification,
return OTHER.
```

**Expected output:**
```text
Technical
```

The expected result is `Technical` because the complaint describes a laptop malfunction.

## 5. Additional Components for More Reliable Prompts

The six components above are a useful starting point. For more complex tasks, consider adding these components when relevant.

| Component | What it defines | When to use |
|---|---|---|
| Context | Background information | When the task depends on surrounding details |
| Input | The data to process | When processing documents, complaints, or records |
| Audience | Who will read the result | When adapting explanations to a specific reader |
| Tone | The communication style | When generating emails, reports, or dialogue |
| Procedure | Steps the model should follow | When the task needs a defined workflow |
| Success Criteria | What makes the answer acceptable | When evaluating or generating solutions |
| Uncertainty Handling | What to do when evidence is insufficient | When unsupported assumptions are risky |
| Validation | Checks the output must pass | When outputs feed into software or workflows |

Do not add every component automatically. More instructions are not necessarily better; relevant, consistent, and unambiguous instructions are better.

## 6. Prompt Engineering Checklist

Before sending a prompt to an LLM, check the following:

- [ ] **Role:** Have I defined the role if the task benefits from one?
- [ ] **Task:** Is the requested action clear?
- [ ] **Context:** Have I provided the necessary background?
- [ ] **Constraints:** Have I specified the important rules and boundaries?
- [ ] **Input:** Have I clearly identified the data to process?
- [ ] **Output Format:** Have I specified the expected response structure?
- [ ] **Examples:** Would examples reduce ambiguity?
- [ ] **Fallback:** Have I defined what to do with unsupported or unclear inputs?
- [ ] **Validation:** Can I check whether the result meets my requirements?

## 7. Final Rule to Remember

Before writing a prompt, answer these questions:

1. **WHO?** — Role
2. **DO WHAT?** — Task
3. **UNDER WHICH RULES?** — Constraints
4. **USING WHAT INFORMATION?** — Context and Input
5. **IN WHAT FORM?** — Output Format
6. **WHAT DOES CORRECT LOOK LIKE?** — Examples
7. **WHAT IF IT CANNOT COMPLETE THE TASK?** — Fallback
8. **HOW WILL I CHECK THE RESULT?** — Validation

This is a practical framework, not a universal formula. For simple questions, a direct instruction may be sufficient. For classification, extraction, automation, and structured generation, a more explicit template can make outputs easier to evaluate and use.
