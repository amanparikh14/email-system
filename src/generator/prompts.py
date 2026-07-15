GEN_SYSTEM = """You are a support agent replying to customer emails on behalf \
of the company.

You will be shown past examples of (customer email -> agent reply). Study \
them only for TONE, STRUCTURE, and how we handle this KIND of request. Do \
NOT copy specific facts from the examples: do not reuse dates, names, order \
numbers, prices, confirmation numbers, or ticket-specific claims that belong \
to a different customer's situation.

Rules:
1. Address the customer's actual issue in this email, not the example's issue.
2. If the customer asks for a specific fact you cannot reliably know (an \
order status, an account balance, whether something was already resolved), \
draft a reply that asks the customer to confirm/provide the detail, or tells \
them you're checking -- never invent the fact. Draft, don't fabricate.
3. End with a clear next step for the customer.
4. Sign off as "The Support Team".
5. Keep the reply focused -- do not pad it with unrelated content from the \
examples.
"""

FEWSHOT_USER_TEMPLATE = """Here are {n} past examples of similar tickets and \
how we replied, for tone and structure reference only:

{examples_block}

Now write a reply to this new customer email. Remember: match the tone and \
structure above, but every fact in your reply must come from THIS email, not \
the examples.

New customer email:
{email}
"""

EXAMPLE_BLOCK_TEMPLATE = """--- Example {n} (category: {category}) ---
Customer email:
{email}

Agent reply:
{reply}
"""

# Used when retrieval fires the fallback signal (no store row is similar
# enough). These are fixed, generic examples that convey voice/politeness
# only -- no domain-specific precedent to (mis)copy from.
FALLBACK_EXAMPLES = [
    {
        "category": "generic",
        "email": "Subject: Question about my account\n\nHi, I have a question about a recent charge on my account. Can someone help?",
        "reply": (
            "Hi,\n\nThanks for reaching out. I'd be happy to help look into that charge. "
            "Could you confirm the approximate date and amount so I can check the right "
            "transaction? Once I have that, I'll get back to you with details.\n\n"
            "The Support Team"
        ),
    },
    {
        "category": "generic",
        "email": "Subject: Not happy with my experience\n\nThis has been frustrating and I'd like it resolved quickly.",
        "reply": (
            "Hi,\n\nI'm sorry to hear this has been frustrating -- that's not the experience "
            "we want for you. I want to get this resolved as quickly as possible. Could you "
            "share a bit more detail on what happened so I can look into it directly?\n\n"
            "The Support Team"
        ),
    },
]


def build_fewshot_prompt(email: str, examples: list[dict]) -> str:
    blocks = "\n".join(
        EXAMPLE_BLOCK_TEMPLATE.format(
            n=i + 1, category=ex["category"], email=ex["email"], reply=ex["reply"]
        )
        for i, ex in enumerate(examples)
    )
    return FEWSHOT_USER_TEMPLATE.format(n=len(examples), examples_block=blocks, email=email)
