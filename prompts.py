from transformers import AutoTokenizer

SYSTEM_PROMPTS = {}
SYSTEM_PROMPTS["normal"] = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
SYSTEM_PROMPTS["qa"] = "You are an expert in retrieval QA. Please answer strictly based on the provided context and options. Return the exact answer only; do not be verbose or provide extra information."
# SYSTEM_PROMPTS["extract"] = "You are a precise and reliable information extractor. Only extract facts that appear in the context verbatim. Do not infer or add any information that is not explicitly stated in the context."
SYSTEM_PROMPTS["extract"] = (
    "You are a precise and reliable information extractor. "
    "Extract only short, atomic, verifiable facts. "
    "Do not infer, explain, speculate, or add information."
)
SYSTEM_PROMPTS["qa-cot"] = """You are an expert retrieval QA decision maker.

Strictly follow these rules:
1) Use only the provided Question, Self-Facts, Retrieved Evidence, and Options.
2) Compare whether the evidence supports, contradicts, or is insufficient for each option.
3) Think step-by-step internally, but DO NOT reveal long chain-of-thought.
4) Return exactly one JSON object with two keys: "Reason" and "Answer".
5) "Answer" MUST be copied VERBATIM from one of the provided options.
6) If evidence is insufficient or conflicting, choose a NOTA/Unknown option only when such an option appears in Options.
7) No extra text before or after the JSON.

Example:
{"Reason": "brief evidence-based diagnosis", "Answer": "<paste exactly one option string>"}"""

SYSTEM_PROMPTS["facts"] = (
    "You are a precise knowledge mining assistant. "
    "Extract only atomic, verifiable facts in short sentences, one per line. "
    "No speculation, no paraphrase, no extra commentary."
)


class PromptGenerator:
    def __init__(self, llm_type, task:str="normal", tokenizer=None):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        if task=="qa":
            self.system_prompt = SYSTEM_PROMPTS["qa"]
        elif task=="extract":
            self.system_prompt = SYSTEM_PROMPTS["extract"]
        elif task=="facts":
            self.system_prompt = SYSTEM_PROMPTS["facts"]
        elif task=="qa-cot":
            self.system_prompt = SYSTEM_PROMPTS["qa-cot"]
        else:
            self.system_prompt = SYSTEM_PROMPTS["normal"]

    # def get_system_prompt(self):
    #     return self.system_prompt


    def _generate_prompt(self, user_prompt):
        return user_prompt
        # return self._generate_prompt_apply_chat_template(user_prompt)
    
    # def _generate_prompt_apply_chat_template(self, user_prompt):
    #     messages = [
    #         {"role": "system", "content": self.system_prompt},
    #         {"role": "user", "content": user_prompt}
    #         ]
    #     prompt = self.tokenizer.apply_chat_template(
    #             messages,
    #             tokenize=False,
    #             add_generation_prompt=True
    #         )
    #     return prompt
    
    def generate_context_directly_prompt(self, user_query):
        prompt = """
        Generate a background document from Wikipedia to answer the given question:
        {question}. Keep the length of the document around 100 words
        """
        return self._generate_prompt(prompt.format(question=user_query))

    # def generate_context_by_factual_knowledge(self, user_query, factual_knowledge):
    #     prompt = """
    #     Given the following question and a set of factual knowledge, generate a background document from Wikipedia that can answer the given question. Keep the length of the document around 100 words.
    #     Question: {question}
    #     Factual Knowledge: {factual_knowledge}
    #     Background Document:
    #     """
    #     return self._generate_prompt(prompt.format(question=user_query, factual_knowledge=factual_knowledge))
    def generate_context_by_factual_knowledge(self, user_query, factual_knowledge):
        prompt = """
    Task:
    Construct a short controlled self-context from the given question and atomic self-knowledge.
    
    Rules:
    1. Use only the provided atomic self-knowledge.
    2. Do not introduce new facts.
    3. Do not answer the question directly.
    4. Keep the context concise and factual.
    5. The context is used only for aligning model prior knowledge with external evidence.
    
    Question:
    {question}
    
    Atomic Self-Knowledge:
    {factual_knowledge}
    
    Controlled Self-Context:
    """
        return self._generate_prompt(
            prompt.format(question=user_query, factual_knowledge=factual_knowledge)
        )

    def generate_factual_knowledge(self, user_query):
        prompt = """
    Task:
    Externalize the model's prior knowledge needed to answer the question.
    
    Rules:
    1. List only atomic, short, and verifiable facts or premises.
    2. One fact per line.
    3. Do not provide explanations.
    4. Do not answer the question directly.
    5. Do not add speculation.
    6. Do not include phrases like "this is needed because".
    
    Question:
    {question}
    
    Atomic Self-Knowledge:
    1.
    """
        return self._generate_prompt(prompt.format(question=user_query))

    def generate_context_extract(self, user_context):
        prompt = """
    Task:
    Extract atomic self-facts from the controlled self-context.
    
    Rules:
    1. Extract only facts explicitly stated in the context.
    2. One atomic fact per line.
    3. No explanation.
    4. No inference.
    5. No duplicate facts.
    
    Controlled Self-Context:
    {context}
    
    Atomic Self-Facts:
    1.
    """
        return self._generate_prompt(prompt.format(context=user_context))
    def generate_qa_prompt(self, context, question, options = None, facts = None):
        normal_w_facts_prompt = """
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

        Follow the steps:
        1. Analyze the **Question** carefully.
        2. Use the **Facts** to provide a clear and accurate answer to the question.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
        
        Example:

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Answer: Fluorine
        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Answer:
        """
        choices_w_facts_prompt = """
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.
        
        Follow the steps:
        1. Analyze the **Question** and the **Options**.
        2. Use the **Facts** to select the most accurate answer from the **Options**.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
        4. Please directly answer the option you want to choose. No modification is allowed.
        
        Example:
        
        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        Answer: Fluorine

        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Options:
        {options}

        Answer:
        """
        normal_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Answer: Fluorine

        New Example:

        Question:
        {question}

        Context:
        {context}

        Answer:
        """
        choices_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:
        Oxygen
        Chlorine
        Fluorine

        Answer: Fluorine

        New Example:

        Question:
        {question}

        Context:
        {context}

        Options:
        {options}

        Answer:
        """
        if options is None:
            if facts is None:
                return normal_wo_facts_prompt.format(question=question, context=context)
            else:
                return normal_w_facts_prompt.format(question=question, context=context, facts=facts)
        else:
            if facts is None:
                return choices_wo_facts_prompt.format(question=question, context=context, options=options)
            else:
                return choices_w_facts_prompt.format(question=question, context=context, options=options, facts=facts)

    def generate_qa_prompt_normal_cot(self, context, question, options=None, facts=None):
        options_text = "\n".join(
            [f"{i+1}. {opt}" for i, opt in enumerate(options or [])]
        )
    
        prompt = """
    Task:
    Answer the question by making a calibrated decision under possible knowledge conflict.
    
    Question:
    {question}
    
    Self-Facts:
    {facts}
    
    Retrieved Evidence:
    {context}
    
    Options:
    {options}
    
    Decision rules:
    1. Compare the question, self-facts, retrieved evidence, and options.
    2. Select the option best supported by the retrieved evidence.
    3. If evidence is conflicting or insufficient, select a NOTA/Unknown option only if it appears in Options.
    4. The final Answer must be copied exactly from Options.
    5. Return only JSON.
    6. Do not output option letters such as A/B/C/D unless the option itself is exactly a letter.
    
    JSON:
    {{"Reason": "brief evidence-based diagnosis", "Answer": "copy one option exactly"}}
    """
        return self._generate_prompt(
            prompt.format(
                question=question,
                facts=facts or "",
                context=context or "",
                options=options_text
            )
        )
        
        
    def generate_qa_prompt_schedule_cot(self, context, question, facts, options=None, task='multiple-choice',example=True):
        # 在提示中添加 "json" 关键词
        prompt = """
        请以 JSON 格式返回答案。
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options...
        """
        normal_w_facts_prompt = """
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.

        Follow the steps:
        1. Analyze the **Question** carefully.
        2. Use the **Facts** to provide a clear and accurate answer to the question.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
        4. Please return in JSON format: Reason: (reason) Answer:(answer)


        Example:

        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        CoT-Answer:
        {{
        "Reason":  
        1. [Fact Analysis]: Facts explicitly state "Fluorine is the most electronegative element"  
        2. [Initial Judgment]: Based on the explicit statement in the facts, fluorine is identified as the element with the highest electronegativity.  
        3. [Context Check]: The context provides additional support by mentioning the Pauling scale and comparing fluorine with chlorine, reinforcing that fluorine has the highest electronegativity. No further information is needed. 
        4. [Final Verification]: The facts and context are consistent and conclusive, confirming that fluorine is the correct answer. 
        "Answer": Fluorine
        }}
        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        CoT-Answer:
        """
        choice_w_facts_prompt = """
        Task Description: 
        Given facts,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the facts and context.
        
        Follow the steps:
        1. Analyze the **Question** and the **Options**.
        2. Use the **Facts** to select the most accurate answer from the **Options**.
        3. Refer to the **Context** If the facts do not contain enough information to answer the question, or if additional information is needed.
        4. Please return in JSON format: Reason: (reason) Answer:(answer)
        
        Example:
        
        Question:  
        Which element has the highest electronegativity?  

        Facts:  
        Electronegativity increases across periods and decreases down groups. Fluorine is the most electronegative element.  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        Please return in JSON format.
        CoT-Answer:
        {{
            "Reason":  
            1. [Fact Analysis]: Facts explicitly state "Fluorine is the most electronegative element"  
            2. [Option Matching]: Option C directly matches the factual declaration  
            3. [Context Check]: No contextual supplementation needed - Facts provide conclusive evidence  
            4. [Final Verification]: No conflicting information; perfect match with Option C  
            "Answer": Fluorine
        }}

        New Example:

        Question:
        {question}

        Facts:
        {facts}

        Context:
        {context}

        Options:
        {options}

        CoT-Answer:
        """

        normal_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Please return in JSON format.
        
        CoT-Answer:
        {{
            "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.
            "Answer": Fluorine
        }}

        New Example:

        Question:
        {question}

        Context:
        {context}

        CoT-Answer:
        """

        choice_wo_facts_prompt = """
        Question:  
        Which element has the highest electronegativity?  

        Context:  
        The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.  

        Options:  
        Oxygen  
        Chlorine  
        Fluorine 

        CoT-Answer:
        {{
        "Reason": According to the facts, electronegativity increases across periods and decreases down groups, and it is stated that fluorine is the most electronegative element. The context also supports this by mentioning that chlorine, which is in the same group as fluorine, has a lower electronegativity due to its larger atomic radius. Therefore, based on the given information, fluorine has the highest electronegativity.
        "Answer": Fluorine
        }}
        New Example:

        Question:
        {question}

        Context:
        {context}

        Options:
        {options}

        CoT-Answer:
        """
        if options is None:
            if facts is None:
                return normal_wo_facts_prompt.format(question=question, context=context)
            else:
                return normal_w_facts_prompt.format(question=question, facts=facts, context=context)
        else:
            if facts is None:
                return choice_wo_facts_prompt.format(question=question, context=context, options=options)
            else:
                return choice_w_facts_prompt.format(question=question, facts=facts, context=context, options=options) 