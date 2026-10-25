# Experimental prompts and instructions

The following protocol text accompanies the saved experimental records. Model identifiers and generation settings describe the historical experiments. They are not instructions to execute this document.

LLM configurations and prompts
LLMs experiment conducted using a version based on Synchronous and Asynchronous User-Customizable Environment for Multi-Agent LLM Interaction 57 available here: https://github.com/Deep-Cognition-Lab/SAUCE 
LLM configuration:
Models Used  Main Experiments
David: Claude 4 Opus (claude-opus-4-20250514)
John: o3 (o3-2025-04-16)
Alice: Gemini 2.5 Pro (gemini-2.5-pro-preview-05-06)
Prompting Configuration by Stage
Question Prompt presented before conversation for confidence extraction and post-conversation:
“The following is a multiple-choice question in medicine. Think about it step by step and select the most appropriate option. Format your answer as: The Answer is (X).”
1. Pre-Discussion (Solo Answering Phase for confidence extraction)
System Prompt:
“You are a helpful assistant in the medical field, skilled in answering and discussing questions.”
2. Discussion Phase (Conversation Mode)
System + Instruction Prompt (for each model):
Your name is [Name]. You are a helpful assistant in the medical field, skilled in answering and discussing questions. Your role is to engage in a multi-agent debate about the following multiple-choice question in medicine. Provide your suggested answer, and critically evaluate the answer suggested by another participant. If you believe their answer is incorrect, clearly state why and provide evidence or reasoning to support your claim: [question text]
Generation Parameters
Claude and Gemini: Temperature = 0.8, max tokens = 750
o3: reasoning_effort="high", max_completion_tokens = 10,000
All models: 1 response per turn (no sampling across multiple completions)
Replication Setup (Figure S8)
For the replication experiment using mid-tier LLMs, the following models were used:
David: Claude 3.5 Haiku (claude-3-5-haiku-20241022)
John: GPT-3.5 Turbo (gpt-3.5-turbo)
Alice: Gemini 1.5 Flash (gemini-1.5-flash-002)
Answer extraction:
We used the EleutherAI/lm-evaluation-harness multiple-choice extraction library  v0.4.0, available at: [https://github.com/EleutherAI/lm-evaluation-harness], with customized regular expressions tailored to capture over 99% of generated response formats.
Confidence extraction and temperature checks
In Experiments 1–2, each model was run five times using the prompt and configuration detailed under LLM configurations and prompts. To reduce positional bias and mitigate memorization effects based on choice order, the answer options were randomly shuffled in each repetition. Confidence was then estimated by computing the frequency of the most commonly selected textual answer (rather than its letter label, due to shuffling), effectively capturing the model's self-consistency across runs. In Experiment 4, confidence instead counts how many of ten independent responses support the model's opening answer; this count is halved and rounded up to give a 0–5 score.
Temperature checks: For Claude 4 and Gemini 2.5 Pro we repeated the confidence extraction at temperatures 0.2 and 0.5. For OpenAI we used GPT-4.1, which allows temperature control, under the same settings. Calibration, item level rank correlations, Potential Conversation Synergy ordering, and the ability of Potential Conversation Synergy to predict conversational gains were consistent across temperatures (Fig. S1, S2, S3, S4).
Human–LLM chatbot configuration
We used GPT-4 Turbo, accessed via a custom Telegram client designed to capture conversations and control both the environment and the agent. The Telegram client reset the conversation history whenever the user triggered a predefined signal to initiate a new question for discussion. The model operated using the following prompts, specifically engineered for the conversational setting:
Initial prompt:
sys_prompt = "you are writing a text message to a student who is your colleague, discussing the following question: {question}. Please keep your answers relatively short but explain yourself. provide answers only in Hebrew. Make sure your answer is related to what the user wrote in his last message for example if the user wrote that he thinks that the answer is ב you either explain why it is true if you think it's right or explain why it is wrong if you think otherwise. if he ask what do you think about one of the options you give your opinion and don't hesitate to correct him if he is wrong, he might be confused"
In this prompt, the Hebrew answer label ב denotes option B.
In conversation prompt:
sys_prompt = "you are writing a text message to a student who is your colleague, discussing the following question: {question}. Please keep your answers short as possible. provide answers only in Hebrew and explain your reasoning a little, no need to call the person by its name more than once. Make sure your answer is related to what the user wrote in his last message."
The post-conversation model answer was extracted using the exact same prompts as in the LLM-LLM configurations (LLM configurations and prompts): the System + Instruction Prompt from the Discussion Phase, along with the Question Prompt presented after incorporating all user and assistant messages.
Participant instructions
Instructions were presented in Hebrew; the following is an English translation. The questionnaire interface and its translation are shown in Figure S10.
Human-Trio Experiment Instructions
Stage 1 (Individual Task) Instructions:
In the first part of the experiment, you will be required to answer multiple-choice questions with two options in the field of internal medicine.
The test consists of 42 questions and is allotted approximately one hour to complete. You will not be able to go back and change your answer to a question after proceeding to the next one.
You must answer the questions without using any aids, including internet searches or consulting with others.
You must select an answer for each question in order to move on to the next one.
Additionally, for each question, you will be asked to indicate your level of confidence in your answer.
Stage 2 (Group Discussion Task) Instructions:
In the second part of the experiment, you will be required to answer the same questions from the first part again. On the next page, you will find a link to a WhatsApp group. Other participants in the experiment will be added to the group. You may freely discuss the test questions within the group before answering each question.
To keep the discussion organized, you are required to go through the questions in the order they appear. One participant should write the question number (in the following format: Question Number) before moving on to the next question.
You are required to answer all questions again, but you are not required to reach agreement on the answer to each question — you should submit the answer you personally believe is most correct.
WhatsApp Group Rules:
In the second part of the experiment, you will be required to answer the same multiple-choice questions again, each with two options, in the field of internal medicine.
The test consists of 42 questions and is allotted approximately one hour to complete. You will not be able to go back and change your answer to a question after moving on to the next one.
You must answer the questions without using any aids, including internet searches or consulting with others, except for discussions within the designated group.
You must select an answer for each question in order to proceed to the next one.
Additionally, for each question, you will be asked to indicate your level of confidence in your answer.
Maintain an organized discussion. At the beginning of each question, make sure that one of the participants writes the transition to the next question in the following format: Question [number] in a separate message (e.g., Question 1).
Avoid using the reply feature (Reply) in WhatsApp and do not use emojis during the discussion.
Throughout the conversation, you are asked to state your level of certainty when presenting your opinion and chosen answer.
Human-LLM Experiment Instructions
Stage 1 (Individual Task) Instructions:In the first part of the experiment, you will be required to answer multiple-choice questions with four options in the field of pediatrics.The test consists of 50 questions and you will have only one hour to complete it. You will not be able to go back and change your answer to a question after proceeding to the next one. You must answer the questions without using any aids, including internet searches or consulting with others. You must select an answer for each question in order to move on to the next one. All questions must be answered within one hour from the start of the test. Additionally, for each question, you will be asked to indicate your level of confidence in your answer.
Stage 2 (Bot Discussion Task) Instructions:In the second part of the experiment, you will be able to use a chatbot designed to help you solve the test questions. You can access it via the Telegram link provided in the initial experiment message. Before you begin, you must provide the bot with the code: "medcom". You must tell the bot which question you are referring to by writing “Question x” in order to discuss question x. You are required to follow this instruction when moving between questions.
The conversation with the bot is open and unrestricted. If the bot asks for the content of the question, you can simply write the prompt to move to the question again (“Question x”).You are required to answer all questions again and are not required to agree with the bot on each answer—you should submit the answer you personally believe is most correct.
 You are not obligated to discuss every question with the bot, but you cannot go back to a previous question in the test form, and you must complete the test within one hour.
You should begin the second part approximately one hour after the start of the experiment. Please avoid closing the experiment window until the end of this part and the conclusion of the experiment.
Stage 2 Instructions:
In the second part of the experiment, you will be required to answer the same multiple-choice questions again, each with four options, in the field of pediatrics.
The test consists of 50 questions, and you will have one hour to complete it. You will not be able to go back and change your answer to a question after moving on to the next one.
You must answer the questions without using any aids, including internet searches or consulting with others, except for discussions within the designated group.
You must select an answer for each question in order to proceed to the next one.
Additionally, for each question, you will be asked to indicate your level of confidence in your answer.
