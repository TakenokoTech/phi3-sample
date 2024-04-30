from dataclasses import dataclass


@dataclass
class Config:
    token = "128k" # "4k"
    k = 5
    p = 0.95
    temperature = 0.95
    repetition_penalty = 2.0
    system_prompt = [
        "I am an excellent assistant AI."
    ]
    user_prompt1 = [
        "Hello, how are you?"
        # "Can you provide ways to eat combinations of bananas and dragonfruits?"
    ]
    assistant_prompt1 = [
        "I am doing great.",
        # "I have a long experience as an engineer.",
        # "My favorite languages are python and kotlin.",
        # "Please ask me anything.",
        # "How can I help you today?"
        # "Sure! Here are some ways to eat bananas and dragonfruits together: ",
        # "1. Banana and dragonfruit smoothie: ",
        # "Blend bananas and dragonfruits together with some milk and honey. ",
        # "2. Banana and dragonfruit salad: ",
        # "Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
    ]
    user_prompt2 = [
        # "Hello."
        # "Tell me what happens if you leave ice in a place exposed to air."
        # "Who are you?"
        # "I would like to hear the Japanese folk tale Momotaro."
        "What is the capital of France?"
    ]
