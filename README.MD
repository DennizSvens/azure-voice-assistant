# Voice Assistant with Azure Cognitive Speech Services, Azure OpenAI (GPT-4) and Langchain

This project demonstrates how to build a voice assistant using Azure Cognitive Speech Services, Azure OpenAI GPT-4, and Langchain. The voice assistant can perform a variety of tasks such as opening applications, fixing bugs in code, locking your computer, and fetching stock prices.

The voice assistant use Langchain Agents to perform task against the terminal and run Python code. You can easily add more tools by adding tools to load_tools(). 

Read more about Langchain and Langchain agents here: https://python.langchain.com/en/latest/modules/agents.html

## Example Prompts

Once the voice assistant is running, you can use the following example prompts to interact with it:

* "Hey Astra, open Microsoft Edge."
* "Hey Astra, fix the bug in the code in my clipboard and set the solution to my clipboard."
* "Hey Astra, lock my computer."
* "Hey Astra, what is the stock price of MSFT?"
* "Hey Astra, what time is it?"

It can also respond to any general queries that the GPT model is capable of answering. However, its capabilities extend beyond that. Since it has access to the terminal and Python through a Langchain agent, it can also interact with your system.


## Prerequisites

- Azure account
- Azure Speech Service: https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/
- Azure OpenAI Service: https://azure.microsoft.com/en-us/products/cognitive-services/openai-service

## Setup

1. Clone this repository to your local machine.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a .env file in the root directory of the project and fill it with your own keys and other details. Use the example.env file in the repository as a reference.

```properties
# Azure Cognitive Speech Services
AZURE_SPEECH_KEY="Replace with your subscription key for Azure Speech Service"
AZURE_SPEECH_REGION="Replace with the region for your Azure Speech Service"
AZURE_WAKEUP_MODEL="astra.table" # This is the default wakeup keyword model for "Hey Astra"
AZURE_SPEECH_VOICE="en-US-AriaNeural" # This is the voice used by Azure Speech Service

# Azure OpenAI GPT-4
OPENAI_API_MODEL="gpt-4-32k" # This can be changed to any other available GPT model that has chat capabilities
OPENAI_API_DEPLOYMENTNAME="Replace with the name of the model you created in Azure Portal"
OPENAI_API_VERSION="2023-03-15-preview" # This is the version of the OpenAI API
OPENAI_API_BASE="https://REPLACETHISWITHYOURPREFIX.openai.azure.com/" # Replace with your OpenAI API base URL
OPENAI_API_KEY="Replace with your Azure OpenAI API key"
OPENAI_API_TYPE="azure" # This is the type of the OpenAI API
```

4. Run the script: 
```bash
python main.py
```

## Usage
Once the voice assistant is running, it will continuously listen for the wakeup keyword "Hey Astra". Once the keyword is detected, it will start listening for speech input, which it will then pass to the OpenAI model for processing. The model's response will be spoken out loud using text-to-speech.

## Contributions
Contributions are welcome! Please feel free to submit a Pull Request.
