import os
import logging
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, initialize_agent, AgentType

# Load environment variables
load_dotenv()
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
OPENAI_API_DEPLOYMENTNAME = os.getenv("OPENAI_API_DEPLOYMENTNAME")
AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION')
AZURE_WAKEUP_MODEL = os.getenv('AZURE_WAKEUP_MODEL')
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
  """Agent class for handling OpenAI interactions"""
  def __init__(self):
    self.chat = AzureChatOpenAI(model_name=OPENAI_API_MODEL, deployment_name=OPENAI_API_DEPLOYMENTNAME)
    self.tools = load_tools(["python_repl", "terminal"], llm=self.chat)
    self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    self.agent = initialize_agent(self.tools, self.chat, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=False, memory=self.memory)
  
  def ask_question(self, question):
    """Ask a question to the OpenAI model"""
    logger.info("[AZURE OPENAI]: Asking question: " + question)
    return self.agent.run(input=question)

class Recognizer:
  """Recognizer class for handling Azure speech recognition and synthesis"""
  def __init__(self, agent):
    self.subscription_key = AZURE_SPEECH_KEY
    self.region = AZURE_SPEECH_REGION
    self.model_path = AZURE_WAKEUP_MODEL
    self.speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
    self.speech_config.speech_recognition_language = 'en-US'
    self.keyword_recognizer = speechsdk.KeywordRecognizer(self.speech_config)
    self.keyword_model = speechsdk.KeywordRecognitionModel(self.model_path)
    self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config)
    self.agent = agent
    self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
    self.speech_config.speech_synthesis_voice_name = AZURE_SPEECH_VOICE
  def handle_speech(self, text):
    """Handle recognized speech and respond using text-to-speech"""
    logger.info("[AZURE SPEECH RECOGNITION]: Recognized speech: " + text)
    response = self.agent.ask_question(text)
    logger.info("[AZURE OPENAI]: Received response: " + response)
    logger.info("[AZURE TTS]: Starting Text To Speech")
    self.speech_synthesizer.speak_text_async(response).get()
    self.listen_for_keyword()

  def listen_for_speech(self):
    """Listen for speech input"""
    logger.info("[AZURE SPEECH RECOGNITION]: Listening for input")
    result = self.speech_recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
      self.handle_speech(text=result.text)

  def listen_for_keyword(self):
    """Listen for a specific keyword to activate speech recognition"""
    logger.info("[AZURE SPEECH RECOGNITION]: Listening for wakeup keyword")
    result = self.keyword_recognizer.recognize_once_async(model=self.keyword_model).get()
    if result.reason == speechsdk.ResultReason.RecognizedKeyword:
      logger.info("[AZURE SPEECH RECOGNITION]: Wakeup word detected")
      self.listen_for_speech()

try:
  agent = Agent()
  recognizer = Recognizer(agent)
  recognizer.listen_for_keyword()
except Exception as e:
  logger.error("An error occurred: ", exc_info=True)
finally:
  input("Press any key to exit...")