from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from database import save_message_to_db, connect_2_db
import os
from dotenv import load_dotenv
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
default_prompt = os.getenv('DEFAULT_PROMPT', "Default Prompt: ")

def chain_setup(user_id, user_name):
    """
    Sets up the conversation chain for a specific user, initializing the memory 
    with past interactions if available.

    Args:
        user_id (str): The user's ID, used to fetch conversation history.
        user_name (str): The user's name, used in the conversation prompt.

    Returns:
        ConversationChain: A configured conversation chain for the user, ready 
        to process new input and generate responses.
    """
    memory = ConversationBufferMemory()

    # Connect to the database and fetch the message history for the user
    _, message_history = connect_2_db()

    try:
        # Retrieve the conversation history from the database
        conv = message_history.find_one({'user_id': user_id})
        
        if conv:
            messages = conv['messages']
            num_messages = len(messages)
            start_index = max(num_messages - 5, 0)  # Consider only the last 5 messages

            # Add the conversation history to the memory buffer
            for i in range(start_index, num_messages):
                message = messages[i]
                if 'user' in message:
                    memory.chat_memory.add_user_message(message['user'])
                elif 'bot' in message:
                    memory.chat_memory.add_ai_message(message['bot'])
        else:
            logger.info(f"No previous conversation history found for user_id {user_id}.")
    except Exception as e:
        logger.error(f"Error fetching conversation history for user_id {user_id}: {e}")
        raise

    # Set up the ChatOpenAI model using the environment variables
    chat = ChatOpenAI(
        temperature=0.75,
        model=os.getenv("OPENAI_MODEL", "default-model"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "default-api-key")
    )

    # Set AI and human prefixes, using environment variables or defaults
    memory.ai_prefix = os.getenv('AI_PREFIX', 'AI')
    memory.human_prefix = os.getenv('HUMAN_PREFIX', 'User')

    # Create the conversation prompt template, incorporating the user's name
    template = default_prompt + f"""
    Our current dialogue:
    {{history}}
    {user_name}: {{input}}
    AI: 
    """
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    # Initialize and return the conversation chain
    conversation = ConversationChain(
        prompt=prompt,
        llm=chat,
        verbose=True,
        memory=memory
    )

    return conversation

def get_chain_response(user_id, user_text, user_name):
    """
    Processes user input by setting up a conversation chain and generating a response 
    from the AI model.

    Args:
        user_id (str): The user's ID, used to identify conversation history.
        user_text (str): The user's input text that needs to be processed.
        user_name (str): The user's name, used in the conversation prompt.

    Returns:
        str: The AI's response to the user's input.
    """
    try:
        # Set up the conversation chain for the user
        conv_chain = chain_setup(user_id=user_id, user_name=user_name)
        
        # Process the user's input and generate a response
        out = conv_chain(user_text)
        
        # Log the conversation history for debugging purposes
        logger.debug(f"Conversation history for user_id {user_id}: {out['history']}")
        
        # Return the generated response
        return out['response']
    except Exception as e:
        # Log any errors encountered during response generation
        logger.error(f"Error generating response for user_id {user_id}: {e}")
        
        # Return a fallback error message
        return "I'm sorry, there was an error processing your request."
