from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
MONGODB_STRING = os.getenv('MONGODB_STRING')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME')

def connect_2_db():
    """
    Connects to the MongoDB database using credentials from the environment 
    variables and returns references to the 'users' and 'message_history' collections.

    Returns:
        tuple: A tuple containing references to the 'users' and 'message_history' 
        collections in the MongoDB database.
    
    Raises:
        Exception: If the connection to the MongoDB database fails.
    """
    try:
        # Connect to MongoDB
        db_name = quote_plus(MONGODB_DB_NAME)
        url = MONGODB_STRING
        client = MongoClient(url)
        db = client[MONGODB_DB_NAME]
        users = db["users"]
        message_history = db["message_history"]
        return users, message_history
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def save_message_to_db(user_id, user_text, model_res):
    """
    Saves a user's message and the corresponding bot response to the 
    'message_history' collection in the MongoDB database.

    Args:
        user_id (str): The ID of the user.
        user_text (str): The text input from the user.
        model_res (str): The bot's response to the user.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        # Connect to the 'message_history' collection
        _, message_history = connect_2_db()
        new_messages = [{'user': user_text},
                        {'bot': model_res}]
        
        # Append messages to an existing conversation or create a new conversation
        result = message_history.update_one(
            {'user_id': user_id},
            {'$push': {'messages': {'$each': new_messages}}},
            upsert=True
        )
        
        # Log success and return True if the operation was successful
        if result.modified_count > 0 or result.upserted_id is not None:
            logger.info(f"Message history for user_id {user_id} updated successfully.")
            return True
        else:
            logger.warning(f"No changes made to message history for user_id {user_id}.")
            return False
    except Exception as e:
        logger.error(f"Failed to save messages to MongoDB for user_id {user_id}: {e}")
        return False
