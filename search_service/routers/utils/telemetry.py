import time
import tiktoken
from typing import Optional, Dict, Any, List
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import pickle
import json


def count_tokens(text: str, model_name: str = "text-embedding-ada-002") -> int:
    """
    Count tokens in text using tiktoken for the specified model.
    
    Args:
        text: Input text to tokenize
        model_name: OpenAI model name (default: text-embedding-ada-002)
    
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def get_output_token_count(embedding_vector: List[float]) -> int:
    """
    Calculate output token count as the length of embedding vector array.
    
    Args:
        embedding_vector: The output embedding vector
    
    Returns:
        Length of the embedding vector (dimension count)
    """
    return len(embedding_vector) if embedding_vector else 0


class TelemetryLogger:
    """
    Handle telemetry logging for embedding and API usage.
    """
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize telemetry logger with database configuration.
        
        Args:
            db_config: Database connection parameters
        """
        self.db_config = db_config
    
    def _get_connection(self):
        """Create and return database connection."""
        return mysql.connector.connect(**self.db_config)
    
    def log_embedding_usage(
        self,
        app_id: int,
        model_id: int,
        input_text: str,
        output_vector: List[float],
        model_name: str = "text-embedding-ada-002"
    ) -> Optional[int]:
        """
        Log embedding usage to TBL_EMBEDDING_USAGE.
        
        Args:
            app_id: Application ID
            model_id: Model ID
            input_text: Input text that was embedded
            output_vector: Output embedding vector
            model_name: Model name for token counting
        
        Returns:
            EU_ID (embedding usage ID) if successful, None otherwise
        """
        connection = None
        cursor = None
        start_time = time.time()
        
        try:
            # Calculate token counts
            input_token_count = count_tokens(input_text, model_name)
            output_token_count = get_output_token_count(output_vector)
            
            # Serialize vector as blob
            vector_blob = pickle.dumps(output_vector)
            
            # Calculate timing
            end_time = time.time()
            
            connection = self._get_connection()
            cursor = connection.cursor()
            
            query = """
            INSERT INTO TBL_EMBEDDING_USAGE 
            (APP_ID, MODEL_ID, INPUT_TEXT, OUTPUT_VECTOR, 
             INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT, START_TIME, END_TIME)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            start_timestamp = datetime.fromtimestamp(start_time)
            end_timestamp = datetime.fromtimestamp(end_time)
            
            cursor.execute(query, (
                app_id,
                model_id,
                input_text,
                vector_blob,
                input_token_count,
                output_token_count,
                start_timestamp,
                end_timestamp
            ))
            
            connection.commit()
            eu_id = cursor.lastrowid
            
            return eu_id
            
        except Error as e:
            print(f"Error logging embedding usage: {e}")
            if connection:
                connection.rollback()
            return None
            
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def log_api_usage(
        self,
        eu_id: int,
        app_id: int,
        input_payload: Dict[str, Any],
        output_payload: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Optional[int]:
        """
        Log API usage to TBL_API_USAGE.
        
        Args:
            eu_id: Embedding usage ID (foreign key)
            app_id: Application ID
            input_payload: API request payload
            output_payload: API response payload
            start_time: Request start time (unix timestamp)
            end_time: Request end time (unix timestamp)
        
        Returns:
            API_USAGE_ID if successful, None otherwise
        """
        connection = None
        cursor = None
        
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            query = """
            INSERT INTO TBL_API_USAGE 
            (EU_ID, APP_ID, INPUT_PAYLOAD, OUTPUT_PAYLOAD, START_TIME, END_TIME)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            start_timestamp = datetime.fromtimestamp(start_time) if start_time else datetime.now()
            end_timestamp = datetime.fromtimestamp(end_time) if end_time else datetime.now()
            
            cursor.execute(query, (
                eu_id,
                app_id,
                json.dumps(input_payload),
                json.dumps(output_payload) if output_payload else None,
                start_timestamp,
                end_timestamp
            ))
            
            connection.commit()
            api_usage_id = cursor.lastrowid
            
            return api_usage_id
            
        except Error as e:
            print(f"Error logging API usage: {e}")
            if connection:
                connection.rollback()
            return None
            
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()


# Convenience functions for standalone usage
def log_embedding_telemetry(
    db_config: Dict[str, Any],
    app_id: int,
    model_id: int,
    input_text: str,
    output_vector: List[float],
    model_name: str = "text-embedding-ada-002"
) -> Optional[int]:
    """
    Standalone function to log embedding telemetry.
    
    Returns:
        EU_ID if successful, None otherwise
    """
    logger = TelemetryLogger(db_config)
    return logger.log_embedding_usage(app_id, model_id, input_text, output_vector, model_name)
