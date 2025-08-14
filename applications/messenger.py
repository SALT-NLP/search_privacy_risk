from typing import Dict, Optional, Any
from fastapi import HTTPException
import uuid
import logging
from applications.base import Application, User
from pydantic import BaseModel
from datetime import datetime
from applications.utils import string_contains

# Configure logging similarly to other applications
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Message(BaseModel):
    id: str
    sender_id: str
    recipient_id: str
    time: str  # Time in format "YYYY-MM-DD hh:mm:ss"
    message: str

class Messenger(Application):
    """
    Messenger application that provides messaging functionalities for sending and receiving text messages.
    """
    def __init__(self, name: str, host: str, port: int, db_folder: Optional[str] = None):
        super().__init__(name, host, port, db_folder)

    def _init_database(self):
        """
        Initialize the Messenger-specific database.
        The database contains the "users" collection (inherited from the base signup/login routes)
        and a "messages" collection for storing sent messages.
        """
        self.database = {
            "users": {},
            "messages": {}
        }

    def _load_database_from_data(self, data: Dict[str, Any]):
        """
        Load the Messenger-specific database from persisted JSON data.
        """
        if "users" in data:
            self.database["users"] = {uid: User(**user_data) for uid, user_data in data["users"].items()}
        if "messages" in data:
            self.database["messages"] = {msg_id: Message(**msg_data) for msg_id, msg_data in data["messages"].items()}

    def _init_app_specific_routes(self):
        """
        Initialize routes specific to the Messenger application based on the provided API specification.
        """

        @self.app.post("/send_message")
        @self.activity_logger("send_message", involved_user_keys=["receiver_id"])
        async def send_message(token: str, recipient_id: str, message: str) -> Dict:
            """
            Sends a text message to a user.
            
            Args:
                token (str): The auth token of the sender.
                recipient_id (str): The user ID of the recipient.
                message (str): The content of the message.
            
            Returns:
                success (bool): Whether the operation was successful.
            """
            sender = self._get_user_by_token(token)
            if not sender:
                raise HTTPException(status_code=401, detail="Invalid token")
            if not recipient_id or not recipient_id.strip():
                raise HTTPException(
                    status_code=400,
                    detail="InvalidRequestException: 'recipient_id' is invalid or missing."
                )
            if not message or not message.strip():
                raise HTTPException(
                    status_code=400,
                    detail="InvalidRequestException: 'message' is invalid or missing."
                )
            if recipient_id not in self.database["users"]:
                raise HTTPException(
                    status_code=400,
                    detail="InvalidRequestException: Recipient does not exist."
                )

            message_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_message = Message(
                id=message_id,
                sender_id=sender.id,
                recipient_id=recipient_id,
                time=timestamp,
                message=message
            )
            async with self.db_lock:
                self.database["messages"][message_id] = new_message
                await self.save_to_db()

            return {"success": True, "receiver_id": recipient_id, "activity_id": message_id, "activity_description": f"New message, message_id: {message_id}"}

        @self.app.get("/get_messages")
        async def get_messages(
            token: str,
            max_count: Optional[int] = 10,
            keyword: Optional[str] = None,
            sender_id: Optional[str] = None
        ) -> Dict:
            """
            Get recent text messages, with optional filtering by keyword and sender_id.
            If an argument is not provided, that filter is not applied.
            
            Args:
                token (str): The auth token of the user.
                max_count (int, optional): The maximum number of messages to return. Default is 10.
                keyword (str, optional): The keyword to filter messages. Default is None.
                sender_id (str, optional): The user id of the sender. Default is None.

            Returns:
                messages (List[Dict]): A list of objects, each containing:
                  - message_id (str): The id of the message.
                  - sender_id (str): The id of the sender.
                  - time (str): The time of the message (formatted as YYYY-MM-DD hh:mm:ss).
                  - message (str): The content of the message.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")

            if not isinstance(max_count, int) or max_count < 1:
                max_count = 10
            if keyword in ('None', 'null', ''):
                keyword = None
            if sender_id in ('None', 'null', ''):
                sender_id = None
            
            sorted_messages = sorted(
                self.database["messages"].values(),
                key=lambda msg: datetime.strptime(msg.time, "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )

            matching_messages = []
            for msg in sorted_messages:
                if user.id == msg.sender_id or user.id == msg.recipient_id:
                    if keyword is not None and keyword.strip():
                        if not string_contains(msg.message.lower(), keyword.lower()):
                            continue
                    if sender_id is not None and msg.sender_id != sender_id:
                        continue
                    matching_messages.append({
                        "message_id": msg.id,
                        "sender_id": msg.sender_id,
                        "time": msg.time,
                        "message": msg.message
                    })
            matching_messages = matching_messages[:max_count]
            return {"messages": matching_messages}