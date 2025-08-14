# facebook.py
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

class Post(BaseModel):
    id: str
    content: str
    sender: str
    time: str  # In the format YYYY-MM-DD hh:mm:ss

class Facebook(Application):
    """
    Facebook application that provides functionalities for creating a new public post,
    searching posts by keyword for a specified user, and reading recent public posts.
    """
    def __init__(self, name: str, host: str, port: int, db_folder: Optional[str] = None):
        super().__init__(name, host, port, db_folder)
    
    def _init_database(self):
        """
        Initialize the Facebook-specific database with a structure for users and posts.
        """
        self.database = {
            "users": {},
            "posts": {}
        }
    
    def _load_database_from_data(self, data: Dict[str, Any]):
        """
        Load Facebook-specific database from the parsed JSON data.
        """
        if "users" in data:
            self.database["users"] = {
                uid: User(**user_data) for uid, user_data in data["users"].items()
            }
        if "posts" in data:
            self.database["posts"] = {
                pid: Post(**post_data) for pid, post_data in data["posts"].items()
            }
    
    def _init_app_specific_routes(self):
        """
        Initialize routes specific to the Facebook application.
        """

        @self.app.post("/create_post")
        @self.activity_logger("create_post", involved_user_keys=["receiver_id"])
        async def create_post(token: str, content: str) -> Dict:
            """
            Create a new public post.
            
            Args:
                token (str): The auth token of the user.
                content (str): The content of the post.
            
            Returns:
                success (bool): Whether the operation was successful.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")
            if not content.strip():
                raise HTTPException(status_code=400, detail="InvalidRequestException: 'content' must not be empty")
            
            post_id = str(uuid.uuid4())
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_post = Post(id=post_id, content=content, sender=user.id, time=now)
            async with self.db_lock:
                self.database["posts"][post_id] = new_post
                await self.save_to_db()
            
            return {"success": True, "receiver_id": "all", "activity_id": post_id, "activity_description": f"New post, post_id: {post_id}"}

        @self.app.get("/get_posts")
        async def get_posts(
            max_count: Optional[int] = 10,
            keyword: Optional[str] = None,
            sender_id: Optional[str] = None
        ) -> Dict:
            """
            Get recent public posts, with optional filtering by keyword and sender_id.
            If an argument is not provided, that filter is not applied.
            
            Args:
                max_count (int, optional): The maximum number of posts to retrieve. Default is 10.
                keyword (str, optional): The keyword to filter posts. Default is None.
                sender_id (str, optional): The sender id of the posts to retrieve. Default is None.

            Returns:
                posts (List[Dict]): A list of objects, each containing:
                  - post_id (str): The id of the post.
                  - sender (str): The id of the sender.
                  - time (str): The time of the post (formatted as YYYY-MM-DD hh:mm:ss).
                  - content (str): The content of the post.
            """
            if not isinstance(max_count, int) or max_count < 1:
                max_count = 10
            if keyword in ('None', 'null', ''):
                keyword = None
            if sender_id in ('None', 'null', ''):
                sender_id = None

            sorted_posts = sorted(
                self.database["posts"].values(),
                key=lambda post: datetime.strptime(post.time, "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )

            matching_posts = []
            for post in sorted_posts:
                if sender_id is not None and post.sender != sender_id:
                    continue
                if keyword is not None and keyword.strip():
                    if not string_contains(post.content.lower(), keyword.lower()):
                        continue
                matching_posts.append({
                    "post_id": post.id,
                    "sender": post.sender,
                    "time": post.time,
                    "content": post.content
                })
            matching_posts = matching_posts[:max_count]
            return {"posts": matching_posts}