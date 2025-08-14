from typing import Dict, Optional, Any, List
from fastapi import HTTPException
import uuid
import logging
from applications.base import Application, User
from pydantic import BaseModel
from applications.utils import string_contains

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Page(BaseModel):
    id: str
    title: str
    content: str
    parent_page_id: Optional[str] = None
    owner: str
    shared_with: List[str] = []


class Notion(Application):
    """
    Notion application that provides page creation, reading, editing,
    sharing, and content searching functionalities.
    """
    def __init__(self, name: str, host: str, port: int, db_folder: Optional[str] = None):
        super().__init__(name, host, port, db_folder)

    def _init_database(self):
        """Initialize the Notion-specific database with users and pages."""
        self.database = {
            "users": {},
            "pages": {}
        }

    def _load_database_from_data(self, data: Dict[str, Any]):
        """Load Notion-specific database from the parsed JSON data."""
        if "users" in data:
            self.database["users"] = {
                uid: User(**user_data) for uid, user_data in data["users"].items()
            }
        if "pages" in data:
            self.database["pages"] = {
                page_id: Page(**page_data) for page_id, page_data in data["pages"].items()
            }

    def _init_app_specific_routes(self):
        """Initialize routes specific to the Notion application."""

        @self.app.post("/create_page")
        async def create_page(token: str, page_title: str, page_content: str, parent_page_id: Optional[str] = None) -> Dict:
            """
            Create a new page.
            
            Args:
                page_title (str): The title of the new page.
                page_content (str): The content of the new page.
                parent_page_id (str, optional): The id of the parent page. If provided, it must exist. (default: :obj:`None`)

            Returns:
                success (bool): Whether the operation was successful.
                page_id (str): The id of the created page.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")
            if not page_title.strip() or not page_content.strip():
                raise HTTPException(
                    status_code=400,
                    detail="InvalidRequestException: 'page_title' and 'page_content' must not be empty"
                )
            if parent_page_id:
                if parent_page_id not in self.database["pages"]:
                    raise HTTPException(
                        status_code=404,
                        detail="NotFoundException: The specified parent page does not exist"
                    )
            page_id = str(uuid.uuid4())
            new_page = Page(
                id=page_id,
                title=page_title,
                content=page_content,
                parent_page_id=parent_page_id,
                owner=user.id,
                shared_with=[]
            )
            async with self.db_lock:
                self.database["pages"][page_id] = new_page
                await self.save_to_db()
            return {"success": True, "page_id": page_id}

        @self.app.get("/get_page")
        async def get_page(token: str, page_id: str) -> Dict:
            """
            Read the content of a page.
            
            Args:
                page_id (str): The id of the page.
            
            Returns:
                page_content (str): The content of the page.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")
            if page_id not in self.database["pages"]:
                raise HTTPException(
                    status_code=404,
                    detail="NotFoundException: The specified page does not exist"
                )
            page = self.database["pages"][page_id]
            if page.owner != user.id and user.id not in page.shared_with:
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: You do not have access to view this page"
                )
            return {"page_content": page.content}

        @self.app.post("/edit_page")
        async def edit_page(token: str, page_id: str, new_content: str) -> Dict:
            """
            Edit an existing page.
            
            Args:
                page_id (str): The id of the page to be edited.
                new_content (str): The new content for the page.

            Returns:
                success (bool): Whether the operation was successful.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")
            if page_id not in self.database["pages"]:
                raise HTTPException(
                    status_code=404,
                    detail="NotFoundException: The specified page does not exist"
                )
            if not new_content.strip():
                raise HTTPException(
                    status_code=400,
                    detail="InvalidRequestException: 'new_content' must not be empty"
                )
            page = self.database["pages"][page_id]
            if page.owner != user.id:
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: You do not have permission to edit this page"
                )
            async with self.db_lock:
                page.content = new_content
                await self.save_to_db()
            return {"success": True}

        @self.app.post("/share_page")
        @self.activity_logger("share_page", involved_user_keys=["receiver_id"])
        async def share_page(token: str, page_id: str, recipient_id: str) -> Dict:
            """
            Share a page with another user.
            
            Args:
                page_id (str): The id of the page to be shared.
                recipient_id (str): The user ID to share the page with.

            Returns:
                success (bool): Whether the operation was successful.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")
            if page_id not in self.database["pages"]:
                raise HTTPException(
                    status_code=404,
                    detail="NotFoundException: The specified page does not exist"
                )
            if not recipient_id or not isinstance(recipient_id, str):
                raise HTTPException(
                    status_code=400,
                    detail="InvalidRequestException: 'recipient_id' must be a non-empty string"
                )
            
            if recipient_id not in self.database["users"]:
                raise HTTPException(
                    status_code=404,
                    detail=f"NotFoundException: User with ID {recipient_id} does not exist"
                )
            
            page = self.database["pages"][page_id]
            if page.owner != user.id:
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: You do not have permission to share this page"
                )
            
            async with self.db_lock:
                if recipient_id not in page.shared_with:
                    page.shared_with.append(recipient_id)
                await self.save_to_db()
            
            return {"success": True, "receiver_id": recipient_id, "activity_id": page_id, "activity_description": f"New shared page, page_id: {page_id}"}

        # TODO: improve this to use semantic search
        @self.app.get("/search_content")
        async def search_content(token: str,
                                 max_count: Optional[int] = 10,
                                 keyword: Optional[str] = None) -> Dict:
            """
            Search pages with optional filtering by keyword.
            If an argument is not provided, that filter is not applied.

            Args:
                max_count (int, optional): The maximum number of search results to return. Default is 10.
                keyword (str, optional): The keyword to search for. Default is None.

            Returns:
                results (List[Dict]): A list of objects, each containing:
                  - id (str): The id of the page.
                  - title (str): The title of the page.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            if not isinstance(max_count, int) or max_count < 1:
                max_count = 10
            if keyword in ('None', 'null', ''):
                keyword = None
            
            results = []

            for page in self.database["pages"].values():
                if page.owner == user.id or user.id in page.shared_with:
                    if keyword is None:
                        results.append({
                            "id": page.id,
                            "title": page.title
                        })
                    elif keyword.strip():
                        lower_keyword = keyword.lower()
                        if string_contains(page.title, lower_keyword) or string_contains(page.content, lower_keyword):
                            results.append({
                                "id": page.id,
                                "title": page.title
                            })

            results = results[:max_count]
            return {"results": results}