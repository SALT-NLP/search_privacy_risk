from typing import Dict, Optional, Any, Set, List, Union
from fastapi import FastAPI, HTTPException
import uvicorn
import uuid
import json
import os
import logging
from utils import unified_call, unified_call_inproc
from pydantic import BaseModel
from functools import wraps
from datetime import datetime
import asyncio
import aiofiles
from camel.toolkits.function_tool import get_openai_tool_schema

# Configure logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class User(BaseModel):
    id: str
    username: str
    password: str  # In practice, should be hashed
    auth_token: Optional[str]

class ActivityLog(BaseModel):
    id: str
    description: str
    app_name: str
    action: str
    timestamp: datetime
    user_id: str
    involved_user_ids: List[str]  # All users involved in this activity
    details: Optional[Dict[str, Any]] = None

class AppClient:
    """
    Base class for interacting with Applications. Used by both Agent and UserClient.
    Stores app connection info and auth tokens for each app.
    """
    def __init__(self):
        self.apps: Dict[str, Dict] = {}  # Store app connection info
        self.auth_tokens: Dict[str, str] = {}  # Store auth tokens for apps
    
    def register_app_info(self, app_name: str, host: str, port: int, auth_token: Optional[str] = None):
        """Register application connection information"""
        self.apps[app_name] = {"host": host, "port": port}
        if auth_token:
            self.auth_tokens[app_name] = auth_token

    def _get_app_name(self, app_name: str) -> Optional[str]:
        """
        Get the app name from self.apps using case-insensitive matching.
        Returns None if no match is found.
        """
        app_name_lower = app_name.lower()
        for registered_app in self.apps:
            if registered_app.lower() == app_name_lower:
                return registered_app
        return None

    async def call_app_function(self, app_name: str, function_name: str, **kwargs) -> Dict:
        """
        Unified function to call any application function.
        Args:
            app_name: Name of the registered application
            function_name: Name of the function to call
            **kwargs: Arguments to pass to the function
        """
        app_name = self._get_app_name(app_name)
        if not app_name:
            raise ValueError(f"App not registered")
            
        app_info = self.apps[app_name]
        base_url = f"http://{app_info['host']}:{app_info['port']}"
        
        # Add auth token if app is registered
        if app_name in self.auth_tokens:
            kwargs["token"] = self.auth_tokens[app_name]
            
        return await unified_call(base_url, function_name, **kwargs)

    async def _try_app_auth(self, app_name: str, user_id: str, username: str, password: str) -> Dict:
        """Helper method to try login or signup with an app"""
        result = await self.call_app_function(app_name, "login", username=username, password=password)
        # If login fails, 401 error, try signup
        if result["status"] == "error" and "401" in result["message"]:
            result = await self.call_app_function(app_name, "signup", user_id=user_id, username=username, password=password)
        return result
    
class InprocAppClient(AppClient):
    """
    In-process version of AppClient.
    """
    def __init__(self):
        super().__init__()
    
    def register_app_info(self, app_name: str, host: str, port: int, auth_token: Optional[str] = None):
        """Register application connection information"""
        self.apps[app_name] = {"host": host, "port": port}
        if auth_token:
            self.auth_tokens[app_name] = auth_token
        
    def call_app_function(self, app_name: str, function_name: str, **kwargs) -> Dict:
        """
        Unified function to call any application function.
        Args:
            app_name: Name of the registered application
            function_name: Name of the function to call
            **kwargs: Arguments to pass to the function
        """
        app_name = self._get_app_name(app_name)
        if not app_name:
            raise ValueError(f"App not registered")
            
        app_info = self.apps[app_name]
        base_url = f"http://{app_info['host']}:{app_info['port']}"
        
        # Add auth token if app is registered
        if app_name in self.auth_tokens:
            kwargs["token"] = self.auth_tokens[app_name]
            
        return unified_call_inproc(base_url, function_name, **kwargs)
    
    def _try_app_auth(self, app_name: str, user_id: str, username: str, password: str) -> Dict:
        """Helper method to try login or signup with an app"""
        result = self.call_app_function(app_name, "login", username=username, password=password)
        # If login fails, 401 error, try signup
        if result["status"] == "error" and "401" in result["message"]:
            result = self.call_app_function(app_name, "signup", user_id=user_id, username=username, password=password)
        return result


class Application:
    """
    Base class for all applications.
    Provides common functionality for user authentication, database management, and API specification.
    """
    def __init__(self, name: str, host: str, port: int, db_folder: Optional[str] = None):
        self.name = name
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.database: Dict[str, Any] = {}  # Empty database structure to be defined by subclasses
        self.db_lock = asyncio.Lock()
        
        # Routes to skip in API specification
        self.skip_routes: Set[str] = {
            "/openapi.json", "/docs", "/redoc", 
            "/signup", "/login", "/get_api_spec", "/get_new_activity",
            "/get_api_spec_in_openai_tool_schema", "/get_health"
        }
        
        # Set database file path based on db_folder
        if db_folder:
            os.makedirs(db_folder, exist_ok=True)
            self.db_file = os.path.join(db_folder, f"{name}_db.json")
        else:
            self.db_file = f"{name}_db.json"
        
        # Initialize database structure
        self._init_database()

        # The database should always have a "users" key
        if "users" not in self.database:
            raise HTTPException(status_code=500, detail="Database not properly initialized")

        # Ensure activity_logs exists
        if "activity_logs" not in self.database:
            self.database["activity_logs"] = []
        
        # Initialize FastAPI routes
        self._init_common_routes()
        self._init_app_specific_routes()
        self.load_from_db()

        print("###########################################################")
        print(f"Application {self.name} initialized at {self.host}:{self.port}")
        print(self.get_api_spec())
        print("###########################################################")
        print("OpenAI Tool Schema:")
        print(json.dumps(self.get_api_spec_in_openai_tool_schema(), indent=4))
        print("###########################################################")
        print()
    
    def _init_database(self):
        """
        Initialize the database structure.
        This method should be overridden by subclasses to define their specific database structure.
        """
        pass
    
    def _init_common_routes(self):
        """Initialize common routes for all applications"""
        @self.app.post("/signup")
        async def signup(user_id: str, username: str, password: str) -> Dict:
            r"""Sign up for an account.

            Args:
                user_id (str): The user ID
                username (str): The username
                password (str): The password

            Returns:
                user_id (str): The user ID
                auth_token (str): The auth token
            """                
            if any(user.username == username for user in self.database["users"].values()):
                raise HTTPException(status_code=400, detail="Username already exists")
            
            auth_token = str(uuid.uuid4())
            user = User(id=user_id, username=username, password=password, auth_token=auth_token)
            async with self.db_lock:
                self.database["users"][user_id] = user
                await self.save_to_db()
            
            return {"user_id": user_id, "auth_token": auth_token}
            
        @self.app.post("/login")
        async def login(username: str, password: str) -> Dict:
            r"""Login to an account.

            Args:
                username (str): The username
                password (str): The password

            Returns:
                user_id (str): The user ID
                auth_token (str): The auth token
            """
            for user in self.database["users"].values():
                if user.username == username and user.password == password:
                    auth_token = str(uuid.uuid4())
                    async with self.db_lock:
                        user.auth_token = auth_token
                        await self.save_to_db()
                    return {"user_id": user.id, "auth_token": auth_token}
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        @self.app.get("/get_api_spec")
        async def get_api_spec() -> Dict:
            r"""Get the API specification of the application.

            Returns:
                api_spec (str): The API specification
            """
            return {"api_spec": self.get_api_spec()}
        
        @self.app.get("/get_api_spec_in_openai_tool_schema")
        async def get_api_spec_in_openai_tool_schema() -> Dict:
            r"""Get the API specification of the application in OpenAI tool schema format.

            Returns:
                api_spec (str): The API specification in OpenAI tool schema format
            """
            return {"api_spec": self.get_api_spec_in_openai_tool_schema()}

        @self.app.get("/get_new_activity")
        async def get_new_activity(token: str, since: Optional[datetime] = None) -> Dict:
            r"""Check if there is any new activity for the authenticated user since the given timestamp.

            Args:
                token (str): The auth token
                since (datetime, optional): The timestamp to check for new activity. If not provided, checks for any activity.

            Returns:
                has_new_activity (bool): Whether there is any new activity
                new_activity_descriptions (List[str]): The descriptions of the new activities
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")
                
            has_new_activity = False
            
            new_activity_descriptions = []
            for log in self.database["activity_logs"]:
                if user.id in log.involved_user_ids or "all" in log.involved_user_ids:
                    # Only consider activities for other users
                    if user.id != log.user_id:
                        if since is None or log.timestamp > since:
                            has_new_activity = True
                            new_activity_descriptions.append(log.description)
            return {
                "has_new_activity": has_new_activity,
                "new_activity_descriptions": new_activity_descriptions
            }

        @self.app.get("/get_health", status_code=204)
        async def get_health():
            return

    def _init_app_specific_routes(self):
        """
        Initialize application-specific routes.
        This method should be overridden by subclasses.
        """
        pass
    
    def _get_user_by_token(self, token: str) -> Optional[User]:
        """Get a user by their auth token"""
        if "users" in self.database:
            for user_id, user in self.database["users"].items():
                if user.auth_token == token:
                    return user
        return None
    
    async def save_to_db(self):
        """Save the database to a file asynchronously"""
        data = {}
        for key, value in self.database.items():
            if isinstance(value, dict):
                # Handle dictionaries of objects (like users)
                data[key] = {k: v.dict() if hasattr(v, 'dict') else v for k, v in value.items()}
            elif isinstance(value, list):
                # Handle lists of objects (like messages)
                data[key] = [item.dict() if hasattr(item, 'dict') else item for item in value]
            else:
                # Handle other types
                data[key] = value
        
        async with aiofiles.open(self.db_file, 'w') as f:
            await f.write(json.dumps(data, default=str))
    
    def load_from_db(self):
        """
        Load the database from a file.
        This is a base implementation that should be overridden by subclasses.
        """
        try:
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                self._load_database_from_data(data)
        except FileNotFoundError:
            pass
    
    def _load_database_from_data(self, data: Dict[str, Any]):
        """
        Load database from the parsed JSON data.
        This method should be overridden by subclasses to handle their specific database structure.
        """
        pass
    
    def run(self):
        """Run the application server"""
        uvicorn.run(self.app, host=self.host, port=self.port)

    def get_api_spec(self) -> str:
        """Returns a string describing the API specification of this Application instance"""
        # Build the specification from registered routes
        routes_spec = []
        for route in self.app.routes:
            # Skip routes that should be excluded from the API spec
            if route.path in self.skip_routes or "/docs" in route.path:
                continue
            
            # Get endpoint parameters from the route's endpoint function signature
            docstring = ""
            if hasattr(route, 'endpoint'):
                docstring = route.endpoint.__doc__ or ""
                docstring = "\n".join([item.strip() for item in docstring.split("\n") if not item.strip().startswith("token (str):")])
            routes_spec.append(f"{len(routes_spec) + 1}. {route.path.strip('/')}\n"
                                f"Description: {docstring.strip()}\n")
        
        spec = f"""
### APPLICATION_NAME: {self.name}

Available Functions:
{chr(10).join(routes_spec)}
"""
        return spec.strip()
    
    def get_api_spec_in_openai_tool_schema(self) -> List[Dict]:
        routes_spec = []
        for route in self.app.routes:
            # Skip routes that should be excluded from the API spec
            if route.path in self.skip_routes or "/docs" in route.path:
                continue

            current_schema = get_openai_tool_schema(route.endpoint)
            if 'token' in current_schema['function']['parameters']['properties']:
                del current_schema['function']['parameters']['properties']['token']
            current_schema['function']['parameters']['required'] = [param for param in current_schema['function']['parameters']['required'] if param != 'token']
            
            current_schema['function']['name'] = self.name + "_" + current_schema['function']['name']
            routes_spec.append(current_schema)
        
        routes_spec = {schema['function']['name']: schema for schema in routes_spec}

        return routes_spec


    def activity_logger(self, action_name: str, involved_user_keys: Optional[List[str]] = None):
        """
        Decorator for automatically logging activities with FastAPI routes.
        
        Args:
            action_name: Name of the action being performed
            involved_user_keys: List of keys in the result dict that contain user IDs to include
                                as involved users (in addition to the authenticated user)
        
        Usage:
            @app.activity_logger("post_message", involved_user_keys=["receiver_id"])
            @app.app.post("/post_message")
            async def post_message(token: str, ...):
                # Function implementation
        """
        def decorator(func):
            # For FastAPI routes, we need to preserve the async nature
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    # Extract token and get user
                    token = kwargs.get('token')
                    user = None
                    if token:
                        user = self._get_user_by_token(token)
                    
                    # Call the original async function
                    result = await func(*args, **kwargs)
                    
                    # Other users involved in the activity, might contain "all"
                    involved_users = []
                    
                    # Add additional involved users based on the specified keys in the result
                    if isinstance(result, dict) and involved_user_keys:
                        for key in involved_user_keys:
                            if key in result and result[key]:
                                user_ids = result[key]
                                # Handle both single user IDs and lists of user IDs
                                if isinstance(user_ids, list):
                                    involved_users.extend([uid for uid in user_ids if uid])
                                else:
                                    involved_users.append(user_ids)
                    
                    # Remove duplicates while preserving order
                    involved_users = list(dict.fromkeys(involved_users))
                    
                    # Log the activity
                    if involved_users:
                        # Create a copy of kwargs without the token for security
                        safe_kwargs = {k: v for k, v in kwargs.items() if k != 'token'}
                        async with self.db_lock:
                            log = ActivityLog(
                                id=result.get("activity_id", str(uuid.uuid4())),
                                description=result.get("activity_description", ""),
                                app_name=self.name,
                                action=action_name,
                                timestamp=datetime.now(),
                                user_id=user.id,
                                involved_user_ids=involved_users,
                                details=safe_kwargs
                            )
                            self.database["activity_logs"].append(log)
                            await self.save_to_db()
                    
                    return result
                return async_wrapper
            else:
                raise ValueError("Activity logger can only be used with async functions")
        return decorator