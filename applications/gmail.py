import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List
from fastapi import HTTPException
from pydantic import BaseModel
from applications.base import Application, User
from applications.utils import string_contains

# Configure logging similarly to other applications
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define an Email model for the Gmail application.
class Email(BaseModel):
    id: str
    from_address: str
    to_address: str
    cc_address: Optional[str] = None
    bcc_address: Optional[str] = None
    time: str  # In format "YYYY-MM-DD HH:mm:ss"
    subject: str
    body: str

class Gmail(Application):
    """
    Gmail application that provides functionalities to send an email,
    search for emails, and read a specific email.
    """
    def _init_database(self):
        # Initialize the Gmail database structure with "users" and "emails"
        self.database = {
            "users": {},
            "emails": {}
        }

    def _load_database_from_data(self, data: Dict[str, Any]):
        # Load users if present.
        if "users" in data:
            self.database["users"] = {
                uid: User(**user_data) for uid, user_data in data["users"].items()
            }
        # Load emails if present.
        if "emails" in data:
            self.database["emails"] = {
                eid: Email(**email_data) for eid, email_data in data["emails"].items()
            }

    def _init_app_specific_routes(self):
        def is_valid_email(email: str) -> bool:
            email = email.strip()
            return ("@" in email) and email.endswith("gmail.com")

        def parse_time(time_str: str) -> str:
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="InvalidRequestException: 'time' is not in the correct format (YYYY-MM-DD hh:mm:ss)"
                )

        @self.app.post("/send_email")
        @self.activity_logger("send_email", involved_user_keys=["receiver_id"])
        async def send_email(
            token: str,
            to_address: str,
            subject: str,
            body: str,
            cc_address: Optional[str] = None,
            bcc_address: Optional[str] = None
        ) -> Dict:
            """
            Send an email to one or multiple recipients, with optional scheduling, cc, and bcc.

            Args:
                token (str): The auth token.
                to_address (str): A string of email addresses to send to, separated by commas.
                cc_address (str, optional): A string of cc'ed email addresses, separated by commas. Default is None.
                bcc_address (str, optional): A string of bcc'ed email addresses, separated by commas. Default is None.
                subject (str): The email subject.
                body (str): The email body.

            Returns:
                success (bool): Whether the email was sent successfully.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")

            if not to_address.strip():
                raise HTTPException(status_code=400, detail="InvalidRequestException: 'to_address' field must not be empty")
            if not subject.strip():
                raise HTTPException(status_code=400, detail="InvalidRequestException: 'subject' field must not be empty")
            if not body.strip():
                raise HTTPException(status_code=400, detail="InvalidRequestException: 'body' field must not be empty")

            for field_value, field_name in [(to_address, "to_address"), (cc_address, "cc_address"), (bcc_address, "bcc_address")]:
                if field_value and field_value.strip():
                    emails = [addr.strip() for addr in field_value.split(",") if addr.strip()]
                    for addr in emails:
                        if not is_valid_email(addr):
                            raise HTTPException(
                                status_code=400,
                                detail=f"InvalidRequestException: The provided email address in '{field_name}' is malformed"
                            )

            send_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            sender_email = user.username.lower().replace(" ", "_") + "@gmail.com"
            from_address = sender_email

            email_id = str(uuid.uuid4())
            new_email = Email(
                id=email_id,
                from_address=from_address,
                to_address=to_address,
                cc_address=cc_address,
                bcc_address=bcc_address,
                time=send_time,
                subject=subject,
                body=body
            )
            async with self.db_lock:
                self.database["emails"][email_id] = new_email
                await self.save_to_db()

            receiver_ids = [addr.split("@")[0] for addr in to_address.split(",")]

            return {"success": True, "receiver_id": receiver_ids, "activity_id": email_id, "activity_description": f"New email, email_id: {email_id}"}

        @self.app.get("/search_emails")
        async def search_emails(
            token: str,
            max_count: Optional[int] = 10,
            keyword: Optional[str] = None,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            from_address: Optional[str] = None,
            to_address: Optional[str] = None
        ) -> Dict:
            """
            Search for emails with optional filtering by keyword, time range, or sender and recipient.
            If an argument is not provided, that filter is not applied.

            Args:
                token (str): The auth token.
                max_count (int, optional): Maximum number of emails to retrieve. Default is 10.
                keyword (str, optional): A keyword to search for in subject and body. Default is None.
                start_time (str, optional): Start time (YYYY-MM-DD hh:mm:ss). Default is None.
                end_time (str, optional): End time (YYYY-MM-DD hh:mm:ss). Default is None.
                from_address (str, optional): Filter by sender's email address. Default is None.
                to_address (str, optional): Filter by recipient's email address. Default is None.

            Returns:
                emails (List[Dict]): A list of matching emails containing 'id', 'subject', 'from_address', 'to_address',
                                      and 'time' (formatted as 'YYYY-MM-DD HH:mm').
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")

            if not isinstance(max_count, int) or max_count < 1:
                max_count = 10
            if keyword in ('None', 'null', ''):
                keyword = None
            if start_time in ('None', 'null', ''):
                start_time = None
            if end_time in ('None', 'null', ''):
                end_time = None
            if from_address in ('None', 'null', ''):
                from_address = None
            if to_address in ('None', 'null', ''):
                to_address = None

            start_dt = None
            end_dt = None
            if start_time:
                try:
                    start_dt = datetime.strptime(start_time.strip(), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    start_dt = None
            if end_time:
                try:
                    end_dt = datetime.strptime(end_time.strip(), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    end_dt = None

            if from_address:
                if not is_valid_email(from_address):
                    raise HTTPException(
                        status_code=400,
                        detail="InvalidRequestException: The provided email address in 'from_address' is malformed"
                    )
            if to_address:
                if not is_valid_email(to_address):
                    raise HTTPException(
                        status_code=400,
                        detail="InvalidRequestException: The provided email address in 'to_address' is malformed"
                    )

            current_email = user.username.lower().replace(" ", "_") + "@gmail.com"

            def user_can_access(email_obj: Email) -> bool:
                if email_obj.from_address.lower() == current_email:
                    return True
                recipients = [addr.strip().lower() for addr in email_obj.to_address.split(",") if addr.strip()]
                cc_list = [addr.strip().lower() for addr in email_obj.cc_address.split(",") if addr.strip()] if email_obj.cc_address else []
                bcc_list = [addr.strip().lower() for addr in email_obj.bcc_address.split(",") if addr.strip()] if email_obj.bcc_address else []
                return current_email in recipients or current_email in cc_list or current_email in bcc_list

            sorted_emails = sorted(
                self.database["emails"].values(),
                key=lambda email: datetime.strptime(email.time, "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )

            matched_emails = []
            for email_obj in sorted_emails:
                if not user_can_access(email_obj):
                    continue

                if keyword is not None and keyword.strip():
                    if not string_contains(email_obj.subject.lower(), keyword.lower()) and not string_contains(email_obj.body.lower(), keyword.lower()):
                        continue

                try:
                    email_dt = datetime.strptime(email_obj.time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if start_dt and email_dt < start_dt:
                    continue
                if end_dt and email_dt > end_dt:
                    continue

                if from_address and email_obj.from_address.lower() != from_address.lower():
                    continue

                if to_address:
                    recipient_list = [addr.strip().lower() for addr in email_obj.to_address.split(",") if addr.strip()]
                    if to_address.lower() not in recipient_list:
                        continue

                matched_emails.append(email_obj)

            matched_emails = matched_emails[:max_count]

            results = []
            for email_obj in matched_emails:
                try:
                    email_dt = datetime.strptime(email_obj.time, "%Y-%m-%d %H:%M:%S")
                    formatted_time = email_dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    formatted_time = email_obj.time
                results.append({
                    "id": email_obj.id,
                    "subject": email_obj.subject,
                    "from_address": email_obj.from_address,
                    "to_address": email_obj.to_address,
                    "time": formatted_time
                })

            return {"emails": results}

        @self.app.get("/get_email")
        async def get_email(token: str, email_id: str) -> Dict:
            """
            Read the content of an email.

            Args:
                token (str): The auth token.
                email_id (str): The unique identifier of the email.

            Returns:
                A dictionary containing the email's 'from_address', 'to_address', 'cc_address', 'bcc_address', 'time', 'subject', and 'body'.
            """
            user = self._get_user_by_token(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")

            if email_id not in self.database["emails"]:
                raise HTTPException(status_code=404, detail="NotFoundException: The 'email_id' is not found.")

            email_obj = self.database["emails"][email_id]

            current_email = user.username.lower().replace(" ", "_") + "@gmail.com"
            access = False
            if email_obj.from_address.lower() == current_email:
                access = True
            else:
                recipients = [addr.strip().lower() for addr in email_obj.to_address.split(",") if addr.strip()]
                cc_list = [addr.strip().lower() for addr in email_obj.cc_address.split(",") if addr.strip()] if email_obj.cc_address else []
                bcc_list = [addr.strip().lower() for addr in email_obj.bcc_address.split(",") if addr.strip()] if email_obj.bcc_address else []
                if current_email in recipients or current_email in cc_list or current_email in bcc_list:
                    access = True

            if not access:
                raise HTTPException(status_code=403, detail="Forbidden: You do not have access to view this email")

            return email_obj.dict()