from applications import Facebook, Notion, Gmail, Messenger
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument("--host", default="localhost", help="Host to run on")
    parser.add_argument("--port", type=int, help="Port to run on")
    
    # App-specific arguments
    parser.add_argument("--app-name", choices=["Facebook", "Notion", "Gmail", "Messenger"], help="Name of the application")
    parser.add_argument("--db-folder", help="Folder to store database files")

    args = parser.parse_args()

    # Create the appropriate application based on the app type
    if args.app_name == "Facebook":
        app = Facebook(args.app_name, args.host, args.port, args.db_folder)
    elif args.app_name == "Notion":
        app = Notion(args.app_name, args.host, args.port, args.db_folder)
    elif args.app_name == "Gmail":
        app = Gmail(args.app_name, args.host, args.port, args.db_folder)
    elif args.app_name == "Messenger":
        app = Messenger(args.app_name, args.host, args.port, args.db_folder)
    else:
        raise ValueError(f"Unknown application name: {args.app_name}")
    app.run()