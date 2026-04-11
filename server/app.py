import os 
import uvicorn 
from env_server import app

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
