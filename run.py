import os

import uvicorn

from sources.Controllers.config import PORT

if __name__ == "__main__":
    reload = os.environ.get("ENV", "production") != "production"
    uvicorn.run(
        "sources:app",
        host="0.0.0.0",
        port=int(PORT),
        reload=reload,
        workers=1,
    )
