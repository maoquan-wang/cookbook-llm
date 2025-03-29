import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


async def run_shell(
    cmd: str | None,
    timeout: float | None = 5.0,  # seconds
):
    if cmd is None:
        return (0, "", "Command is None")

    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return (
            process.returncode or 0,
            stdout.decode(),
            stderr.decode(),
        )
    except asyncio.TimeoutError as e:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    finally:
        return (1, "", f"Command '{cmd}' timed out after {timeout} seconds")


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World."}


@app.post("/root/command")
async def root_command(request: Request):
    api_key = request.headers.get("api-key", None)
    if api_key not in ["rootagent"]:
        return {}

    body = await request.json()
    stream = body.get("stream", False)
    command = body.get("command", None)

    if stream:
        return StreamingResponse(
            iter([f"chunk1 {command}", f"chunk2 {command}"]),
            media_type="text/plain",
        )
    else:
        _, stdout, stderr = await run_shell(command)
        return {"cmd": command, "message": stdout + stderr}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
