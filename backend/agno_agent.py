"""Backward-compatible Agno example script.

This file intentionally focuses on the browser-extension case, but the real reusable runtime
lives in agno_runtime.py. That file can be used directly outside the browser-extension flow too.
"""

import asyncio
import os

from agno_runtime import run_prompt

ORIGIN = os.getenv("BROWSER_ORIGIN", "https://example.com")


async def demo_browser_extension_flow():
   prompt = (
       "Calculate the value of 583 * 22, write down the result into a local variable, "
       "and then scroll down 400 pixels on my current active browser tab."
   )
   print(await run_prompt(prompt, origin=ORIGIN, include_browser_tools=True))


if __name__ == "__main__":
   asyncio.run(demo_browser_extension_flow())
