import os
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

BATCH_SIZE = 50  # Number of files to delete per batch
counter = 0  # Shared counter
lock = asyncio.Lock()  # Lock to manage counter updates safely

async def delete_file(file, total_files):
    """Deletes a single file asynchronously and updates the counter."""
    global counter
    await asyncio.to_thread(file.delete)  # Offload blocking delete operation

    async with lock:  # Ensure safe counter update
        counter += 1
        print(f"Deleted {counter}/{total_files} files: {file.name}")

async def batch_delete(files):
    """Deletes files in batches with parallel execution."""
    total_files = len(files)
    tasks = set()  # Track running tasks

    for file in files:
        task = asyncio.create_task(delete_file(file, total_files))
        tasks.add(task)

        if len(tasks) >= BATCH_SIZE:  # Wait when batch limit is reached
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Ensure remaining tasks complete
    if tasks:
        await asyncio.gather(*tasks)

async def main():
    """Main function to list and delete files in batches."""
    files = list(genai.list_files())  # Fetch all files
    total_files = len(files)

    print(f"Total files found: {total_files}")

    if not files:
        print("No files to delete.")
        return

    print("Starting batch deletion...")
    await batch_delete(files)  # Delete files asynchronously
    print("All files deleted.")

    remaining_files = list(genai.list_files())  # Check remaining files
    print(f"Remaining files: {len(remaining_files)}")

asyncio.run(main())
