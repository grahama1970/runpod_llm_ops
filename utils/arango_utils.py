import asyncio
from loguru import logger
from verifaix.arangodb_helper.arango_client import connect_to_arango_client

async def truncate_cache_collection(arango_config, db=None):
    logger.info(f"Attempting to truncate cache collection '{arango_config['cache_collection_name']}'")

    if db is None:
        logger.info(f"Connecting to ArangoDB at {arango_config['host']}")
        db = await asyncio.to_thread(connect_to_arango_client, arango_config)

    collection_name = arango_config['cache_collection_name']

    # Check if the collection exists before attempting to truncate
    if db.has_collection(collection_name):
        collection = db.collection(collection_name)
        await asyncio.to_thread(collection.truncate)
        logger.info(f"Truncated cache collection '{collection_name}'")
    else:
        logger.warning(f"Collection '{collection_name}' does not exist. Skipping truncation.")