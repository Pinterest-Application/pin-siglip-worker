import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import requests
import torch
from PIL import Image
from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load configurations from environment variables
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
if not BOOTSTRAP_SERVERS:
    raise ValueError("KAFKA_BOOTSTRAP_SERVERS environment variable is not set!")

INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "pin-image-process-topic")
OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "pin-image-embedding-topic")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "pin-vectorizer-group")
MODEL_NAME = os.getenv("MODEL_NAME", "google/siglip-so400m-patch14-384")
AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_DOWNLOAD_THREADS = int(os.getenv("NUM_DOWNLOAD_THREADS", "8"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

logger.info(f"Loading model: {MODEL_NAME} (Device: {device}, Dtype: {dtype})")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
model.eval()

consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    group_id=CONSUMER_GROUP,
    auto_offset_reset=AUTO_OFFSET_RESET,
    enable_auto_commit=False,
    max_poll_records=BATCH_SIZE,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    acks="all",
    retries=3,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def fetch_image(item):
    pin_id = item.get("pinId")
    url = item.get("mediaUrl")
    try:
        logger.info(f"Fetching image from SeaweedFS for pinId: {pin_id}")
        res = requests.get(url, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        img = Image.open(io.BytesIO(res.content)).convert("RGB")
        return {"item": item, "image": img, "error": None}
    except Exception as e:
        return {"item": item, "image": None, "error": str(e)}


logger.info(f"Worker successfully started. Listening on topic: {INPUT_TOPIC}")

while True:
    records_dict = consumer.poll(timeout_ms=1000, max_records=BATCH_SIZE)
    if not records_dict:
        continue

    batch_items = [msg.value for partition_msgs in records_dict.values() for msg in partition_msgs]
    if not batch_items:
        continue

    batch_pin_ids = [item.get("pinId") for item in batch_items]
    logger.info(f"Received batch of {len(batch_items)} record(s). Target pinIds: {batch_pin_ids}")

    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_THREADS) as executor:
        download_results = list(executor.map(fetch_image, batch_items))

    valid_results = [r for r in download_results if r["image"] is not None]

    for r in download_results:
        if r["error"]:
            logger.error(f"Image download failed for pinId: {r['item'].get('pinId')} | Reason: {r['error']}")

    if not valid_results:
        logger.warning(f"No downloadable images in batch. Skipping pinIds: {batch_pin_ids}")
        consumer.commit()
        continue

    active_pin_ids = [r["item"].get("pinId") for r in valid_results]
    logger.info(f"Computing embeddings for {len(active_pin_ids)} image(s). Active pinIds: {active_pin_ids}")

    images = [r["image"] for r in valid_results]
    inputs = processor(images=images, return_tensors="pt").to(device)
    if device == "cuda":
        inputs = {k: v.to(dtype=dtype) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            features = features.pooler_output

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        embeddings = features.to(torch.float32).cpu().tolist()

    for r, embedding in zip(valid_results, embeddings):
        item = r["item"]
        pin_id = item.get("pinId")
        payload = {
            "pinId": pin_id,
            "mediaUrl": item.get("mediaUrl"),
            "mediaWidth": item.get("mediaWidth"),
            "mediaHeight": item.get("mediaHeight"),
            "embedding": embedding,
        }
        producer.send(OUTPUT_TOPIC, key=str(pin_id).encode("utf-8"), value=payload)
        logger.info(f"Published 1152-d embedding to topic '{OUTPUT_TOPIC}' for pinId: {pin_id}")

    producer.flush()
    consumer.commit()
    logger.info(f"Successfully processed batch and committed offset for pinIds: {active_pin_ids}")