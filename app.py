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

# Konfiqurasiyanın Environment Variables-dan oxunması
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
if not BOOTSTRAP_SERVERS:
    raise ValueError("KAFKA_BOOTSTRAP_SERVERS environment dəyişəni təyin edilməyib!")

INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "pin-image-process-topic")
OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "pin-image-embedding-topic")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "pin-vectorizer-group")
MODEL_NAME = os.getenv("MODEL_NAME", "google/siglip-so400m-patch14-384")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_DOWNLOAD_THREADS = int(os.getenv("NUM_DOWNLOAD_THREADS", "8"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

logger.info(f"Model yüklənir: {MODEL_NAME} (Device: {device}, Dtype: {dtype})")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
model.eval()

consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    group_id=CONSUMER_GROUP,
    auto_offset_reset="latest",
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
    url = item.get("mediaUrl")
    try:
        res = requests.get(url, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        img = Image.open(io.BytesIO(res.content)).convert("RGB")
        return {"item": item, "image": img, "error": None}
    except Exception as e:
        return {"item": item, "image": None, "error": str(e)}


logger.info(f"Worker başladıldı. Dinlənilir: {INPUT_TOPIC}")

while True:
    records_dict = consumer.poll(timeout_ms=1000, max_records=BATCH_SIZE)
    if not records_dict:
        continue

    batch_items = [msg.value for partition_msgs in records_dict.values() for msg in partition_msgs]
    if not batch_items:
        continue

    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_THREADS) as executor:
        download_results = list(executor.map(fetch_image, batch_items))

    valid_results = [r for r in download_results if r["image"] is not None]

    for r in download_results:
        if r["error"]:
            logger.error(f"Şəkil yüklənmədi (pinId: {r['item'].get('pinId')}): {r['error']}")

    if not valid_results:
        consumer.commit()
        continue

    images = [r["image"] for r in valid_results]
    inputs = processor(images=images, return_tensors="pt").to(device)
    if device == "cuda":
        inputs = {k: v.to(dtype=dtype) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        embeddings = features.to(torch.float32).cpu().tolist()

    for r, embedding in zip(valid_results, embeddings):
        item = r["item"]
        payload = {
            "pinId": item.get("pinId"),
            "mediaUrl": item.get("mediaUrl"),
            "mediaWidth": item.get("mediaWidth"),
            "mediaHeight": item.get("mediaHeight"),
            "embedding": embedding,
        }
        producer.send(OUTPUT_TOPIC, key=str(item.get("pinId")).encode("utf-8"), value=payload)

    producer.flush()
    consumer.commit()
    logger.info(f"{len(valid_results)} ədəd şəkil batch şəklində emal edildi.")