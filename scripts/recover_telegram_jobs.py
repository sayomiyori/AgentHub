import logging
import time

from app.platform.processor import recover_jobs_once


def main() -> None:
    while True:
        try:
            recover_jobs_once()
        except Exception:
            logging.getLogger(__name__).warning("Telegram job recovery unavailable")
        time.sleep(5)


if __name__ == "__main__":
    main()
