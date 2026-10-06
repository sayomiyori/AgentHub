import logging
import time

from app.platform.replies import recover_replies_once


def main() -> None:
    while True:
        try:
            recover_replies_once()
        except Exception:
            logging.getLogger(__name__).warning("Telegram answer recovery unavailable")
        time.sleep(5)


if __name__ == "__main__":
    main()
