from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.usage_tracker import get_usage_stats

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("/stats")
def usage_stats(db: Session = Depends(get_db)) -> dict:
    return get_usage_stats(db)
