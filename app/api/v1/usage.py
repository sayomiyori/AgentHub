from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.platform.resource_access import read_scope
from app.services.usage_tracker import get_usage_stats

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("/stats")
def usage_stats(tenant_id: UUID | None = Depends(read_scope), db: Session = Depends(get_db)) -> dict:
    return get_usage_stats(db, tenant_id=tenant_id)
