from enum import Enum

class AlertType(Enum):
    INCREASE = "increase"
    DROP = "drop"
    SPIKE = "spike"
    NONE = "none"
    DUMMY = "dummy"
    

class EventType(Enum):
    QUERY_REWRITE_DONE = "query_rewrite_done"
    RETRIEVAL_DONE = "retrieval_done"
    GENERATION_DONE = "generation_done"
